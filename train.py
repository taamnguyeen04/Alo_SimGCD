import argparse

import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups
from teacher import _build_teacher, _ema_update_teacher, _teacher_momentum
from pseudo import (CEView, audit_pseudo_samples, build_uq_to_true_label,
                    collect_novel_pseudo_from_unlabeled,
                    collect_pseudo_labels_from_unlabeled,
                    compute_unsupervised_class_scores, count_labeled_per_class,
                    count_pseudo_into, evaluate_train_labeled_per_class_accuracy,
                    log_labeled_distribution, log_pseudo_audit,
                    update_train_loader)


def train(student, train_loader, test_loader, unlabelled_train_loader, args):
    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )

    # ----------------------
    # ADAPART optimizers (mirrors BaCon: part queries + late backbone learnable,
    # gate on its own LR; prototypes EMA-only. Late backbone params live in
    # BOTH optimizers, same as BaCon's dead-gradient fix.)
    # ----------------------
    optimizer_part = optimizer_gate = scheduler_part = scheduler_gate = None
    part_bank = getattr(student, 'part_bank', None)
    if getattr(args, 'use_parts', False) and part_bank is not None:
        from backbone_adapter import is_late_block_param
        part_module = student.part_module
        optimizer_part = SGD(part_module.parameters(), lr=0.05, momentum=0.9,
                             weight_decay=1e-4)
        optimizer_gate = SGD([part_bank.gate_logits], lr=0.01, momentum=0.9)
        late_params = [p for n, p in student.backbone.named_parameters()
                       if is_late_block_param(n, 11) and p.requires_grad]
        if late_params:
            optimizer_part.add_param_group({'params': late_params, 'lr': 0.01})
        scheduler_part = lr_scheduler.CosineAnnealingLR(
            optimizer_part, T_max=args.epochs, eta_min=0.05 * 1e-3)
        scheduler_gate = lr_scheduler.CosineAnnealingLR(
            optimizer_gate, T_max=args.epochs, eta_min=0.01 * 1e-3)
        args.logger.info('[AdaPart] optimizers: part(lr=0.05) + gate(lr=0.01) ready.')


    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )

    # ----------------------
    # MOMENTUM TEACHER (ported from BaCon A1): slow EMA copy of the global
    # path (backbone+projector), stable distill target for cluster loss
    # instead of student.detach(). Teacher stays eval forever; student.train()
    # below only touches the student.
    # ----------------------
    teacher_m0 = float(getattr(args, 'teacher_m0', 0.996))
    teacher = teacher_source_params = None
    if getattr(args, 'use_momentum_teacher', False):
        teacher, teacher_source_params = _build_teacher(student, args)
        args.logger.info(f'[TEACHER] Momentum teacher ON (m0={teacher_m0} -> 1.0 cosine).')
    else:
        args.logger.info('[TEACHER] OFF - cluster loss targets = student.detach() (legacy).')

    # ----------------------
    # PSEUDO LABELING state (ported from BaCon; diagnostics use train GT only)
    # ----------------------
    if getattr(args, 'enable_pseudo_labeling', False):
        if getattr(args, 'use_parts', False):
            ce_view = CEView(student.backbone, student.projector)
        else:
            ce_view = CEView(student[0], student[1])
        pseudo_iteration = 0
        pseudo_samples_added = 0
        used_pseudo_uq_idxs = set()
        args.pseudo_events = []
        args.uq2true = build_uq_to_true_label(train_loader.dataset.unlabelled_dataset)
        orig_labeled_counts = {}
        count_labeled_per_class(train_loader.dataset.labelled_dataset, orig_labeled_counts)
        labeled_class_counts = dict(orig_labeled_counts)
        pseudo_added_per_class = {}
        args.logger.info("\n[PSEUDO LABELING] Enabled - Mode {mode}".format(mode=args.pseudo_mode))
        args._novel_iteration = 0
        args._novel_dim_members = {}
        args._novel_prev_dbi = None
        if getattr(args, 'enable_novel_pseudo', False):
            args.logger.info("[NOVEL-PSEUDO] Enabled.")
        else:
            args.logger.info("[NOVEL-PSEUDO] Disabled - known-only selection (D1).")

    # # inductive
    # best_test_acc_lab = 0
    # # transductive
    # best_train_acc_lab = 0
    # best_train_acc_ubl = 0 
    # best_train_acc_all = 0

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        # A1: momentum theo lich cosine (chi dung khi teacher bat)
        if teacher is not None:
            args._teacher_m = _teacher_momentum(epoch, args.epochs, teacher_m0)

        student.train()
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_proj, student_out = student(images)
                if teacher is not None:
                    # A1: dap an tu teacher cham (on dinh), khong phai student.detach()
                    with torch.no_grad():
                        _, teacher_out = teacher(images)
                else:
                    teacher_out = student_out.detach()

                # clustering, sup
                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                # clustering, unsup
                cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss

                # represent learning, unsup
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # representation learning, sup
                student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '

                loss = 0
                loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss

                # AdaPart gate regularization (BaCon weight 0.05; no
                # distribution-adaptive term: SimGCD has no dist_est).
                if part_bank is not None:
                    from part_modules import compute_gate_reg_loss
                    loss_gate, _, _ = compute_gate_reg_loss(part_bank.get_gates())
                    loss = loss + 0.05 * loss_gate
                    pstr += f'gate: {loss_gate.item():.4f} '

            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            if optimizer_part is not None:
                optimizer_part.zero_grad()
            if optimizer_gate is not None:
                optimizer_gate.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
                if optimizer_part is not None:
                    optimizer_part.step()
                if optimizer_gate is not None:
                    optimizer_gate.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                if optimizer_part is not None:
                    fp16_scaler.step(optimizer_part)
                if optimizer_gate is not None:
                    fp16_scaler.step(optimizer_gate)
                fp16_scaler.update()

            # A1: momentum teacher update moi step (sau optimizer, ke ca fp16)
            if teacher is not None:
                _ema_update_teacher(teacher.parameters(), teacher_source_params,
                                    args._teacher_m)

            # AdaPart prototype EMA (after warmup, per batch, no_grad; known from
            # GT labels, novel from fused pseudo + conf filter).
            # NOTE: view-0 = first B rows (images = cat(view0, view1), labels are
            # B-sized per item). BaCon takes [:B//n_views] here (quarter batch —
            # latent bug, kept frozen there to not disturb running numbers).
            if part_bank is not None and epoch >= min(30, args.epochs - 1):
                with torch.no_grad():
                    B_view = class_labels.size(0)
                    r_v0 = student.last_r_norm[:B_view].detach()
                    lab_v0 = class_labels
                    mask_v0 = mask_lab
                    if mask_v0.sum() > 0:
                        part_bank.update_ema(r_v0[mask_v0], lab_v0[mask_v0])
                    if (not mask_v0.all()) and not getattr(args, 'ablate_confidence', False):
                        f0 = student_out.chunk(2)[0].detach()
                        novel_mask = ~mask_v0
                        pseudo_pred = f0[novel_mask].argmax(dim=-1)
                        w = torch.softmax(f0[novel_mask] / args.tau_c, dim=-1).max(dim=-1)[0]
                        part_bank.update_ema_novel(r_v0[novel_mask], pseudo_pred, w)

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        args.logger.info('Testing on unlabelled examples in the training data...')
        all_acc, old_acc, new_acc, nmi, ari = test(
            student, unlabelled_train_loader, epoch=epoch,
            save_name='Train ACC Unlabelled', args=args)
        args.logger.info('Testing on disjoint test set...')
        t_acc, t_old, t_new, t_nmi, t_ari = test(
            student, test_loader, epoch=epoch,
            save_name='Test ACC', args=args)


        args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(t_acc, t_old, t_new))

        # Best tracking (transductive All; legacy code only kept last epoch).
        if all_acc > getattr(args, '_best_all', -1):
            args._best_all, args._best_old, args._best_new = all_acc, old_acc, new_acc
            args._best_nmi, args._best_ari, args._best_epoch = nmi, ari, epoch
            try:
                torch.save({'model': student.state_dict(), 'epoch': epoch + 1},
                           args.model_path + '.best.pt')
                args.logger.info(f'New best All {all_acc:.4f} (epoch {epoch}) -> model.pt.best.pt')
            except Exception as e:
                args.logger.warning(f'Best save failed: {e}')
        # Best tracking, disjoint test (the number comparable to BaCon track).
        if t_acc > getattr(args, '_best_t_all', -1):
            args._best_t_all, args._best_t_old, args._best_t_new = t_acc, t_old, t_new
            args._best_t_nmi, args._best_t_ari = t_nmi, t_ari
            args._best_t_epoch = epoch

        # AdaPart gate camera (cheap; answers "which slots does each class use?").
        if part_bank is not None and (epoch % 10 == 0 or epoch == args.epochs - 1):
            with torch.no_grad():
                _g = part_bank.get_gates()
                args.logger.info(f'[AdaPart] gates: mean={_g.mean().item():.3f} '
                                 f'max={_g.max().item():.3f} '
                                 f'frac_open(>0.5)={(_g > 0.5).float().mean().item():.3f} '
                                 f'active_slots_per_class~={_g.sum(dim=-1).mean().item():.2f}')

        # ----------------------
        # PSEUDO LABELING update (ported from BaCon; GT audit is diagnostics only)
        # ----------------------
        if getattr(args, 'enable_pseudo_labeling', False) and args.pseudo_update_freq > 0:
            args.current_epoch = epoch
            warmup = getattr(args, 'pseudo_warmup_epoch', 30)
            should_update = (epoch >= warmup) and ((epoch - warmup) % args.pseudo_update_freq == 0)
            if should_update and pseudo_iteration < args.max_pseudo_iterations:
                pseudo_iteration += 1
                args._novel_gate_summary = None
                args.logger.info("\n" + "=" * 60)
                args.logger.info(f"[PSEUDO LABELING] Iteration {pseudo_iteration}")
                args.logger.info("=" * 60)
                unlab_loader = DataLoader(
                    train_loader.dataset.unlabelled_dataset,
                    batch_size=256, shuffle=False, num_workers=0)
                if args.pseudo_mode == 1:
                    lab_loader = DataLoader(
                        train_loader.dataset.labelled_dataset,
                        batch_size=256, shuffle=False, num_workers=0)
                    train_stats = evaluate_train_labeled_per_class_accuracy(
                        ce_view, lab_loader, args.num_labeled_classes)
                    best_class = max(train_stats.keys(),
                                     key=lambda c: train_stats[c]['acc'])
                    args.logger.info(f"[MODE 1] Best train class: {best_class} "
                                     f"(train-acc: {train_stats[best_class]['acc']:.3f})")
                    target_class = best_class
                    args._mode3_skip = False
                elif args.pseudo_mode == 3:
                    from pseudo import _resolve_conf_bar
                    bar = _resolve_conf_bar(args)
                    best_class, scores, details, cstats = compute_unsupervised_class_scores(
                        ce_view, unlab_loader, args.num_labeled_classes,
                        conf_bar=bar, min_hi=getattr(args, 'pseudo_min_hi', 10))
                    args.logger.info(f"[MODE 3] Unlabeled conf: p50={cstats['p50']:.3f} "
                                     f"p90={cstats['p90']:.3f} max={cstats['max']:.3f} "
                                     f"(bar={bar:.4f})")
                    if best_class is None:
                        args.logger.info("[MODE 3] SKIP iteration: model not confident enough yet.")
                        target_class = None
                        args._mode3_skip = True
                    else:
                        d = details[best_class]
                        args.logger.info(f"[MODE 3] Best count class: {best_class} "
                                         f"(n_hi: {d['n_hi']}, n: {d['n']})")
                        target_class = best_class
                        args._mode3_skip = False
                elif args.pseudo_mode == 0:
                    target_class = None
                    args.logger.info("[MODE 0] Known door OFF — novel-only run.")
                    args._mode3_skip = False
                else:
                    target_class = None
                    args.logger.info(f"[MODE 2] Threshold: {args.confidence_threshold}")
                    args._mode3_skip = False

                if args.pseudo_mode == 0 or \
                        (args.pseudo_mode == 3 and getattr(args, '_mode3_skip', False)):
                    new_pseudo, newly_used = [], set()
                else:
                    new_pseudo, newly_used = collect_pseudo_labels_from_unlabeled(
                        ce_view, unlab_loader, mode=args.pseudo_mode,
                        target_class=target_class,
                        max_samples=args.max_samples_per_class,
                        threshold=getattr(args, 'confidence_threshold', 0.9),
                        used_uq_idxs=used_pseudo_uq_idxs,
                        top_ratio=getattr(args, 'pseudo_top_ratio', 0.8),
                        max_label=args.num_labeled_classes if args.pseudo_mode == 2 else None)
                used_pseudo_uq_idxs |= newly_used

                new_novel_pseudo = []
                if getattr(args, 'enable_novel_pseudo', False):
                    n_warm = getattr(args, 'novel_warmup_epoch', 50)
                    n_freq = getattr(args, 'novel_update_freq', 10)
                    n_max = getattr(args, 'max_novel_iterations', 2)
                    args._novel_iteration = getattr(args, '_novel_iteration', 0)
                    if epoch >= n_warm and ((epoch - n_warm) % n_freq == 0) \
                            and args._novel_iteration < n_max:
                        args._novel_iteration += 1
                        args.logger.info(f"[NOVEL-PSEUDO] Iteration {args._novel_iteration} "
                                         f"(epoch {epoch})")
                        new_novel_pseudo = collect_novel_pseudo_from_unlabeled(
                            ce_view, ce_view.backbone, unlab_loader,
                            train_loader, args, used_uq_idxs=used_pseudo_uq_idxs)
                        used_pseudo_uq_idxs |= {s['uq_idx'] for s in new_novel_pseudo}
                if new_novel_pseudo:
                    args.logger.info(f"Collected {len(new_novel_pseudo)} NOVEL pseudo samples "
                                     f"(+{len(new_pseudo)} known)")
                    new_pseudo = new_pseudo + new_novel_pseudo

                audit = audit_pseudo_samples(new_pseudo, args.uq2true,
                                             num_labeled=args.num_labeled_classes)
                log_pseudo_audit(audit, pseudo_iteration, target_class, args)
                if new_pseudo:
                    train_loader = update_train_loader(
                        train_loader, train_loader.dataset, new_pseudo)
                    pseudo_samples_added += len(new_pseudo)
                    for s in new_pseudo:
                        pseudo_added_per_class[s['label']] = pseudo_added_per_class.get(s['label'], 0) + 1
                    count_pseudo_into(labeled_class_counts, new_pseudo)
                    log_labeled_distribution(orig_labeled_counts, labeled_class_counts,
                                             pseudo_added_per_class, args)
                    args.pseudo_events.append({
                        'iteration': pseudo_iteration, 'epoch': epoch,
                        'n_selected': len(new_pseudo), **audit,
                        'novel_gate': getattr(args, '_novel_gate_summary', None),
                        'pseudo_samples_total': pseudo_samples_added,
                    })
                    args.logger.info(f"Total pseudo samples added: {pseudo_samples_added}")
                    args.logger.info("=" * 60 + "\n")

        # Step schedule
        exp_lr_scheduler.step()
        if scheduler_part is not None:
            scheduler_part.step()
        if scheduler_gate is not None:
            scheduler_gate.step()

        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        torch.save(save_dict, args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))

    if getattr(args, '_best_all', -1) >= 0:
        args.logger.info('Best (transductive) over {} epochs: All {:.4f} | Old {:.4f} | New {:.4f} '
                         '(epoch {}, NMI {} ARI {})'.format(
                             args.epochs, args._best_all, args._best_old, args._best_new,
                             args._best_epoch,
                             f"{args._best_nmi:.4f}" if args._best_nmi is not None else 'n/a',
                             f"{args._best_ari:.4f}" if args._best_ari is not None else 'n/a'))
    if getattr(args, '_best_t_all', -1) >= 0:
        args.logger.info('Best (DISJOINT test, comparable to BaCon track): All {:.4f} | Old {:.4f} | New {:.4f} '
                         '(epoch {}, NMI {} ARI {})'.format(
                             args._best_t_all, args._best_t_old, args._best_t_new,
                             args._best_t_epoch,
                             f"{args._best_t_nmi:.4f}" if args._best_t_nmi is not None else 'n/a',
                             f"{args._best_t_ari:.4f}" if args._best_t_ari is not None else 'n/a'))

        # if old_acc_test > best_test_acc_lab:
        #     
        #     args.logger.info(f'Best ACC on old Classes on disjoint test set: {old_acc_test:.4f}...')
        #     args.logger.info('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        #     
        #     torch.save(save_dict, args.model_path[:-3] + f'_best.pt')
        #     args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))
        #     
        #     # inductive
        #     best_test_acc_lab = old_acc_test
        #     # transductive            
        #     best_train_acc_lab = old_acc
        #     best_train_acc_ubl = new_acc
        #     best_train_acc_all = all_acc
        # 
        # args.logger.info(f'Exp Name: {args.exp_name}')
        # args.logger.info(f'Metrics with best model on test set: All: {best_train_acc_all:.4f} Old: {best_train_acc_lab:.4f} New: {best_train_acc_ubl:.4f}')


def test(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc, nmi, ari = log_accs_from_preds(
        y_true=targets, y_pred=preds, mask=mask,
        T=epoch, eval_funcs=args.eval_funcs, save_name=save_name, args=args)

    return all_acc, old_acc, new_acc, nmi, ari


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--seed', type=int, default=-1,
                        help='Train seed for paper repeats (e.g. 0/1). '
                             '-1 = legacy unseeded behavior (not reproducible). '
                             'Split files stay fixed; only init/sampler/aug vary.')
    parser.add_argument('--backbone', type=str, default='dinov2_vitb14',
                        choices=['dino_vitb16', 'dinov2_vitb14', 'dinov2_vitb14_reg'],
                        help='ViT backbone (default DINOv2-B/14 for the generality track; '
                             'dino_vitb16 reproduces the original paper numbers)')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    # ----------------------
    # MOMENTUM TEACHER (ported from BaCon A1; opt-in, default off to match old runs)
    # ----------------------
    parser.add_argument('--use_momentum_teacher', action='store_true', default=False,
                        help='EMA copy of the global path as stable distill target '
                             'instead of student.detach().')
    parser.add_argument('--teacher_m0', type=float, default=0.996,
                        help='Initial EMA momentum (cosine schedule -> 1.0).')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default=None, type=str)

    # ----------------------
    # ADAPART (ported from BaCon; underscore style like this repo)
    # ----------------------
    parser.add_argument('--use_parts', action='store_true', default=False,
                        help='Enable AdaPart fused-logits (global + part).')
    parser.add_argument('--num_slots', type=int, default=3,
                        help='Latent part slots per class (M).')
    parser.add_argument('--part_lambda', type=float, default=0.5,
                        help='Weight of part logits in fused score.')
    parser.add_argument('--tau_c', type=float, default=0.1,
                        help='Temperature for novel-EMA confidence.')
    parser.add_argument('--ablate_confidence', action='store_true', default=False,
                        help='Disable confidence filtering for novel prototype updates.')

    # ----------------------
    # PSEUDO LABELING (ported from BaCon; underscore style like this repo)
    # ----------------------
    parser.add_argument('--enable_pseudo_labeling', action='store_true', default=False)
    parser.add_argument('--pseudo_mode', type=int, default=1, choices=[0, 1, 2, 3],
                        help='0=known door OFF (novel-only), 1=best train-acc, '
                             '2=confidence threshold, 3=best high-conf count')
    parser.add_argument('--confidence_threshold', type=float, default=0.9)
    parser.add_argument('--pseudo_conf_bar', type=float, default=0.5)
    parser.add_argument('--pseudo_bar_k', type=float, default=2.0,
                        help='Mode 3 relative bar = k / num_classes (k<=0: absolute bar)')
    parser.add_argument('--pseudo_min_hi', type=int, default=10)
    parser.add_argument('--pseudo_top_ratio', type=float, default=0.8)
    parser.add_argument('--max_samples_per_class', type=int, default=500)
    parser.add_argument('--pseudo_update_freq', type=int, default=10)
    parser.add_argument('--max_pseudo_iterations', type=int, default=20)
    parser.add_argument('--pseudo_warmup_epoch', type=int, default=30)
    parser.add_argument('--enable_novel_pseudo', action='store_true', default=False,
                        help='Novel door D2 (cluster-anchored consensus). Single toggle.')
    parser.add_argument('--novel_warmup_epoch', type=int, default=50)
    parser.add_argument('--novel_update_freq', type=int, default=10)
    parser.add_argument('--max_novel_iterations', type=int, default=2)
    parser.add_argument('--novel_max_samples', type=int, default=100)
    parser.add_argument('--novel_jaccard_th', type=float, default=0.6)
    parser.add_argument('--novel_agree_th', type=float, default=0.7)
    parser.add_argument('--novel_min_size', type=int, default=10)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    init_experiment(args, runner_name=['simgcd'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    from backbone_adapter import load_backbone, backbone_spec, set_finetune_blocks
    backbone = load_backbone(args.backbone)

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))

    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    spec = backbone_spec(args.backbone)
    args.image_size = spec['image_size']
    args.feat_dim = spec['feat_dim']
    args.patch_size = spec['patch_size']
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    # Only finetune layers from block 'args.grad_from_block' onwards.
    # set_finetune_blocks handles DINOv1 (block.<i>) and DINOv2
    # (blocks.<i> / chunked) param names (see backbone_adapter.py).
    set_finetune_blocks(backbone, args.grad_from_block)


    args.logger.info(f"model build: backbone={args.backbone} "
                     f"(feat_dim={args.feat_dim}, patch={args.patch_size})")

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)

    # --------------------
    # TRAIN SEED (generality track; legacy code seeded nothing -> runs were
    # unreproducible. -1 = legacy unseeded; >=0 = explicit repeat seed.
    # Split files stay fixed; only init/sampler/aug vary.)
    # --------------------
    import random as _random
    import numpy as _np
    _seed = getattr(args, 'seed', -1)
    try:
        _seed = int(_seed)
    except Exception:
        _seed = -1
    args.train_seed = _seed
    if _seed is not None and _seed >= 0:
        torch.manual_seed(_seed)
        torch.cuda.manual_seed(_seed)
        torch.cuda.manual_seed_all(_seed)
        _np.random.seed(_seed)
        _random.seed(_seed)
        args.logger.info(f'[SEED] explicit train_seed={_seed}')
    else:
        args.logger.info('[SEED] legacy unseeded run (not reproducible; use --seed for paper repeats)')

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    # Disjoint test set (generality track): transductive numbers alone are NOT
    # comparable to the BaCon track (disjoint test). Evaluate both every epoch.
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD (+ optional AdaPart wrapper)
    # ----------------------
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    if getattr(args, 'use_parts', False):
        from part_modules import (LatentPartModule, PartPrototypeBank,
                                  PartFusedModel)
        from backbone_adapter import is_late_block_param
        part_module = LatentPartModule(dim=args.feat_dim,
                                       num_slots=args.num_slots).to(device)
        part_bank = PartPrototypeBank(num_classes=args.num_classes,
                                      num_slots=args.num_slots,
                                      dim=args.feat_dim).to(device)
        nn.init.constant_(part_bank.gate_logits, -1.0)  # a~=0.27, open gradually
        model = PartFusedModel(backbone, projector, part_module, part_bank,
                               part_lambda=args.part_lambda).to(device)
        args.logger.info(f'[AdaPart] Enabled: M={args.num_slots} slots, '
                         f'C={args.num_classes} classes, d={args.feat_dim}')
    else:
        part_module = part_bank = None
        model = nn.Sequential(backbone, projector).to(device)

    # ----------------------
    # TRAIN
    # ----------------------
    train(model, train_loader, test_loader_labelled, test_loader_unlabelled, args)
