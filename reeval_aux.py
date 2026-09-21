"""Re-eval an AuxTest50 best_test checkpoint with 3 eval heads (no retrain).

1. fused-argmax   (current SimGCD eval; expect ~64.66/89.41/40.78 @ep28)
2. global-argmax   (native branch only)
3. KMeans(196) on norm([z_cls || r_pool]) (BaCon-style contract)

Usage (Windows, venv-hosts):
  $env:SIMGCD_SCARS_ROOT="D:/data/car"
  venv-hosts\\Scripts\\python.exe SimGCD\\reeval_aux.py --ckpt C:\\path\\model.pt.best_test.pt
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# SSB split / imb files are referenced via relative paths (like train.py),
# so always run from this folder regardless of caller cwd.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import exp_root  # noqa: F401 (keeps import parity with train.py)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--kmeans_seed', type=int, default=0)
    return p.parse_args()


def hungarian_acc(y_true, y_pred, mask):
    from scipy.optimize import linear_sum_assignment as linear_assignment
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    mask = np.asarray(mask, dtype=bool)
    D = int(max(y_pred.max(), y_true.max())) + 1
    w = np.zeros((D, D), dtype=int)
    for a, b in zip(y_pred, y_true):
        w[a, b] += 1
    ind = np.vstack(linear_assignment(w.max() - w)).T
    ind_map = {j: i for i, j in ind}
    total = w[ind[:, 0], ind[:, 1]].sum() / len(y_true)
    old_gt = set(y_true[mask].tolist())
    new_gt = set(y_true[~mask].tolist())
    old = sum(w[ind_map[i], i] for i in old_gt) / sum(w[:, i].sum() for i in old_gt)
    new = sum(w[ind_map[i], i] for i in new_gt) / sum(w[:, i].sum() for i in new_gt)
    return total, old, new


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device, flush=True)

    from data.get_datasets import get_datasets, get_class_splits
    from data.augmentations import get_transform
    from model import DINOHead
    from backbone_adapter import load_backbone, backbone_spec
    from part_modules import PartFusedModel, LatentPartModule, PartPrototypeBank
    import torch.nn as nn

    # ---- replicate train.py dataset setup (SSB scars, test transforms) ----
    targs = SimpleNamespace(
        dataset_name='scars', use_ssb_splits=True, imb_ratio=None,
        prop_train_labels=0.5,
        interpolation=3, crop_pct=0.875, transform='imagenet',
    )
    targs = get_class_splits(targs)
    num_labeled = len(targs.train_classes)
    num_classes = num_labeled + len(targs.unlabeled_classes)
    print(f'labeled={num_labeled} total={num_classes}', flush=True)

    train_transform, test_transform = get_transform('imagenet', image_size=224, args=targs)
    train_ds, test_ds, unlab_test_ds, _ = get_datasets(
        'scars', train_transform, test_transform, targs)
    # NOTE: get_datasets sets target_transform to known+novel -> 0..195,
    # so Old == labels < num_labeled, exactly like train.py test().
    unlab_loader = DataLoader(unlab_test_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=False)

    # ---- rebuild PartFusedModel exactly as train.py ----
    backbone = load_backbone('dinov2_vitb14')
    spec = backbone_spec('dinov2_vitb14')
    projector = DINOHead(in_dim=spec['feat_dim'], out_dim=num_classes, nlayers=3)
    part_module = LatentPartModule(dim=spec['feat_dim'], num_slots=3)
    part_bank = PartPrototypeBank(num_classes=num_classes, num_slots=3,
                                  dim=spec['feat_dim'])
    model = PartFusedModel(backbone, projector, part_module, part_bank,
                           part_lambda=0.5).to(device)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'loaded {args.ckpt} (epoch {ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"}), '
          f'missing={len(missing)}, unexpected={len(unexpected)}', flush=True)
    if missing or unexpected:
        print('missing:', missing, '\nunexpected:', unexpected, flush=True)
    model.eval()

    from backbone_adapter import forward_backbone_tokens

    for name, loader in [('train-unlabelled', unlab_loader), ('disjoint-test', test_loader)]:
        fused_pred, glob_pred, feats, targets, masks = [], [], [], [], []
        with torch.no_grad():
            for images, label, _ in tqdm(loader, desc=name):
                images = images.to(device)
                proj, fused = model(images)
                glob = model.last_global_logits
                # BaCon-style feature: norm([z_cls || gate-weighted r_pool])
                cls, patches, _ = forward_backbone_tokens(model.backbone, images)
                r_norm, _ = model.part_module(patches)
                with torch.no_grad():
                    gates = torch.sigmoid(model.part_bank.gate_logits)  # (C, M)
                # r_pool per predicted-fused class is circular; use mean-slot
                # pool weighted by mean gate (class-agnostic, eval-only)
                a = gates.mean(dim=0)  # (M,)
                r_pool = (r_norm * a.view(1, -1, 1)).sum(dim=1)
                feat = torch.cat([F.normalize(cls, dim=-1),
                                  F.normalize(r_pool, dim=-1)], dim=-1)
                feats.append(F.normalize(feat, dim=-1).cpu())
                fused_pred.append(fused.argmax(1).cpu())
                glob_pred.append(glob.argmax(1).cpu())
                label = np.asarray(label)
                targets.append(label)
                masks.append(np.array([int(x) < num_labeled for x in label]))
        fused_pred = np.concatenate([p.numpy() for p in fused_pred])
        glob_pred = np.concatenate([p.numpy() for p in glob_pred])
        feats = torch.cat(feats).numpy()
        targets = np.concatenate(targets)
        masks = np.concatenate(masks)

        a, o, n = hungarian_acc(targets, fused_pred, masks)
        print(f'[{name}] fused-argmax : All {a:.4f} | Old {o:.4f} | New {n:.4f}', flush=True)
        a, o, n = hungarian_acc(targets, glob_pred, masks)
        print(f'[{name}] global-argmax: All {a:.4f} | Old {o:.4f} | New {n:.4f}', flush=True)

        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=num_classes, random_state=args.kmeans_seed,
                    n_init=10).fit(feats)
        a, o, n = hungarian_acc(targets, km.labels_, masks)
        print(f'[{name}] kmeans-feat  : All {a:.4f} | Old {o:.4f} | New {n:.4f}', flush=True)


if __name__ == '__main__':
    main()
