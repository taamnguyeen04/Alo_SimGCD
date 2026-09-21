"""Dual pseudo-labeling ported from BaCon (generality track, plansimgcd.md).

Credit: selection/collection/audit logic mirrors BaCon/model/bacon.py
(known doors Mode 0/1/2/3 + novel door D2 cluster-anchored consensus +
2-door ground-truth audit). Adapted for SimGCD:
  - SimGCD student returns (proj, logits); access it through CEView below so
    all functions keep the model[0](images)->feats / model[1](feats)->logits
    convention.
  - Device-portable (cuda:0 when available, else cpu) so local CPU smokes work;
    on Modal GPU it behaves like BaCon's hardcoded cuda path.
  - No BaCon-only globals (no dist_est/part refs); labeled fallback loader only.

Nothing here trains by itself: train() in train.py calls
collect_pseudo_labels_from_unlabeled() and collect_novel_pseudo_from_unlabeled()
at epoch end, merges both doors, audits once, rebuilds the loader once.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Adapter: SimGCD student -> BaCon-style CE view
# ---------------------------------------------------------------------------

class _BackboneView:
    def __init__(self, backbone):
        self._backbone = backbone

    def __call__(self, images):
        return self._backbone(images)

    def parameters(self, *a, **k):
        return self._backbone.parameters(*a, **k)


class _HeadView:
    def __init__(self, head):
        self._head = head

    def __call__(self, feats):
        out = self._head(feats)
        # DINOHead-style heads return (proj, logits); plain heads return logits.
        if isinstance(out, (list, tuple)):
            _, logits = out
            return logits
        return out

    def parameters(self, *a, **k):
        return self._head.parameters(*a, **k)


class CEView:
    """Wrap a (backbone, head) pair so pseudo code written against BaCon's
    convention works unchanged:
      ce[0](images) -> raw feats; ce[1](feats) -> class logits; ce.eval().
    Raw backbone stays reachable as ce.backbone (for CL features in D2).

    Pass the GLOBAL branch (backbone + projector), never the fused wrapper:
    pseudo decisions must mirror BaCon's CE branch, not fused logits.
      plain student:    CEView(student[0], student[1])
      PartFusedModel:   CEView(student.backbone, student.projector)
    """

    def __init__(self, backbone, head):
        self.backbone = backbone
        self._head = head

    def __getitem__(self, i):
        if int(i) == 0:
            return _BackboneView(self.backbone)
        return _HeadView(self._head)

    def eval(self):
        self.backbone.eval()
        self._head.eval()

    def train(self, mode=True):
        self.backbone.train(mode)
        self._head.train(mode)


def _infer_device(module):
    try:
        return str(next(module.parameters()).device)
    except Exception:
        pass
    try:  # BaCon-style CE view: try the wrapped backbone first
        return str(next(module[0].parameters()).device)
    except Exception:
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def _first_view(images):
    if isinstance(images, (list, tuple)):
        return images[0]
    return images


def _pct(n, d):
    return 100.0 * n / d if d > 0 else 0.0


def _top(counter_dict, k=5):
    return dict(sorted(counter_dict.items(), key=lambda kv: kv[1], reverse=True)[:k])


# ---------------------------------------------------------------------------
# Known doors (Mode 0/1/2/3)
# ---------------------------------------------------------------------------

def _resolve_conf_bar(args, default=0.5):
    """Relative bar: bar = k / C (random-guess level is 1/C).

    --pseudo_bar_k > 0 (default 2.0): bar = k / num_classes.
    --pseudo_bar_k <= 0: absolute --pseudo_conf_bar (default 0.5).
    """
    try:
        k = float(getattr(args, 'pseudo_bar_k', 2.0))
    except Exception:
        k = 2.0
    if k > 0:
        try:
            nc = int(getattr(args, 'num_classes', 0) or 0)
        except Exception:
            nc = 0
        if nc > 0:
            return k / nc
    try:
        return float(getattr(args, 'pseudo_conf_bar', default))
    except Exception:
        return default


def evaluate_train_labeled_per_class_accuracy(model, labelled_loader, num_labeled,
                                              device=None):
    """Mode 1 (Cách A): per-class acc on TRAIN-LABELED (clean, no test leak)."""
    device = device or _infer_device(model[0])
    if hasattr(model, 'eval'):
        model.eval()
    stats = {c: {'correct': 0, 'total': 0} for c in range(num_labeled)}
    with torch.no_grad():
        for batch in labelled_loader:
            images = _first_view(batch[0]).to(device)
            labels = batch[1]
            if not torch.is_tensor(labels):
                labels = torch.tensor(labels)
            labels = labels.to(device)
            feats = F.normalize(model[0](images), dim=-1)
            preds = model[1](feats).argmax(dim=1)
            for p, t in zip(preds, labels):
                t = int(t.item())
                if 0 <= t < num_labeled:
                    stats[t]['total'] += 1
                    if int(p.item()) == t:
                        stats[t]['correct'] += 1
    for c in stats:
        tot = stats[c]['total']
        stats[c]['acc'] = stats[c]['correct'] / tot if tot > 0 else 0.0
    return stats


def compute_unsupervised_class_scores(model, unlab_loader, num_labeled, device=None,
                                      min_count=50, conf_bar=0.5, min_hi=10):
    """Mode 3 (Cách B): rank classes by COUNT of high-conf samples.

    best = argmax n_hi (tiebreak mean_conf) among classes with n_hi >= min_hi.
    No fallback: returns best_class=None when nothing passes -> caller SKIPs
    the iteration instead of injecting noise.
    Returns (best_class_or_None, scores, details, conf_stats).
    """
    device = device or _infer_device(model[0])
    if hasattr(model, 'eval'):
        model.eval()
    conf_lists = {c: [] for c in range(num_labeled)}
    all_confs = []
    with torch.no_grad():
        for batch in unlab_loader:
            images = _first_view(batch[0]).to(device)
            feats = F.normalize(model[0](images), dim=-1)
            probs = F.softmax(model[1](feats), dim=1)
            confs, preds = probs.max(dim=1)
            for p, c in zip(preds, confs):
                p = int(p.item())
                cf = float(c.item())
                all_confs.append(cf)
                if 0 <= p < num_labeled:
                    conf_lists[p].append(cf)
    import numpy as _np
    if all_confs:
        _a = _np.array(all_confs)
        conf_stats = {'p50': float(_np.percentile(_a, 50)),
                      'p90': float(_np.percentile(_a, 90)),
                      'max': float(_a.max())}
    else:
        conf_stats = {'p50': 0.0, 'p90': 0.0, 'max': 0.0}
    scores, details = {}, {}
    for c, lst in conf_lists.items():
        n = len(lst)
        mean_c = sum(lst) / n if n > 0 else 0.0
        hi = sum(1 for v in lst if v >= conf_bar)
        hi09 = sum(1 for v in lst if v >= 0.9)
        scores[c] = hi
        details[c] = {'n': n, 'mean_conf': mean_c, 'n_hi09': hi09, 'n_hi': hi}
    eligible = [c for c in scores if details[c]['n_hi'] >= min_hi]
    if not eligible:
        return None, scores, details, conf_stats
    best_class = max(eligible, key=lambda c: (scores[c], details[c]['mean_conf']))
    return best_class, scores, details, conf_stats


def collect_pseudo_labels_from_unlabeled(model, unlabeled_loader, mode, target_class,
                                         max_samples, threshold, used_uq_idxs=None,
                                         top_ratio=0.8, max_label=None, device=None):
    """Known-door collection. Mode 0 -> always ([], used).

    device=None -> inferred from the model (cuda:0 if available else cpu);
    pass device='cpu' explicitly for local CPU smokes / unit tests.
    """
    if mode == 0:
        return [], (used_uq_idxs if used_uq_idxs is not None else set())
    device = device or _infer_device(model)
    if hasattr(model, 'eval'):
        model.eval()
    if used_uq_idxs is None:
        used_uq_idxs = set()
    candidates = []
    with torch.no_grad():
        for batch in unlabeled_loader:
            images = _first_view(batch[0]).to(device)
            uq_idxs = batch[2]
            feats = F.normalize(model[0](images), dim=-1)
            probs = F.softmax(model[1](feats), dim=1)
            confs, preds = probs.max(dim=1)
            for img, pred, conf, uq in zip(images, preds, confs, uq_idxs):
                uq = int(uq)
                if uq in used_uq_idxs:
                    continue
                p, c = pred.item(), conf.item()
                if mode in (1, 3):
                    if p == target_class:
                        candidates.append({'image': img.cpu(), 'label': p,
                                           'confidence': c, 'uq_idx': uq})
                elif mode == 2:
                    if c >= threshold and (max_label is None or p < max_label):
                        candidates.append({'image': img.cpu(), 'label': p,
                                           'confidence': c, 'uq_idx': uq})
    if mode in (1, 3):
        candidates.sort(key=lambda s: s['confidence'], reverse=True)
        n_keep = min(int(len(candidates) * top_ratio), max_samples)
        selected = candidates[:n_keep]
    else:
        per_class = {}
        for s in candidates:
            per_class.setdefault(s['label'], []).append(s)
        selected = []
        for lbl, items in per_class.items():
            items.sort(key=lambda s: s['confidence'], reverse=True)
            selected.extend(items[:max_samples])
        selected.sort(key=lambda s: s['confidence'], reverse=True)
    return selected, {s['uq_idx'] for s in selected}


# ---------------------------------------------------------------------------
# Novel door (D2): cluster-anchored consensus. Never raises.
# ---------------------------------------------------------------------------

def collect_novel_pseudo_from_unlabeled(student_ce, cl_backbone, unlab_loader,
                                       train_loader, args, used_uq_idxs=None):
    """Steps: fused single pass (CL feats + CE logits) -> KMeans x2 (fit
    subsample<=15000, assign full) -> align via labeled nearest-centroid +
    Hungarian (leftover clusters = novel candidates) -> 3 gates (stability
    Jaccard on uq sets / silhouette mean >= median on <=2000 subsample /
    agreement: majority CE-pred novel dim) + size guard -> DBI gate (skip if
    >5% worse) -> no-remap guard -> per-dim cap by CE confidence.

    Returns same sample schema as the known door. Any failure -> warning + [].
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import davies_bouldin_score, silhouette_samples
        from scipy.optimize import linear_sum_assignment as linear_assignment
    except Exception as e:
        args.logger.warning(f"[NOVEL-PSEUDO] sklearn/scipy missing ({e}) — skipping.")
        try:
            args._novel_gate_summary = {'status': 'skipped-no-sklearn'}
        except Exception:
            pass
        return []

    try:
        import numpy as np
        device = _infer_device(cl_backbone)
        NL = int(args.num_labeled_classes)
        K = int(args.num_classes)
        j_th = float(getattr(args, 'novel_jaccard_th', 0.6))
        a_th = float(getattr(args, 'novel_agree_th', 0.7))
        min_size = int(getattr(args, 'novel_min_size', 10))
        cap = int(getattr(args, 'novel_max_samples', 100))
        used_uq = used_uq_idxs if used_uq_idxs is not None else set()

        # NOTE: eval() without restore mirrors the known door on purpose.
        student_ce.eval()
        cl_backbone.eval()

        U_feats, U_pred, U_conf, U_uq, U_imgs = [], [], [], [], []
        with torch.no_grad():
            for batch in unlab_loader:
                raw = _first_view(batch[0])
                images = raw.to(device)
                uq = batch[2]
                cf = F.normalize(cl_backbone(images), dim=-1)
                ce_f = F.normalize(student_ce[0](images), dim=-1)
                probs = F.softmax(student_ce[1](ce_f), dim=1)
                conf, pred = probs.max(dim=1)
                U_feats.append(cf.cpu())
                U_pred.append(pred.cpu())
                U_conf.append(conf.cpu())
                U_uq.extend([int(x) for x in uq])
                U_imgs.extend([im.cpu() for im in raw])
        U_feats = torch.cat(U_feats).numpy()
        U_pred = torch.cat(U_pred).numpy().astype(int)
        U_conf = torch.cat(U_conf).numpy()
        U_uq = np.array(U_uq)
        N = len(U_feats)

        rng = np.random.RandomState(int(getattr(args, '_novel_iteration', 0)) + 12345)
        fit_idx = rng.choice(N, size=min(N, 15000), replace=False)
        km0 = KMeans(n_clusters=K, random_state=0, n_init=10).fit(U_feats[fit_idx])
        km1 = KMeans(n_clusters=K, random_state=1, n_init=10).fit(U_feats[fit_idx])
        cent0 = km0.cluster_centers_
        lab0 = ((U_feats[:, None, :] - cent0[None, :, :]) ** 2).sum(-1).argmin(axis=1)
        lab1 = ((U_feats[:, None, :] - km1.cluster_centers_[None, :, :]) ** 2).sum(-1).argmin(axis=1)

        lab_loader = DataLoader(train_loader.dataset.labelled_dataset,
                                batch_size=256, shuffle=False, num_workers=0)
        L_feats, L_true = [], []
        with torch.no_grad():
            for batch in lab_loader:
                images = _first_view(batch[0]).to(device)
                labels = batch[1]
                if not torch.is_tensor(labels):
                    labels = torch.tensor(labels)
                f = F.normalize(cl_backbone(images), dim=-1)
                L_feats.append(f.cpu())
                L_true.extend([int(x) for x in labels])
        L_feats = torch.cat(L_feats).numpy()
        L_true = np.array(L_true)
        L_assign = ((L_feats[:, None, :] - cent0[None, :, :]) ** 2).sum(-1).argmin(axis=1)
        w = np.zeros((K, NL), dtype=int)
        for c, t in zip(L_assign, L_true):
            if 0 <= t < NL:
                w[c, t] += 1
        rows, _ = linear_assignment(w.max() - w)
        known_clusters = set(int(r) for r in rows)
        leftover = [c for c in range(K) if c not in known_clusters]

        mem0 = {c: set(U_uq[lab0 == c].tolist()) for c in range(K)}
        mem1 = {c: set(U_uq[lab1 == c].tolist()) for c in range(K)}

        def _jacc(a, b):
            u = len(a | b)
            return len(a & b) / u if u > 0 else 0.0

        sub_idx = rng.choice(N, size=min(N, 2000), replace=False)
        sil = silhouette_samples(U_feats[sub_idx], lab0[sub_idx])
        sil_mean = {}
        for c in range(K):
            m = lab0[sub_idx] == c
            sil_mean[c] = float(sil[m].mean()) if m.any() else -1.0
        sil_med = float(np.median(list(sil_mean.values())))
        dbi = float(davies_bouldin_score(U_feats[sub_idx], lab0[sub_idx]))
        prev_dbi = getattr(args, '_novel_prev_dbi', None)
        if prev_dbi is not None and dbi > prev_dbi * 1.05:
            args.logger.info(f"[NOVEL-PSEUDO] DBI {dbi:.3f} worse than prev {prev_dbi:.3f} "
                             "(>5%) — skipping novel inject this iter.")
            try:
                args._novel_gate_summary = {'status': 'skipped-dbi',
                                            'dbi': dbi, 'prev_dbi': prev_dbi}
            except Exception:
                pass
            return []
        args._novel_prev_dbi = dbi

        dim_members = getattr(args, '_novel_dim_members', None)
        if dim_members is None:
            dim_members = {}
            args._novel_dim_members = dim_members
        selected, gate_rows = [], []
        for c in leftover:
            members = np.where(lab0 == c)[0]
            size = len(members)
            stab = max((_jacc(mem0[c], mem1[b]) for b in range(K)), default=0.0)
            s_mean = sil_mean.get(c, -1.0)
            mpred = U_pred[members] if size else np.array([], dtype=int)
            novel_mask = mpred >= NL
            if novel_mask.sum() == 0:
                gate_rows.append((c, size, stab, s_mean, '-', 0.0, 'drop:no-novel-pred'))
                continue
            vals, counts = np.unique(mpred[novel_mask], return_counts=True)
            top_dim = int(vals[counts.argmax()])
            agree = float(counts.max() / size)
            reason = 'keep'
            if size < min_size:
                reason = f'drop:size<{min_size}'
            elif stab < j_th:
                reason = f'drop:stab<{j_th}'
            elif s_mean < sil_med:
                reason = 'drop:sil<median'
            elif agree < a_th:
                reason = f'drop:agree<{a_th}'
            else:
                prev_set = dim_members.get(top_dim, set())
                if prev_set and _jacc(set(U_uq[members].tolist()), prev_set) < 0.3:
                    reason = 'drop:remap-flip'
            gate_rows.append((c, size, stab, s_mean, top_dim, agree, reason))
            if reason != 'keep':
                continue
            order = members[np.argsort(-U_conf[members])]
            kept, new_member_uqs = 0, set()
            for idx in order:
                if kept >= cap:
                    break
                uq = int(U_uq[idx])
                if uq in used_uq:
                    continue
                used_uq.add(uq)
                new_member_uqs.add(uq)
                selected.append({'image': U_imgs[idx], 'label': top_dim,
                                 'confidence': float(U_conf[idx]), 'uq_idx': uq})
                kept += 1
            dim_members[top_dim] = new_member_uqs

        args.logger.info(f"[NOVEL-PSEUDO] K={K} leftover={len(leftover)} "
                         f"sil_med={sil_med:.3f} dbi={dbi:.3f} -> selected {len(selected)}")
        for (c, size, stab, s_mean, td, agree, reason) in gate_rows:
            args.logger.info(f"  cluster {c:>3}: n={size:5d} stab={stab:.2f} "
                             f"sil={s_mean:+.2f} dim={td} agree={agree:.2f} {reason}")
        try:
            from collections import Counter as _Counter
            args._novel_gate_summary = {
                'status': 'kept', 'n_leftover': int(len(leftover)),
                'n_selected': int(len(selected)),
                'sil_med': float(sil_med), 'dbi': float(dbi),
                'reason_hist': dict(_Counter(r for (_, _, _, _, _, _, r) in gate_rows)),
                'per_dim': dict(_Counter(int(s['label']) for s in selected))}
        except Exception:
            pass
        return selected
    except Exception as e:
        import traceback
        args.logger.warning(f"[NOVEL-PSEUDO] Failed (non-fatal, known-door unaffected): {e}")
        args.logger.warning(traceback.format_exc(limit=5))
        try:
            args._novel_gate_summary = {'status': 'failed', 'error': str(e)[:200]}
        except Exception:
            pass
        return []


# ---------------------------------------------------------------------------
# Pseudo dataset / loader rebuild / audit (2-door) / evidence table
# ---------------------------------------------------------------------------

def create_pseudo_dataset(pseudo_samples):
    class PseudoDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.images = [s['image'] for s in samples]
            self.labels = [s['label'] for s in samples]
            self.uq_idxs = [s['uq_idx'] for s in samples]

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img = self.images[idx]
            return [img, img.clone()], self.labels[idx], self.uq_idxs[idx]

    return PseudoDataset(pseudo_samples)


def update_train_loader(train_loader, train_dataset, new_pseudo_samples):
    """Rebuild loader with pseudo samples appended to the labeled side."""
    if not new_pseudo_samples:
        return train_loader
    pseudo_dataset = create_pseudo_dataset(new_pseudo_samples)
    original_labeled = train_dataset.labelled_dataset
    original_unlabeled = train_dataset.unlabelled_dataset

    class CombinedLabeledDataset(torch.utils.data.Dataset):
        def __init__(self, ds1, ds2):
            self.ds1 = ds1
            self.ds2 = ds2

        def __len__(self):
            return len(self.ds1) + len(self.ds2)

        def __getitem__(self, idx):
            if idx < len(self.ds1):
                return self.ds1[idx]
            return self.ds2[idx - len(self.ds1)]

    combined_labeled = CombinedLabeledDataset(original_labeled, pseudo_dataset)
    from data.data_utils import MergedDataset
    new_train_dataset = MergedDataset(labelled_dataset=combined_labeled,
                                      unlabelled_dataset=original_unlabeled)
    label_len = len(combined_labeled)
    unlab_len = len(original_unlabeled)
    sample_weights = [1 if i < label_len else label_len / unlab_len
                      for i in range(len(new_train_dataset))]
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.DoubleTensor(sample_weights), num_samples=len(new_train_dataset))
    return DataLoader(new_train_dataset, batch_size=train_loader.batch_size,
                      shuffle=False, sampler=sampler, drop_last=True,
                      pin_memory=True, num_workers=train_loader.num_workers)


def build_uq_to_true_label(unlabelled_dataset):
    """Map uq_idx -> true label over the unlabeled pool (audit ONLY)."""
    mapping = {}

    def _walk(ds):
        inner = getattr(ds, 'datasets', None)
        if inner is not None:
            for d in inner:
                _walk(d)
            return
        targets = getattr(ds, 'targets', None)
        if targets is None:
            data = getattr(ds, 'data', None)
            if data is not None and hasattr(data, 'target'):
                import pandas as _pd  # SimGCD CUB-style metadata
                targets = (data['target'].to_numpy() - 1).tolist()
        if targets is None and hasattr(ds, 'target'):
            # CarsDataset: .target is a 1-based list (see stanford_cars.py
            # __getitem__: target = self.target[idx] - 1, then
            # target_transform). Mirror that exactly for the audit map.
            try:
                import numpy as _np
                _raw = _np.asarray(ds.target).reshape(-1)
                if len(_raw) == len(ds):
                    _lbl = [_int0 - 1 for _int0 in _raw.astype(int).tolist()]
                    _tform = getattr(ds, 'target_transform', None)
                    if _tform is not None:
                        _lbl = [_tform(_l) for _l in _lbl]
                    targets = _lbl
            except Exception:
                targets = None
        uq_idxs = getattr(ds, 'uq_idxs', None)
        if targets is not None and uq_idxs is not None:
            for t, u in zip(targets, uq_idxs):
                mapping[int(u)] = int(t)

    _walk(unlabelled_dataset)
    return mapping


def count_labeled_per_class(labeled_dataset, out):
    targets = getattr(labeled_dataset, 'targets', None)
    if targets is not None:
        for t in targets:
            out[int(t)] = out.get(int(t), 0) + 1
    elif hasattr(labeled_dataset, 'target'):
        # CarsDataset sibling of the branch above (1-based -> -1, then
        # target_transform), so the distribution log isn't empty on Cars.
        try:
            import numpy as _np
            _raw = _np.asarray(labeled_dataset.target).reshape(-1)
            if len(_raw) == len(labeled_dataset):
                _tform = getattr(labeled_dataset, 'target_transform', None)
                for _t in _raw.astype(int).tolist():
                    _l = int(_t) - 1
                    if _tform is not None:
                        _l = int(_tform(_l))
                    out[_l] = out.get(_l, 0) + 1
                return
        except Exception:
            pass
        inner = getattr(labeled_dataset, 'datasets', None)
        if inner is not None:
            for d in inner:
                count_labeled_per_class(d, out)
    else:
        inner = getattr(labeled_dataset, 'datasets', None)
        if inner is not None:
            for d in inner:
                count_labeled_per_class(d, out)


def count_pseudo_into(class_counts, pseudo_samples):
    for s in pseudo_samples:
        c = int(s['label'])
        class_counts[c] = class_counts.get(c, 0) + 1


def audit_pseudo_samples(pseudo_samples, uq2true, num_labeled):
    """2-door audit (diagnostics only). Pseudo-labels are class IDs already.

    D1 (label < num_labeled): n_true_correct / n_wrong_old /
      n_novel_contamination + novel_true_labels (+ conf lists per outcome).
    D2 (label >= num_labeled): n_novel_correct / n_known_leakage /
      n_novel_confused + known_leak_sources / novel_confused_true (+ confs).
    """
    from collections import Counter
    import math as _math
    n_true_correct = n_wrong_old = n_novel = 0
    novel_true_labels = Counter()
    n_novel_correct = n_known_leak = n_novel_conf = 0
    known_leak_sources = Counter()
    novel_confused_true = Counter()
    n_d1 = n_d2 = 0
    n_miss = 0  # Fix B: uq_idx absent from uq2true (empty audit map) —
    # never let that masquerade as "nothing to audit" again (pre-92dec06 Cars
    # bug printed n=0 next to "Collected 633 NOVEL pseudo samples").
    c_ok, c_wo, c_sw, c_nok, c_lk, c_cf = [], [], [], [], [], []
    for s in pseudo_samples:
        true_lbl = uq2true.get(int(s['uq_idx']))
        if true_lbl is None:
            n_miss += 1
            continue
        pseudo_lbl = int(s['label'])
        try:
            conf = float(s.get('confidence', float('nan')))
        except Exception:
            conf = float('nan')
        has_conf = not _math.isnan(conf)
        if pseudo_lbl < num_labeled:
            n_d1 += 1
            if true_lbl < num_labeled:
                if true_lbl == pseudo_lbl:
                    n_true_correct += 1
                    if has_conf:
                        c_ok.append(conf)
                else:
                    n_wrong_old += 1
                    if has_conf:
                        c_wo.append(conf)
            else:
                n_novel += 1
                novel_true_labels[true_lbl] += 1
                if has_conf:
                    c_sw.append(conf)
        else:
            n_d2 += 1
            if true_lbl >= num_labeled:
                if true_lbl == pseudo_lbl:
                    n_novel_correct += 1
                    if has_conf:
                        c_nok.append(conf)
                else:
                    n_novel_conf += 1
                    novel_confused_true[true_lbl] += 1
                    if has_conf:
                        c_cf.append(conf)
            else:
                n_known_leak += 1
                known_leak_sources[true_lbl] += 1
                if has_conf:
                    c_lk.append(conf)
    return {
        'n_selected_gt': len(pseudo_samples),
        'n_unmapped': n_miss,  # Fix B: audit coverage, not selection quality
        'n_selected_d1': n_d1, 'n_selected_d2': n_d2,
        'n_true_correct': n_true_correct, 'n_wrong_old': n_wrong_old,
        'n_novel_contamination': n_novel,
        'novel_true_labels': dict(novel_true_labels),
        'n_novel_correct': n_novel_correct, 'n_known_leakage': n_known_leak,
        'n_novel_confused': n_novel_conf,
        'known_leak_sources': dict(known_leak_sources),
        'novel_confused_true': dict(novel_confused_true),
        'conf_correct': c_ok, 'conf_wrong_old': c_wo, 'conf_swallowed': c_sw,
        'conf_novel_correct': c_nok, 'conf_leak': c_lk, 'conf_confused': c_cf,
    }


def log_pseudo_audit(audit, pseudo_iteration, target_class, args):
    """D1 legacy format + D2 block + cumulative evidence table."""
    sel = audit['n_selected_gt']
    if sel == 0:
        args.logger.info(f"[PSEUDO AUDIT iter {pseudo_iteration}] No samples selected — nothing to audit.")
        return
    d1 = audit.get('n_selected_d1', sel)
    d2 = audit.get('n_selected_d2', 0)
    # Fix B: loud failure when the audit map is blind. Pre-92dec06 Cars runs
    # printed "n=0 — nothing to audit" next to thousands of injected samples
    # because uq2true was empty; surfaces that state instead of hiding it.
    n_unmapped = int(audit.get('n_unmapped', 0) or 0)
    if sel > 0 and n_unmapped > 0.5 * sel:
        args.logger.warning(f"[PSEUDO AUDIT iter {pseudo_iteration}] audit map covers only "
                            f"{sel - n_unmapped}/{sel} samples ({n_unmapped} unmapped) — "
                            "uq2true is empty/partial; check build_uq_to_true_label "
                            "for this dataset. D1/D2 tables below are NOT ground truth.")
    args.logger.info("\n" + "-" * 60)
    args.logger.info(f"[PSEUDO AUDIT iter {pseudo_iteration}] Ground-truth check of selected pseudo labels"
                     + (f" (target class {target_class})" if target_class is not None else ""))
    pct_ok = _pct(audit['n_true_correct'], d1)
    args.logger.info(f"  [D1 known door] n={d1}")
    args.logger.info(f"    Correct (old==old):        {audit['n_true_correct']:4d}/{d1} = {pct_ok:5.1f}%")
    args.logger.info(f"    WRONG OLD class:           {audit['n_wrong_old']:4d}/{d1} = {_pct(audit['n_wrong_old'], d1):5.1f}%")
    args.logger.info(f"    NOVEL class swallowed:     {audit['n_novel_contamination']:4d}/{d1} = {_pct(audit['n_novel_contamination'], d1):5.1f}%")
    if audit['novel_true_labels']:
        args.logger.info(f"    True identities of swallowed novel samples: "
                         f"{dict(sorted(audit['novel_true_labels'].items()))}")
    verdict = ("OK (<10% wrong)" if pct_ok >= 90 else
               "BIASED (10-30% wrong)" if pct_ok >= 70 else
               "HEAVILY BIASED (>30% wrong)")
    args.logger.info(f"    VERDICT D1: {verdict}")
    if d2 == 0:
        args.logger.info("  [D2 novel door] n=0 — known-only selection, nothing to audit.")
    else:
        pct_nok = _pct(audit.get('n_novel_correct', 0), d2)
        pct_leak = _pct(audit.get('n_known_leakage', 0), d2)
        pct_conf = _pct(audit.get('n_novel_confused', 0), d2)
        args.logger.info(f"  [D2 novel door] n={d2}")
        args.logger.info(f"    Correct (novel==novel):    {audit.get('n_novel_correct', 0):4d}/{d2} = {pct_nok:5.1f}%")
        args.logger.info(f"    KNOWN leaked as novel:     {audit.get('n_known_leakage', 0):4d}/{d2} = {pct_leak:5.1f}%")
        args.logger.info(f"    NOVEL-vs-NOVEL confused:   {audit.get('n_novel_confused', 0):4d}/{d2} = {pct_conf:5.1f}%")
        if audit.get('known_leak_sources'):
            args.logger.info(f"    Known classes leaking out: {_top(audit['known_leak_sources'])}")
        if audit.get('novel_confused_true'):
            args.logger.info(f"    Novel classes getting mixed: {_top(audit['novel_confused_true'])}")
        verdict2 = ("OK (<15% bad)" if (pct_leak + pct_conf) < 15 else
                    "RISKY (15-40% bad)" if (pct_leak + pct_conf) < 40 else
                    "HEAVILY BIASED (>40% bad)")
        args.logger.info(f"    VERDICT D2: {verdict2}")
    _log_audit_evidence_table(audit, pseudo_iteration, args)
    args.logger.info("-" * 60 + "\n")


def _log_audit_evidence_table(audit, pseudo_iteration, args):
    history = list(getattr(args, 'pseudo_events', []) or [])
    rows = []
    cum_d1_eaten = cum_d2_leak = cum_d2_conf = 0
    for ev in history:
        cum_d1_eaten += ev.get('n_novel_contamination', 0)
        cum_d2_leak += ev.get('n_known_leakage', 0)
        cum_d2_conf += ev.get('n_novel_confused', 0)
        rows.append((ev.get('iteration', '?'), ev.get('epoch', '?'),
                     ev.get('n_selected_d1', ev.get('n_selected_gt', 0)),
                     ev.get('n_novel_contamination', 0), cum_d1_eaten,
                     ev.get('n_selected_d2', 0),
                     ev.get('n_known_leakage', 0), cum_d2_leak))
    cum_d1_eaten += audit.get('n_novel_contamination', 0)
    cum_d2_leak += audit.get('n_known_leakage', 0)
    cum_d2_conf += audit.get('n_novel_confused', 0)
    cur_epoch = getattr(args, 'current_epoch', '?')
    rows.append((pseudo_iteration, cur_epoch,
                 audit.get('n_selected_d1', audit.get('n_selected_gt', 0)),
                 audit.get('n_novel_contamination', 0), cum_d1_eaten,
                 audit.get('n_selected_d2', 0),
                 audit.get('n_known_leakage', 0), cum_d2_leak))
    args.logger.info("  [EVIDENCE] Contamination across iterations (per-iter | cumulative):")
    args.logger.info("    iter | epoch | d1_sel | d1_novel_eaten(iter|cum) | d2_sel | d2_known_leak(iter|cum)")
    for r in rows:
        args.logger.info(f"    {r[0]:>4} | {str(r[1]):>5} | {r[2]:6d} | "
                         f"{r[3]:6d} | {r[4]:6d}          | {r[5]:6d} | "
                         f"{r[6]:6d} | {r[7]:6d}")
    from collections import Counter
    eaten, leaked = Counter(), Counter()
    for ev in history:
        eaten.update(ev.get('novel_true_labels', {}) or {})
        leaked.update(ev.get('known_leak_sources', {}) or {})
    eaten.update(audit.get('novel_true_labels', {}) or {})
    leaked.update(audit.get('known_leak_sources', {}) or {})
    if eaten:
        args.logger.info(f"    Cumulative novel classes eaten (top5): {_top(dict(eaten))}")
    if leaked:
        args.logger.info(f"    Cumulative known classes leaked (top5): {_top(dict(leaked))}")
    if cum_d2_conf:
        args.logger.info(f"    Cumulative novel-vs-novel confused: {cum_d2_conf}")


def log_labeled_distribution(orig_counts, cur_counts, pseudo_added_per_class, args):
    """Log how the labeled-set distribution drifts as pseudo samples pile up."""
    import numpy as np
    all_classes = sorted(set(orig_counts) | set(cur_counts))
    orig_vals = np.array([orig_counts.get(c, 0) for c in all_classes], dtype=float)
    cur_vals = np.array([cur_counts.get(c, 0) for c in all_classes], dtype=float)
    orig_max = orig_vals.max() if orig_vals.max() > 0 else 1.0
    cur_max = cur_vals.max() if cur_vals.max() > 0 else 1.0
    imb_orig = orig_max / max(orig_vals.min(), 1)
    imb_cur = cur_max / max(cur_vals.min(), 1)
    args.logger.info("[PSEUDO DISTRIBUTION] Labeled-set composition after injection:")
    for c, o, v in zip(all_classes, orig_vals, cur_vals):
        if int(v - o) != 0 or o > 0:
            args.logger.info(f"  Class {c}: {int(o):5d} -> {int(v):5d}  (+{int(v - o)})")
    args.logger.info(f"  Imbalance ratio (max/min): {imb_orig:.2f} -> {imb_cur:.2f}"
                     f"   {'(WORSE)' if imb_cur > imb_orig * 1.05 else '(~same/better)'}")
