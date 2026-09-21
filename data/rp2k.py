"""
RP2k dataloader cho SimGCD — mất cân bằng TỰ NHIÊN, KHÔNG cần file .pt.

Bối cảnh (đọc kĩ trước khi dùng):
---------------------------------
1. RP2k (Retail Product 2000) vốn đã long-tail rất nặng:
   - ~381k ảnh / 2233 lớp (theo meta.csv), tên lớp toàn tiếng Trung.
   - max ~10.628 ảnh (lớp `others`), min = 1 ảnh, tỉ lệ max/min > 10.000.
   - median chỉ ~47 ảnh/lớp. 32% số lớp (<20 ảnh) chiếm <2% tổng ảnh.
   - Vì vậy nhánh `imb_ratio` + `.pt` kiểu CUB/Cars (tạo mất cân bằng NHÂN TẠO
     từ dữ liệu gốc cân bằng) là KHÔNG cần thiết và KHÔNG áp dụng ở đây.
   - Dataloader này giữ NGUYÊN phân bố tự nhiên: subsample labelled theo
     uniform trên instances (giữ long-tail), unlabelled = phần còn lại.

2. Vấn đề font tiếng Trung + thư mục trên đĩa bị mã hóa sai:
   - `meta.csv` là UTF-8 chuẩn: đọc BẮT BUỘC `encoding='utf-8-sig'`.
     Mở bằng Excel/PowerShell mặc định sẽ ra `???`.
   - Tên thư mục trên đĩa (`rp2k_dataset/all/train/<tên>/...`) đã bị lỗi font
     (ví dụ `555n+êså¦té½n+ë` thay vì `555（冰炫）`), nên KHÔNG thể dùng
     `os.path.join(root, image_path)` trực tiếp.
   - Tên FILE ảnh (basename, ví dụ `0514_10877_0.99999917.jpg`) phần lớn là ASCII
     và khớp giữa meta và đĩa. Dataloader này resolve bằng basename + vote
     theo folder (xem `_resolve_meta_to_fs`).
   - File rác trên đĩa (ảnh chụp màn hình `.png` kiểu `截屏2019-...`, file ngoài
     meta) không có trong meta sẽ bị BỎ QUA vì ta drive-by-meta.
   - File tên có chữ Trung (`*副本*.jpg`, `微信图片_*.jpg`, ...) bị garbled trên
     đĩa (`*së»µ£¼*.jpg`, ...) được khôi phục bằng fallback key
     (digits, is_copy, copy_num) + ràng buộc folder vote đúng class — thực tế
     khôi phục 256/256 train + 41/41 test, thiếu 0 dòng. Chỉ khi `strict=True`
     và vẫn thiếu mới raise.

3. Giao thức GCD (giống hệt nhánh balanced của cub.py / stanford_cars.py):
   - `whole_train` = toàn bộ split train đã resolve.
   - `train_labelled` = lọc `train_classes` (known) rồi subsample ngẫu nhiên
     `prop_train_labels` (mặc định 0.5, uniform trên instances).
   - `train_unlabelled` = `whole_train - train_labelled` (gồm known còn lại
     + toàn bộ novel). Không giao nhau với labelled theo `uq_idxs`.
   - `test` = toàn bộ split test đã resolve.
   - `get_datasets()` ở ngoài sẽ gán `target_transform` remap về
     0..(K+N-1) nên ở đây KHÔNG remap sớm (giống cars, khác cub có +1 offset).

Quy ước class id:
------------------
- Canonical id 0..N-1 = thứ tự `sorted()` trên UNION train+test `instance_id`
  (deterministic, không phụ thuộc tần suất). N=2233 với bản hiện tại.
- `train_classes=range(1116)`, `unlabeled_classes=range(1116, 2233)` là split
  mặc định nửa-nửa (xem `get_rp2k_default_splits` + `get_class_splits`).
- Lưu ý: có 71 lớp chỉ có ở train và 8 lớp chỉ có ở test nên một vài
  known/novel sẽ có 0 mẫu ở một split — dataloader chỉ warning, không tự lọc,
  để người dùng chủ động quyết định (lọc `min_samples`, gộp `others`, ...).

Tham khảo: `data/cub.py` (CustomCub2011 + subsample_*), 
`data/stanford_cars.py` (CarsDataset + canonical reorder + imb branch).
"""

import os
import re
from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

from data.data_utils import subsample_instances
from config import rp2k_root


# ---------------------------------------------------------------------------
# Cache toàn cục trong process (tránh đọc meta / quét đĩa 2 lần khi dựng
# cả train + test trong cùng một lần gọi get_rp2k_datasets).
# ---------------------------------------------------------------------------
_META_CACHE = {}
_FS_INDEX_CACHE = {}
_CANONICAL_CACHE = {}


# ---------------------------------------------------------------------------
# 1. Đọc meta.csv (UTF-8) + canonical class mapping
# ---------------------------------------------------------------------------

def _meta_path(root):
    """Đường dẫn meta.csv (cho phép root trỏ thẳng vào thư mục chứa meta)."""
    cand1 = os.path.join(root, 'meta.csv')
    if os.path.isfile(cand1):
        return cand1
    cand2 = os.path.join(root, 'RP2k', 'meta.csv')
    if os.path.isfile(cand2):
        return cand2
    return cand1  # để báo lỗi rõ ràng ở nơi đọc


def _load_meta(root):
    """Đọc meta.csv, thêm cột `split` và `basename`. Cache theo root."""
    root = os.path.expanduser(root)
    if root in _META_CACHE:
        return _META_CACHE[root].copy()

    mp = _meta_path(root)
    if not os.path.isfile(mp):
        raise FileNotFoundError(
            f'Không tìm thấy meta.csv ở {mp}. '
            f'Đặt root = thư mục chứa meta.csv (ví dụ D:/data/RP2k).')

    # utf-8-sig: chịu được cả file có/không BOM; đọc đúng tên tiếng Trung.
    df = pd.read_csv(mp, encoding='utf-8-sig')
    need = {'image_path', 'instance_id', 'domain'}
    if not need.issubset(set(df.columns)):
        raise ValueError(f'meta.csv thiếu cột, cần {need}, có {list(df.columns)}')

    df['image_path'] = df['image_path'].astype(str).str.replace('\\', '/', regex=False)
    # split = phần thứ 2 của 'all/train/...' hoặc 'all/test/...'
    df['split'] = df['image_path'].str.split('/').str[1]
    df['basename'] = df['image_path'].str.split('/').str[-1]
    # folder ghi trong meta (tên Trung chuẩn, DÙNG ĐỂ VOTE, không dùng để mở file)
    df['meta_folder'] = df['image_path'].str.split('/').str[2]

    _META_CACHE[root] = df
    return df.copy()


def _canonical_classes(meta_df):
    """Sorted union instance_id -> (classes, class_to_idx, idx_to_class)."""
    classes = sorted(meta_df['instance_id'].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    return classes, class_to_idx, idx_to_class


def _get_canonical(root):
    if root in _CANONICAL_CACHE:
        return _CANONICAL_CACHE[root]
    meta = _load_meta(root)
    out = _canonical_classes(meta)
    _CANONICAL_CACHE[root] = out
    return out


# ---------------------------------------------------------------------------
# 2. Quét đĩa + vote folder -> class (xử lý lỗi font thư mục)
# ---------------------------------------------------------------------------

def _scan_split_fs(root, split):
    """Quét 1 split trên đĩa.

    Trả về:
      basename_to_paths: dict basename -> list[full_path]
      folder_to_files:   dict folder_abs_path -> list[basename]
    Cache theo (root, split) vì quét ~340k file mất vài giây.
    """
    key = (os.path.expanduser(root), split)
    if key in _FS_INDEX_CACHE:
        return _FS_INDEX_CACHE[key]

    split_dir = os.path.join(os.path.expanduser(root), 'rp2k_dataset', 'all', split)
    if not os.path.isdir(split_dir):
        # fallback: root có thể đã trỏ thẳng vào rp2k_dataset/all
        alt = os.path.join(os.path.expanduser(root), split)
        if os.path.isdir(alt):
            split_dir = alt
        else:
            raise FileNotFoundError(
                f'Không tìm thấy thư mục split: {split_dir} (đã thử {alt})')

    basename_to_paths = defaultdict(list)
    folder_to_files = {}
    for cur_root, _dirs, files in os.walk(split_dir):
        if not files:
            continue
        basenames = []
        for fn in files:
            # Bỏ file hệ thống macOS/Windows nếu có
            if fn.startswith('._') or fn in ('.DS_Store', 'Thumbs.db'):
                continue
            full = os.path.join(cur_root, fn)
            basename_to_paths[fn].append(full)
            basenames.append(fn)
        if basenames:
            folder_to_files[cur_root] = basenames

    out = (dict(basename_to_paths), folder_to_files)
    _FS_INDEX_CACHE[key] = out
    return out


def _vote_folder_to_class(df_split, folder_to_files):
    """Vote mỗi folder đĩa -> instance_id bằng overlap basename.

    Vì basename trùng ~0.9% giữa các lớp nên dùng majority vote trên toàn bộ
    file trong folder, không quyết định bằng 1 file đơn lẻ.
    Trả về dict folder_abs_path -> instance_id (chỉ folder vote được).
    """
    # basename -> Counter(instance_id) từ meta (train/test riêng theo df_split)
    b2c = defaultdict(Counter)
    for bn, cls in zip(df_split['basename'].tolist(), df_split['instance_id'].tolist()):
        b2c[bn][cls] += 1

    folder_vote = {}
    for folder, basenames in folder_to_files.items():
        votes = Counter()
        for bn in basenames:
            if bn in b2c:
                # nếu basename trùng nhiều lớp, mỗi lớp 1 phiếu (tránh bias)
                for cls in b2c[bn].keys():
                    votes[cls] += 1
        if votes:
            folder_vote[folder] = votes.most_common(1)[0][0]
    return folder_vote


def _resolve_meta_to_fs(df_split, basename_to_paths, folder_vote,
                          folder_to_files=None, verbose=True):
    """Resolve từng dòng meta -> đường dẫn thật trên đĩa (2 passes).

    Pass 1 (exact): khớp basename chính xác. Ưu tiên path nằm trong folder đã
      vote đúng class khi basename trùng (~0.9% trường hợp `0514_*.jpg` dùng
      chung cho nhiều lớp).
    Pass 2 (fallback, cho file tên có chữ Trung bị lỗi font trên đĩa):
      - `*副本*.jpg` trong meta <-> `*së»µ£¼*.jpg` / `*tÜäsë»µ£¼*.jpg` trên đĩa.
      - `微信图片_xxx.jpg`, `金标生抽.jpg`, ... <-> tên garbled cùng folder.
      - Key fallback = (digits, is_copy, copy_num). Ví dụ
        `1103_74556 - 副本 (2).jpg` và `1103_74556 - së»µ£¼ (2).jpg` cùng key
        `(('1103','74556','2'), True, '2') nên khớp nhau.
      - Chỉ chấp nhận candidate NẰM TRONG folder đã vote đúng class (đảm bảo
        nhãn lớp đúng; sai instance front/back trong cùng lớp vẫn chấp nhận
        được cho bài toán phân loại). Nếu key rỗng (tên thuần Trung không số)
        mà có nhiều candidate toàn cục thì BẮT BUỘC phải qua folder vote,
        nếu không sẽ bỏ qua để tránh gán nhầm lớp.
    Trả về (resolved_paths, missing_mask).
    """
    # đảo vote: class -> set(folders)
    class_to_folders = defaultdict(set)
    for folder, cls in folder_vote.items():
        class_to_folders[cls].add(folder)

    basenames = df_split['basename'].tolist()
    classes = df_split['instance_id'].tolist()

    resolved = [None] * len(df_split)
    missing = np.zeros(len(df_split), dtype=bool)
    n_dup_fallback = 0

    # ---- Pass 1: exact ----
    still = []
    for i, (bn, cls) in enumerate(zip(basenames, classes)):
        cands = basename_to_paths.get(bn, [])
        if not cands:
            still.append(i)
            continue
        if len(cands) == 1:
            resolved[i] = cands[0]
            continue
        good_folders = class_to_folders.get(cls, set())
        picked = next((p for p in cands if os.path.dirname(p) in good_folders), None)
        if picked is None:
            picked = cands[0]
            n_dup_fallback += 1
        resolved[i] = picked

    if verbose and n_dup_fallback:
        print(f'[RP2k] {n_dup_fallback} dòng basename trùng phải lấy candidate đầu tiên '
              f'(không nằm trong folder vote đúng class).')

    if not still:
        return resolved, missing

    # ---- Pass 2: fallback cho tên có chữ Trung ----
    if folder_to_files is None:
        folder_to_files = {}

    # index fallback toàn cục: key -> list[full_path]
    fb_index = defaultdict(list)
    # index theo folder để tra nhanh: folder -> list[(fb_key, full_path)]
    folder_fb = {}
    for bn, paths in basename_to_paths.items():
        k = _fallback_key(bn, _is_copy_fs)
        for p in paths:
            fb_index[k].append(p)
    for folder, bns in folder_to_files.items():
        lst = []
        for bn in bns:
            # tìm 1 path đại diện cho bn trong folder này
            for p in basename_to_paths.get(bn, []):
                if os.path.dirname(p) == folder:
                    lst.append((_fallback_key(bn, _is_copy_fs), p))
                    break
        folder_fb[folder] = lst

    n_fb_ok, n_fb_dup = 0, 0
    for i in still:
        bn, cls = basenames[i], classes[i]
        k = _fallback_key(bn, _is_copy_meta)
        # 1) tìm trong folder vote đúng class trước
        cands = []
        for fo in class_to_folders.get(cls, set()):
            for fk, p in folder_fb.get(fo, []):
                if fk == k:
                    cands.append(p)
        if len(cands) == 1:
            resolved[i] = cands[0]
            n_fb_ok += 1
            continue
        if len(cands) > 1:
            # cùng lớp nên lấy cái đầu vẫn đúng nhãn (vd 正面/反面)
            resolved[i] = sorted(cands)[0]
            n_fb_ok += 1
            n_fb_dup += 1
            continue
        # 2) không có trong folder vote: chỉ nhận nếu key toàn cục duy nhất
        #    và key có số (tránh tên thuần Trung rỗng như '模板.jpg' gán bừa)
        glob = fb_index.get(k, [])
        if len(glob) == 1 and k[0]:
            resolved[i] = glob[0]
            n_fb_ok += 1
            continue
        missing[i] = True

    if verbose:
        print(f'[RP2k] fallback tên Trung: khôi phục {n_fb_ok}/{len(still)} dòng '
              f'(trong đó {n_fb_dup} dòng nhiều candidate cùng lớp, lấy 1). '
              f'Còn thiếu {int(missing.sum())} dòng.')
    return resolved, missing


def _is_copy_meta(basename):
    """File copy Windows trong meta: chứa '副本'."""
    return '副本' in basename


def _is_copy_fs(basename):
    """File copy trên đĩa: '副本' còn nguyên hoặc đã garbled.

    Quan sát thực tế: '副本' -> 'së»µ£¼', '的副本' -> 'tÜäsë»µ£¼'.
    Dấu hiệu chung: '»µ' / 'Üä' / 'së»' xuất hiện trong tên garbled.
    """
    return ('副本' in basename) or ('»µ' in basename) \
        or ('Üä' in basename) or ('së»' in basename)


def _digits_key(basename):
    """Tuple các dãy số trong tên file, vd '1103_74556 (2).jpg' -> (...)."""
    return tuple(re.findall(r'\d+', basename))


def _copy_num(basename):
    """Số copy trailing: '(2).jpg' -> '2', '副本 4.jpg'/'¼ 4.jpg' -> '4'."""
    m = re.search(r'\((\d+)\)\s*\.[A-Za-z]+$', basename)
    if m:
        return m.group(1)
    m = re.search(r'[副本¼]\s*(\d+)\.[A-Za-z]+$', basename)
    if m:
        return m.group(1)
    return None


def _fallback_key(basename, is_copy_fn):
    """Key khớp meta (Trung chuẩn) với đĩa (garbled)."""
    return (_digits_key(basename), bool(is_copy_fn(basename)), _copy_num(basename))


# ---------------------------------------------------------------------------
# 3. Dataset chính (API giống CustomCub2011 / CarsDataset)
# ---------------------------------------------------------------------------

class RP2KDataset(Dataset):
    """Một split (train/test) của RP2k, drive-by-meta + resolve qua basename.

    Attributes sau khi init (giống cub/cars để `get_datasets` dùng được):
      data (list[str]): đường dẫn ảnh thật đã resolve.
      target (list[int]): class id canonical 0-based (CHƯA remap GCD).
      uq_idxs (np.ndarray): 0..N-1, dùng để tách labelled/unlabelled.
      target_transform: callable hoặc None, do get_datasets gán sau.
      classes / class_to_idx / idx_to_class: mapping canonical toàn bộ RP2k.
      instance_ids (list[str]): tên Trung từng mẫu (song song với data).
    """

    def __init__(self, root=rp2k_root, split='train', transform=None,
                 loader=default_loader, strict=False, verbose=True,
                 skip_missing=True):
        if split not in ('train', 'test'):
            raise ValueError("split phải là 'train' hoặc 'test'")
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.loader = loader
        self.target_transform = None

        meta = _load_meta(self.root)
        self.classes, self.class_to_idx, self.idx_to_class = _get_canonical(self.root)

        df_split = meta[meta['split'] == split].reset_index(drop=True)
        if len(df_split) == 0:
            raise ValueError(f'meta.csv không có dòng nào split={split!r}')

        basename_to_paths, folder_to_files = _scan_split_fs(self.root, split)
        folder_vote = _vote_folder_to_class(df_split, folder_to_files)
        resolved, missing = _resolve_meta_to_fs(
            df_split, basename_to_paths, folder_vote,
            folder_to_files=folder_to_files, verbose=verbose)

        if verbose:
            n_vote = len(folder_vote)
            n_folders = len(folder_to_files)
            print(f'[RP2k:{split}] folders đĩa: {n_folders}, vote được: {n_vote}, '
                  f'dòng meta: {len(df_split)}, mất file: {int(missing.sum())}')

        if missing.any():
            miss_df = df_split[missing]
            # liệt kê vài mẫu để debug (tên Trung ok vì console UTF-8)
            examples = miss_df['image_path'].head(5).tolist()
            msg = (f'[RP2k:{split}] thiếu {int(missing.sum())}/{len(df_split)} file '
                   f'trên đĩa (vd: {examples}).')
            if strict:
                raise FileNotFoundError(msg + ' strict=True nên dừng.')
            if verbose:
                print(msg + ' Đã bỏ qua (skip_missing=True).')
        if skip_missing:
            keep = ~missing
            df_split = df_split[keep].reset_index(drop=True)
            resolved = [p for p, m in zip(resolved, keep.tolist()) if m]

        # chốt data/target theo canonical id
        self.data = resolved
        self.instance_ids = df_split['instance_id'].tolist()
        self.target = [self.class_to_idx[c] for c in self.instance_ids]
        # .targets (0-based, alias của .target): pseudo.py build_uq_to_true_label /
        # count_labeled_per_class đọc nhánh `targets` TRƯỚC nhánh `target`-1-based
        # (quy ước Cars). Không có attr này thì audit map + phân bố pseudo bị lệch
        # -1 class. Remap GCD qua target_transform là identity trên rp2k
        # (dict[c]=c) nên alias này đúng cả trước lẫn sau khi get_datasets gán.
        self.targets = list(self.target)
        self.uq_idxs = np.arange(len(self.data))

        # thống kê nhanh imbalance của split này (giữ lại để debug)
        counts = Counter(self.target)
        vals = np.array(list(counts.values()))
        self.split_stats = {
            'n_samples': len(self.data),
            'n_classes': len(counts),
            'min': int(vals.min()) if len(vals) else 0,
            'max': int(vals.max()) if len(vals) else 0,
            'mean': float(vals.mean()) if len(vals) else 0.0,
            'median': float(np.median(vals)) if len(vals) else 0.0,
        }
        if verbose:
            s = self.split_stats
            print(f"[RP2k:{split}] n={s['n_samples']}, classes={s['n_classes']}, "
                  f"min={s['min']}, median={s['median']:.1f}, "
                  f"mean={s['mean']:.1f}, max={s['max']}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        target = self.target[idx]
        try:
            img = self.loader(path)
        except Exception as e:
            raise RuntimeError(f'Không đọc được ảnh RP2k: {path}') from e

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, self.uq_idxs[idx]

    def get_class_counts(self):
        """Counter {canonical_id: số ảnh} trong split này (giữ imbalance)."""
        return Counter(self.target)

    def get_class_name(self, target_id):
        """Tên tiếng Trung của 1 canonical id."""
        return self.idx_to_class[int(target_id)]


# ---------------------------------------------------------------------------
# 4. Subsample helpers (khớp API cub.py / stanford_cars.py)
# ---------------------------------------------------------------------------

def subsample_dataset(dataset, idxs):
    """Giữ lại subset theo positional idxs (giống CarsDataset)."""
    idxs = np.asarray(idxs, dtype=np.int64)
    dataset.data = np.array(dataset.data, dtype=object)[idxs].tolist()
    dataset.target = np.array(dataset.target)[idxs].tolist()
    # targets (alias pseudo.py, xem __init__) phải đồng bộ theo target
    if hasattr(dataset, 'targets') and dataset.targets is not None:
        dataset.targets = list(dataset.target)
    # instance_ids song song (nếu có)
    if hasattr(dataset, 'instance_ids') and dataset.instance_ids is not None:
        dataset.instance_ids = np.array(dataset.instance_ids, dtype=object)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]
    return dataset


def subsample_classes(dataset, include_classes):
    """Lọc theo canonical ids (0-based, KHÔNG +1 như CUB/Cars, KHÔNG remap).

    Remap về 0..K+N-1 do `get_datasets` đảm nhiệm qua target_transform.
    """
    include_set = set(int(c) for c in include_classes)
    cls_idxs = [i for i, t in enumerate(dataset.target) if int(t) in include_set]
    if len(cls_idxs) == 0:
        raise ValueError(f'subsample_classes: không còn mẫu nào với {len(include_set)} lớp yêu cầu.')
    # cảnh báo lớp yêu cầu nhưng không có mẫu (vd lớp test-only khi lấy train)
    have = set(int(t) for t in dataset.target)
    missing_cls = include_set - have
    if missing_cls:
        print(f'[RP2k] cảnh báo: {len(missing_cls)} lớp yêu cầu không có mẫu trong split '
              f'{getattr(dataset, "split", "?")} (vd ids {sorted(list(missing_cls))[:5]}). '
              f'Nguyên nhân thường: 71 lớp train-only / 8 lớp test-only.')
    return subsample_dataset(dataset, np.array(cls_idxs, dtype=np.int64))


def get_train_val_indices(train_dataset, val_split=0.2):
    """Stratified split labelled -> train/val theo tỉ lệ (giống cub/cars).

    Xử lý lớp nhỏ: int(val_split * n) có thể =0 với n=1..4 -> lớp đó không có
    val (giữ hết cho train). Đây là hành vi mong muốn với long-tail RP2k,
    tránh crash `np.random.choice(..., size=0)` hay mất hẳn lớp 1 mẫu.
    """
    train_classes = np.unique(train_dataset.target)
    train_idxs, val_idxs = [], []
    for cls in train_classes:
        cls_idxs = np.where(np.array(train_dataset.target) == cls)[0]
        n_val = int(val_split * len(cls_idxs))
        if n_val <= 0:
            train_idxs.extend(cls_idxs.tolist())
            continue
        # đảm bảo không lấy hết (giữ ít nhất 1 mẫu train)
        n_val = min(n_val, len(cls_idxs) - 1)
        v_ = np.random.choice(cls_idxs, replace=False, size=n_val)
        v_set = set(v_.tolist())
        t_ = [x for x in cls_idxs.tolist() if x not in v_set]
        train_idxs.extend(t_)
        val_idxs.extend(v_.tolist())
    return train_idxs, val_idxs


# ---------------------------------------------------------------------------
# 5. Split mặc định + hàm dựng GCD datasets (KHÔNG imb_ratio)
# ---------------------------------------------------------------------------

def get_rp2k_default_splits(root=rp2k_root, n_known=None):
    """Chia canonical classes nửa-nửa: known đầu, novel sau.

    - Sắp xếp theo `sorted()` (Unicode) nên deterministic, không bias tần suất.
    - Mặc định n_known = N//2 (2233 -> 1116 known / 1117 novel).
    - Trả về (train_classes, unlabeled_classes) là `range` như cub/cars.
    """
    classes, _, _ = _get_canonical(os.path.expanduser(root))
    n = len(classes)
    if n_known is None:
        n_known = n // 2
    if not (0 < n_known < n):
        raise ValueError(f'n_known={n_known} không hợp lệ với N={n}')
    return range(n_known), range(n_known, n)


def get_rp2k_datasets(train_transform, test_transform,
                      train_classes=None, prop_train_labels=0.5,
                      split_train_val=False, seed=0, imb_ratio=None):
    """Dựng dict GCD cho RP2k, giữ imbalance tự nhiên.

    Args:
      train_transform / test_transform: transform cho train/test. Train sẽ được
        bọc ContrastiveLearningViewGenerator ở train.py nên ở đây chỉ nhận base.
      train_classes: iterable canonical ids làm known. None -> nửa đầu (1116).
      prop_train_labels: tỉ lệ instances KNOWN được gán nhãn (uniform random,
        GIỮ long-tail). 0.5 như SimGCD mặc định.
      split_train_val: True -> tách labelled thành train/val stratified 80/20.
      seed: seed cho stratified + subsample (subsample_instances dùng seed 0
        nội bộ như repo hiện tại — giữ để nhất quán với cub/cars).
      imb_ratio: PHẢI là None. RP2k đã mất cân bằng sẵn nên cấm truyền
        --imb_ratio (tránh nhầm với nhánh .pt của cub/cars).

    Returns:
      dict {'train_labelled', 'train_unlabelled', 'val', 'test'} giống cub/cars.
    """
    if imb_ratio is not None:
        raise ValueError(
            'RP2k đã mất cân bằng tự nhiên, KHÔNG dùng --imb_ratio / file .pt. '
            'Hãy chạy với imb_ratio=None (mặc định).')

    np.random.seed(seed)

    if train_classes is None:
        train_classes, _ = get_rp2k_default_splits(rp2k_root)
    train_classes = list(train_classes)

    # 1) Toàn bộ train (giữ nguyên long-tail)
    whole_training_set = RP2KDataset(root=rp2k_root, split='train',
                                     transform=train_transform, verbose=True)

    # 2) Labelled = known classes + subsample prop_train_labels (uniform)
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set),
                                               include_classes=train_classes)
    subsample_indices = subsample_instances(
        train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(deepcopy(train_dataset_labelled),
                                               subsample_indices)

    # 3) Tách train/val stratified (optional,jaminlong-tail từng lớp)
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled),
                                                     np.array(train_idxs, dtype=np.int64))
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled),
                                                   np.array(val_idxs, dtype=np.int64)) \
        if len(val_idxs) > 0 else None
    if val_dataset_labelled_split is not None:
        val_dataset_labelled_split.transform = test_transform

    # 4) Unlabelled = whole - labelled (gồm known còn lại + toàn bộ novel)
    #    So sánh bằng uq_idxs gốc như cub/cars để đảm bảo không giao nhau.
    unlabelled_indices = set(whole_training_set.uq_idxs.tolist()) - \
        set(train_dataset_labelled.uq_idxs.tolist())
    train_dataset_unlabelled = subsample_dataset(
        deepcopy(whole_training_set),
        np.array(sorted(unlabelled_indices), dtype=np.int64))

    # 5) Test đầy đủ (giữ imbalance test tự nhiên)
    test_dataset = RP2KDataset(root=rp2k_root, split='test',
                               transform=test_transform, verbose=True)

    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }
    return all_datasets


if __name__ == '__main__':
    # Smoke test nhanh: python -m data.rp2k  (chạy từ repo root SimGCD)
    _tc, _uc = get_rp2k_default_splits()
    print(f'default splits: known={len(_tc)}, novel={len(_uc)}')
    x = get_rp2k_datasets(None, None, train_classes=_tc,
                          prop_train_labels=0.5, split_train_val=False)
    print('Lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')
    print('Overlap labelled/unlabelled (phải rỗng):')
    print(set(x['train_labelled'].uq_idxs.tolist()) &
          set(x['train_unlabelled'].uq_idxs.tolist()))
    print('Tổng train (labelled+unlabelled):',
          len(set(x['train_labelled'].uq_idxs.tolist())) +
          len(set(x['train_unlabelled'].uq_idxs.tolist())))
    print(f"Num Labelled Classes: {len(set(x['train_labelled'].target))}")
    print(f"Num Unlabelled Classes: {len(set(x['train_unlabelled'].target))}")
