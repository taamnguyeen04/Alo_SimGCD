import os
import pandas as pd
import numpy as np
import torch
from copy import deepcopy
from scipy import io as mat_io

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

from data.data_utils import subsample_instances
from config import car_root

class CarsDataset(Dataset):
    """
        Cars Dataset
    """
    def __init__(self, train=True, limit=0, data_dir=car_root, transform=None):

        # NOTE (generality track, scars): the Modal volume carries a repackaged
        # layout — car_data/car_data/{train,test}/<class>/*.jpg plus
        # anno_{train,test}.csv (filename,x1,y1,x2,y2,1-based-class) — instead of
        # the official devkit/*.mat + cars_train//cars_test/ tree. Verified 1:1
        # against the official release (8144 train / 8041 test rows, labels
        # 1..196 fully covered; bboxes are unused by SimGCD, which reads only
        # filename + class from .mat). Official layout, when present, is used
        # unchanged; otherwise this branch reproduces identical (path,
        # 1-based-label) pairs so the SSB protocol is unaffected.
        repack_img_root = os.path.join(data_dir, 'car_data', 'car_data',
                                       'train' if train else 'test')
        repack_anno = os.path.join(data_dir, f'anno_{"train" if train else "test"}.csv')
        use_repack = (not os.path.exists(os.path.join(data_dir, 'devkit'))) \
            and os.path.isdir(repack_img_root) and os.path.isfile(repack_anno)

        self.loader = default_loader
        self.train = train
        self.transform = transform

        if use_repack:
            import csv as _csv
            index = {}
            for _root, _dirs, _files in os.walk(repack_img_root):
                for _f in _files:
                    if _f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        index.setdefault(_f, os.path.join(_root, _f))
            self.data, self.target = [], []
            with open(repack_anno, newline='') as _fh:
                for _i, _row in enumerate(_csv.reader(_fh)):
                    if limit and _i > limit:
                        break
                    _path = index.get(_row[0])
                    if _path is None:
                        raise FileNotFoundError(
                            f'[scars-repack] {_row[0]} listed in {repack_anno} '
                            f'but not found under {repack_img_root}')
                    self.data.append(_path)
                    self.target.append(int(_row[5]))  # 1-based, .mat convention
            _expected = 8144 if train else 8041
            assert len(self.data) == _expected, \
                f'[scars-repack] got {len(self.data)} rows, official split has {_expected}'
            assert set(self.target) == set(range(1, 197)), \
                '[scars-repack] label coverage is not the full official 1..196'
        else:
            metas = os.path.join(data_dir, 'devkit/cars_train_annos.mat') if train else os.path.join(data_dir, 'devkit/cars_test_annos_withlabels.mat')
            data_dir = os.path.join(data_dir, 'cars_train/') if train else os.path.join(data_dir, 'cars_test/')

            self.data_dir = data_dir
            self.data = []
            self.target = []

            if not isinstance(metas, str):
                raise Exception("Train metas must be string location !")
            labels_meta = mat_io.loadmat(metas)

            for idx, img_ in enumerate(labels_meta['annotations'][0]):
                if limit:
                    if idx > limit:
                        break

                # self.data.append(img_resized)
                self.data.append(data_dir + img_[5][0])
                # if self.mode == 'train':
                self.target.append(img_[4][0][0])

        self.uq_idxs = np.array(range(len(self)))
        self.target_transform = None

    def __getitem__(self, idx):

        image = self.loader(self.data[idx])
        target = self.target[idx] - 1

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        idx = self.uq_idxs[idx]

        return image, target, idx

    def __len__(self):
        return len(self.data)


def subsample_dataset(dataset, idxs):

    dataset.data = np.array(dataset.data)[idxs].tolist()
    dataset.target = np.array(dataset.target)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset, include_classes=range(160)):

    include_classes_cars = np.array(include_classes) + 1     # SCars classes are indexed 1 --> 196 instead of 0 --> 195
    cls_idxs = [x for x, t in enumerate(dataset.target) if t in include_classes_cars]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset

def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.target)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.target == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def _reorder_cars_canonical(dataset):
    """Reorder into BaCon canonical order (sorted basename) and reset uq_idxs.

    The precomputed .pt splits are POSITIONAL into BaCon's sorted-filename
    ordering; SimGCD's native order (.mat file order / anno-csv row order) is
    NOT verified to match (unlike CUB, checked 1:1). Canonicalizing here
    makes the imb branch layout-independent (official devkit AND repack).
    """
    names = [os.path.basename(p) for p in dataset.data]
    if len(set(names)) != len(names):
        raise ValueError('duplicate basenames in cars train set — canonical order ambiguous')
    order = sorted(range(len(names)), key=lambda i: names[i])
    dataset.data = [dataset.data[i] for i in order]
    dataset.target = [dataset.target[i] for i in order]
    dataset.uq_idxs = np.arange(len(order))
    return dataset


def _load_cars_imb_arrays(split_dir, dataset_size):
    """Load + fail-fast validate the 3 .pt split files (dup/range/overlap)."""
    def _load(name):
        path = os.path.join(split_dir, name)
        try:
            arr = np.asarray(torch.load(path, map_location='cpu',
                                        weights_only=False),
                             dtype=np.int64).reshape(-1)
        except TypeError:  # very old torch without weights_only
            arr = np.asarray(torch.load(path, map_location='cpu'),
                             dtype=np.int64).reshape(-1)
        if len(arr) != len(np.unique(arr)):
            raise ValueError(f'Duplicate indices in {path}')
        if len(arr) and (arr.min() < 0 or arr.max() >= dataset_size):
            raise ValueError(f'Out-of-range indices in {path} '
                             f'(expected [0, {dataset_size - 1}])')
        return arr
    groups = {n: _load(f) for n, f in
              [('l_k', 'l_k_uq_idxs.pt'), ('unl_k', 'unl_k_uq_idxs.pt'),
               ('unl_unk', 'unl_unk_uq_idxs.pt')]}
    pooled = np.concatenate([groups['l_k'], groups['unl_k'], groups['unl_unk']])
    if len(pooled) != len(np.unique(pooled)):
        raise ValueError(f'cars uq split groups overlap ({split_dir})')
    return groups


def get_scars_imb_class_splits(imb_ratio, k=98, n_classes=196):
    """Known/novel classes straight from the precomputed BaCon cars split.

    Eval identity MUST match supervision: SSB classes do NOT apply when an
    imb split is used. Derivation uses the same canonical ordering as the
    imb data branch below, so the two can never disagree.
    """
    imb_ratio = int(imb_ratio)
    split_dir = os.path.join('data_uq_idxs_bacon', f'cars196_k{k}_imb{imb_ratio}')
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f'Precomputed cars imbalance split not found: {split_dir}')
    whole = CarsDataset(data_dir=car_root, transform=None, train=True)
    if len(whole) != 8144:
        raise ValueError(f'cars train N={len(whole)} != 8144: wrong data source?')
    whole = _reorder_cars_canonical(whole)
    groups = _load_cars_imb_arrays(split_dir, len(whole))
    canon_targets = [(int(t) - 1) for t in whole.target]  # 1-based -> 0-based
    known = sorted(set(canon_targets[i] for i in groups['l_k'])
                   | set(canon_targets[i] for i in groups['unl_k']))
    novel = sorted(set(canon_targets[i] for i in groups['unl_unk']))
    # Fail-fast: a wrong-order split scatters labels, giving ~196 'known'
    # classes instead of exactly k (plus novel complement) — crash in minutes,
    # never train garbage for hours.
    if len(known) != k or len(novel) != n_classes - k \
            or (set(known) & set(novel)) or len(set(known) | set(novel)) != n_classes:
        raise ValueError(f'cars imb{imb_ratio}: bad known/novel '
                         f'({len(known)}/{len(novel)} classes) — split/order mismatch?')
    return known, novel


def get_scars_datasets(train_transform, test_transform, train_classes=range(160), prop_train_labels=0.8,
                    split_train_val=False, seed=0, imb_ratio=None):

    np.random.seed(seed)

    if imb_ratio is not None:
        imb_ratio = int(imb_ratio)
        split_dir = os.path.join('data_uq_idxs_bacon', f'cars196_k98_imb{imb_ratio}')
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f'Precomputed cars imbalance split not found: {split_dir}')

        whole_training_set = CarsDataset(data_dir=car_root, transform=train_transform, train=True)
        if len(whole_training_set) != 8144:
            raise ValueError(f'cars train N={len(whole_training_set)} != 8144: wrong data source?')
        whole_training_set = _reorder_cars_canonical(whole_training_set)
        groups = _load_cars_imb_arrays(split_dir, len(whole_training_set))
        l_k, unl_k, unl_unk = groups['l_k'], groups['unl_k'], groups['unl_unk']

        train_dataset_labelled = subsample_dataset(deepcopy(whole_training_set), l_k)
        train_dataset_unlabelled = subsample_dataset(
            deepcopy(whole_training_set),
            np.concatenate([unl_k, unl_unk]).astype(np.int64)
        )
        test_dataset = CarsDataset(data_dir=car_root, transform=test_transform, train=False)

        all_datasets = {
            'train_labelled': train_dataset_labelled,
            'train_unlabelled': train_dataset_unlabelled,
            'val': None,
            'test': test_dataset,
        }
        return all_datasets

    # Init entire training set
    whole_training_set = CarsDataset(data_dir=car_root, transform=train_transform, train=True)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    test_dataset = CarsDataset(data_dir=car_root, transform=test_transform, train=False)

    # Either split train into train and val or use test set as val
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

    x = get_scars_datasets(None, None, train_classes=range(98), prop_train_labels=0.5, split_train_val=False)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].target))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].target))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')