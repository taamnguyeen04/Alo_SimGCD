import os
import pandas as pd
import numpy as np
import torch
from copy import deepcopy

from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

from data.data_utils import subsample_instances
from config import cub_root


class CustomCub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, target_transform=None, loader=default_loader, download=True):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.loader = loader
        self.train = train


        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.uq_idxs = np.array(range(len(self)))

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.uq_idxs[idx]


def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.data = dataset.data[mask]
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset


def subsample_classes(dataset, include_classes=range(160)):

    include_classes_cub = np.array(include_classes) + 1     # CUB classes are indexed 1 --> 200 instead of 0 --> 199
    cls_idxs = [x for x, (_, r) in enumerate(dataset.data.iterrows()) if int(r['target']) in include_classes_cub]

    # TODO: For now have no target transform
    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.data['target'])

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.data['target'] == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_cub_imb_class_splits(imb_ratio):
    """Known/novel classes read straight from the precomputed BaCon split.

    Eval identity MUST match supervision: SSB classes do NOT apply when an
    imb split is used (overlap is only ~45/100). Positional indices in the
    .pt files address the sorted-image_id train ordering with 0-based
    targets (label-1) — same convention as make_cub_lt_splits.py and
    CustomCub2011 below (target-1).
    """
    imb_ratio = int(imb_ratio)
    split_dir = os.path.join('data_uq_idxs_bacon', f'cub200_k100_imb{imb_ratio}')
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f'Precomputed CUB imbalance split not found: {split_dir}')
    l_k = torch.load(os.path.join(split_dir, 'l_k_uq_idxs.pt'),
                     map_location='cpu', weights_only=False)
    unl_k = torch.load(os.path.join(split_dir, 'unl_k_uq_idxs.pt'),
                       map_location='cpu', weights_only=False)
    unl_unk = torch.load(os.path.join(split_dir, 'unl_unk_uq_idxs.pt'),
                         map_location='cpu', weights_only=False)

    txt_dir = os.path.join(cub_root, 'CUB_200_2011')
    labels, is_train = {}, {}
    with open(os.path.join(txt_dir, 'image_class_labels.txt')) as f:
        for line in f:
            k, v = line.split()
            labels[int(k)] = int(v) - 1
    with open(os.path.join(txt_dir, 'train_test_split.txt')) as f:
        for line in f:
            k, v = line.split()
            is_train[int(k)] = int(v)
    targets = np.array([labels[i] for i in sorted(labels) if is_train.get(i) == 1])

    known = sorted(set(targets[np.asarray(l_k)].tolist())
                   | set(targets[np.asarray(unl_k)].tolist()))
    novel = sorted(set(targets[np.asarray(unl_unk)].tolist()))
    assert len(known) == 100 and len(novel) == 100, \
        f'imb{imb_ratio}: expected 100/100 classes, got {len(known)}/{len(novel)}'
    assert not (set(known) & set(novel)), 'imb split known/novel overlap!'
    return known, novel


def get_cub_datasets(train_transform, test_transform, train_classes=range(160), prop_train_labels=0.8,
                    split_train_val=False, seed=0, download=False, imb_ratio=None):

    if imb_ratio is not None:
        if not isinstance(imb_ratio, int):
            imb_ratio = int(imb_ratio)

        split_dir = os.path.join('data_uq_idxs_bacon', f'cub200_k100_imb{imb_ratio}')
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f'Precomputed CUB imbalance split not found: {split_dir}')

        l_k = np.array(torch.load(os.path.join(split_dir, 'l_k_uq_idxs.pt'),
                                    map_location='cpu', weights_only=False))
        unl_k = np.array(torch.load(os.path.join(split_dir, 'unl_k_uq_idxs.pt'),
                                    map_location='cpu', weights_only=False))
        unl_unk = np.array(torch.load(os.path.join(split_dir, 'unl_unk_uq_idxs.pt'),
                                      map_location='cpu', weights_only=False))

        whole_training_set = CustomCub2011(root=cub_root, transform=train_transform, train=True, download=download)
        train_dataset_labelled = subsample_dataset(deepcopy(whole_training_set), l_k)
        train_dataset_unlabelled = subsample_dataset(
            deepcopy(whole_training_set),
            np.concatenate([unl_k, unl_unk]).astype(np.int64)
        )
        test_dataset = CustomCub2011(root=cub_root, transform=test_transform, train=False)

        all_datasets = {
            'train_labelled': train_dataset_labelled,
            'train_unlabelled': train_dataset_unlabelled,
            'val': None,
            'test': test_dataset,
        }
        return all_datasets

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = CustomCub2011(root=cub_root, transform=train_transform, train=True, download=download)

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
    test_dataset = CustomCub2011(root=cub_root, transform=test_transform, train=False)

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

    x = get_cub_datasets(None, None, split_train_val=False,
                         train_classes=range(100), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].data["target"].values))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].data["target"].values))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')