# -----------------
# DATASET ROOTS
# -----------------
import os as _os
cifar_10_root = '${DATASET_DIR}/cifar10'
cifar_100_root = '${DATASET_DIR}/cifar100'
# NOTE (generality track): env override for Modal/Docker runs; default is the
# PARENT of CUB_200_2011 (code joins root/'CUB_200_2011'/... itself).
# Local Windows default: D:/data  (fixes legacy '$D:/data/CUB_200_2011' value
# which had a stray '$' and pointed one level too deep).
cub_root = _os.environ.get('SIMGCD_CUB_ROOT', 'D:/data')
# NOTE (generality track, scars/aircraft): same override pattern as CUB. The
# legacy '${DATASET_DIR}/...' placeholders below are NOT expanded by Python
# (no expandvars call anywhere), so without these envs both loaders get garbage
# paths on Modal. Local default keeps legacy behavior.
car_root = _os.environ.get('SIMGCD_SCARS_ROOT', '${DATASET_DIR}/cars')
aircraft_root = _os.environ.get('SIMGCD_AIRCRAFT_ROOT', '${DATASET_DIR}/fgvc-aircraft-2013b')
herbarium_dataroot = '${DATASET_DIR}/herbarium_19'
imagenet_root = '${DATASET_DIR}/ImageNet'
# NOTE (rp2k): Retail Product 2000, mất cân bằng tự nhiên, tên lớp tiếng Trung
# (meta.csv UTF-8). Local Windows default: D:/data/RP2k (thư mục chứa meta.csv
# + rp2k_dataset/all/{train,test}/...). Override bằng SIMGCD_RP2K_ROOT.
rp2k_root = _os.environ.get('SIMGCD_RP2K_ROOT', 'D:/data/RP2k')

# OSR Split dir
osr_split_dir = 'data/ssb_splits'

# -----------------
# OTHER PATHS
# -----------------
exp_root = 'dev_outputs' # All logs and checkpoints will be saved here