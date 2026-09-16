# Parametric Classification for Generalized Category Discovery: A Baseline Study


<p align="center">
    <a href="https://openaccess.thecvf.com/content/ICCV2023/html/Wen_Parametric_Classification_for_Generalized_Category_Discovery_A_Baseline_Study_ICCV_2023_paper.html"><img src="https://img.shields.io/badge/-ICCV%202023-68488b"></a>
    <a href="https://arxiv.org/abs/2211.11727"><img src="https://img.shields.io/badge/arXiv-2211.11727-b31b1b"></a>
    <a href="https://wen-xin.info/simgcd"><img src="https://img.shields.io/badge/Project-Website-blue"></a>
  <a href="https://github.com/CVMI-Lab/SlotCon/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
</p>
<p align="center">
	Parametric Classification for Generalized Category Discovery: A Baseline Study (ICCV 2023)<br>
  By
  <a href="https://wen-xin.info">Xin Wen</a>*, 
  <a href="https://bzhao.me/">Bingchen Zhao</a>*, and 
  <a href="https://xjqi.github.io/">Xiaojuan Qi</a>.
</p>

![teaser](assets/teaser.jpg)

Generalized Category Discovery (GCD) aims to discover novel categories in unlabelled datasets using knowledge learned from labelled samples.
Previous studies argued that parametric classifiers are prone to overfitting to seen categories, and endorsed using a non-parametric classifier formed with semi-supervised $k$-means.

However, in this study, we investigate the failure of parametric classifiers, verify the effectiveness of previous design choices when high-quality supervision is available, and identify unreliable pseudo-labels as a key problem. We demonstrate that two prediction biases exist: the classifier tends to predict seen classes more often, and produces an imbalanced distribution across seen and novel categories. 
Based on these findings, we propose a simple yet effective parametric classification method that benefits from entropy regularisation, achieves state-of-the-art performance on multiple GCD benchmarks and shows strong robustness to unknown class numbers.
We hope the investigation and proposed simple framework can serve as a strong baseline to facilitate future studies in this field.

## Running

### Dependencies

```
pip install -r requirements.txt
```

### Config

Set paths to datasets and desired log directories in ```config.py```


### Datasets

We use fine-grained benchmarks in this paper, including:

* [The Semantic Shift Benchmark (SSB)](https://github.com/sgvaze/osr_closed_set_all_you_need#ssb) and [Herbarium19](https://www.kaggle.com/c/herbarium-2019-fgvc6)

We also use generic object recognition datasets, including:

* [CIFAR-10/100](https://pytorch.org/vision/stable/datasets.html) and [ImageNet-100/1K](https://image-net.org/download.php)


### Scripts

**Train the model**:

```
bash scripts/run_${DATASET_NAME}.sh
```

We found picking the model according to 'Old' class performance could lead to possible over-fitting, and since 'New' class labels on the held-out validation set should be assumed unavailable, we suggest not to perform model selection, and simply use the last-epoch model.

## Results
Our results:

<table><thead><tr><th>Source</th><th colspan="3">Paper (3 runs) </th><th colspan="3">Current Github (5 runs) </th></tr></thead><tbody><tr><td>Dataset</td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td></tr><tr><td>CIFAR10</td><td>97.1±0.0</td><td>95.1±0.1</td><td>98.1±0.1</td><td>97.0±0.1</td><td>93.9±0.1</td><td>98.5±0.1</td></tr><tr><td>CIFAR100</td><td>80.1±0.9</td><td>81.2±0.4</td><td>77.8±2.0</td><td>79.8±0.6</td><td>81.1±0.5</td><td>77.4±2.5</td></tr><tr><td>ImageNet-100</td><td>83.0±1.2</td><td>93.1±0.2</td><td>77.9±1.9</td><td>83.6±1.4</td><td>92.4±0.1</td><td>79.1±2.2</td></tr><tr><td>ImageNet-1K</td><td>57.1±0.1</td><td>77.3±0.1</td><td>46.9±0.2</td><td>57.0±0.4</td><td>77.1±0.1</td><td>46.9±0.5</td></tr><tr><td>CUB</td><td>60.3±0.1</td><td>65.6±0.9</td><td>57.7±0.4</td><td>61.5±0.5</td><td>65.7±0.5</td><td>59.4±0.8</td></tr><tr><td>Stanford Cars</td><td>53.8±2.2</td><td>71.9±1.7</td><td>45.0±2.4</td><td>53.4±1.6</td><td>71.5±1.6</td><td>44.6±1.7</td></tr><tr><td>FGVC-Aircraft</td><td>54.2±1.9</td><td>59.1±1.2</td><td>51.8±2.3</td><td>54.3±0.7</td><td>59.4±0.4</td><td>51.7±1.2</td></tr><tr><td>Herbarium 19</td><td>44.0±0.4</td><td>58.0±0.4</td><td>36.4±0.8</td><td>44.2±0.2</td><td>57.6±0.6</td><td>37.0±0.4</td></tr></tbody></table>

## Fixed imbalanced CUB split

Use `--uq_split cub200_k100_imb10` with `train.py`, or
`--uq-split cub200_k100_imb10` with the Modal launcher. This fixed split uses
867 labelled-known, 870 unlabelled-known, and 586 unlabelled-novel training
images from `data_uq_idxs_bacon/cub200_k100_imb10/`.
The `k100` split defines classes 0-99 as known and 100-199 as novel; when this
split is selected it intentionally overrides the semantic SSB class split.

Full model (known + novel pseudo-labels, parts, and momentum teacher) on Modal:

```bash
modal run -d modal_train_simgcd.py::launch \
  --dataset-name cub \
  --uq-split cub200_k100_imb10 \
  --backbone dinov2-vitb14 \
  --seed 0 \
  --memax-weight 2 \
  --epochs 200 \
  --exp-name-suffix CUB_IMB10_FULL \
  --enable-pseudo-labeling \
  --pseudo-mode 1 \
  --confidence-threshold 0.9 \
  --pseudo-top-ratio 0.8 \
  --max-samples-per-class 500 \
  --pseudo-update-freq 10 \
  --max-pseudo-iterations 20 \
  --pseudo-warmup-epoch 30 \
  --pseudo-conf-bar 0.5 \
  --pseudo-bar-k 2.0 \
  --pseudo-min-hi 10 \
  --enable-novel-pseudo \
  --novel-warmup-epoch 50 \
  --novel-update-freq 10 \
  --max-novel-iterations 6 \
  --novel-max-samples 200 \
  --novel-jaccard-th 0.6 \
  --novel-agree-th 0.7 \
  --novel-min-size 10 \
  --use-parts \
  --num-slots 3 \
  --part-lambda 0.5 \
  --tau-c 0.1 \
  --use-momentum-teacher \
  --teacher-m0 0.996
```

## Citing this work

If you find this repo useful for your research, please consider citing our paper:

```
@inproceedings{wen2023simgcd,
    author    = {Wen, Xin and Zhao, Bingchen and Qi, Xiaojuan},
    title     = {Parametric Classification for Generalized Category Discovery: A Baseline Study},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2023},
    pages     = {16590-16600}
}
```

## Acknowledgements

The codebase is largely built on this repo: https://github.com/sgvaze/generalized-category-discovery.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Rerun 4 cấu hình × 3 datasets (12 lệnh Modal + 12 lệnh server thường)

> Tất cả chạy từ chính thư mục này (`SimGCD/`). File `modal_train_simgcd.py` trong thư mục này là bản copy của `gcd-hosts/modal_train_simgcd.py` (đã chỉnh `LOCAL_REPO` về current dir) để folder tự chứa toàn bộ.
> Quy ước flag: Modal CLI dùng **gạch ngang** `-` (`--dataset-name`, `--enable-pseudo-labeling`; boolean chỉ gọi tên, không kèm `true`); `train.py` dùng **gạch dưới** `_` (`--dataset_name`, `--enable_pseudo_labeling`).
> Tên dataset SimGCD: `cub` (CUB-200), `scars` (Stanford Cars), `aircraft` (FGVC-Aircraft).

**Mapping 4 cấu hình → flag** (backbone chung `dinov2_vitb14`, `epochs=200`, `seed=0`):

| # | Cấu hình | Flag thêm vào |
|---|----------|----------------|
| A | SimGCD baseline | (không thêm gì) |
| B | Dual pseudo-only (known Mode-1 + novel, **không** parts/teacher) | `--enable_pseudo_labeling --pseudo_mode 1 ... --enable_novel_pseudo ...` |
| C | Multi-prototype + teacher-only (3 slots/lớp + teacher EMA, **không** pseudo) | `--use_parts --num_slots 3 --use_momentum_teacher` |
| D | Dual + multi-prototype + teacher (full ⭐) | cả hai cụm trên |

**Giả định chung (đã gõ rõ trong lệnh để không phụ thuộc default code):** `batch_size=128`, `grad_from_block=11`, `sup_weight=0.35`, `transform=imagenet`, `eval=v2`, `warmup_teacher_temp=0.07/0.04/30`, `use_ssb_splits`; `memax_weight`: CUB=`2`, Cars/Aircraft=`1` (theo `scripts/run_*.sh`); pseudo-known Mode-1 (warmup 30, freq 10, iters 20, top-ratio 0.8, cap 500); novel warmup 50 / freq 10 / **max-iters 6 / cap 200** (khác default code là 2×100 nên phải gõ rõ); parts `num_slots=3, part_lambda=0.5, tau_c=0.1`; teacher `m0=0.996`. Exp-name Modal tự sinh `simgcd-<data>-<variant>-dinov2b14-seed0-<stamp>-<suffix>`.

### A. 12 lệnh Modal (detached với `-d`)

```bash
# Lần đầu chạy aircraft trên Modal: nạp data 1 lần (idempotent, scars đã có sẵn)
modal run modal_train_simgcd.py::fetch_aircraft

# --- CUB-200 (100 known / 100 novel) ---
# A1. Baseline
modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --seed 0 --memax-weight 2 --epochs 200 --exp-name-suffix Rerun_CUB_SimBase
# B1. Dual-only (known Mode-1 + novel)
modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --seed 0 --memax-weight 2 --epochs 200 --exp-name-suffix Rerun_CUB_SimDual --enable-pseudo-labeling --pseudo-mode 1 --confidence-threshold 0.9 --pseudo-top-ratio 0.8 --max-samples-per-class 500 --pseudo-update-freq 10 --max-pseudo-iterations 20 --pseudo-warmup-epoch 30 --pseudo-conf-bar 0.5 --pseudo-bar-k 2.0 --pseudo-min-hi 10 --enable-novel-pseudo --novel-warmup-epoch 50 --novel-update-freq 10 --max-novel-iterations 6 --novel-max-samples 200 --novel-jaccard-th 0.6 --novel-agree-th 0.7 --novel-min-size 10
# C1. Multi-prototype (3 slots) + teacher-only, không pseudo
modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --seed 0 --memax-weight 2 --epochs 200 --exp-name-suffix Rerun_CUB_SimProtoT --use-parts --num-slots 3 --part-lambda 0.5 --tau-c 0.1 --use-momentum-teacher --teacher-m0 0.996
# D1. Full: dual + multi-prototype + teacher
modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --seed 0 --memax-weight 2 --epochs 200 --exp-name-suffix Rerun_CUB_SimFull --enable-pseudo-labeling --pseudo-mode 1 --confidence-threshold 0.9 --pseudo-top-ratio 0.8 --max-samples-per-class 500 --pseudo-update-freq 10 --max-pseudo-iterations 20 --pseudo-warmup-epoch 30 --pseudo-conf-bar 0.5 --pseudo-bar-k 2.0 --pseudo-min-hi 10 --enable-novel-pseudo --novel-warmup-epoch 50 --novel-update-freq 10 --max-novel-iterations 6 --novel-max-samples 200 --novel-jaccard-th 0.6 --novel-agree-th 0.7 --novel-min-size 10 --use-parts --num-slots 3 --part-lambda 0.5 --tau-c 0.1 --use-momentum-teacher --teacher-m0 0.996

# --- Stanford Cars = scars (98 known / 98 novel) ---
# A2. Baseline
modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --seed 0 --memax-weight 1 --epochs 200 --exp-name-suffix Rerun_CARS_SimBase
# B2. Dual-only (known Mode-1 + novel)
modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --seed 0 --memax-weight 1 --epochs 200 --exp-name-suffix Rerun_CARS_SimDual --enable-pseudo-labeling --pseudo-mode 1 --confidence-threshold 0.9 --pseudo-top-ratio 0.8 --max-samples-per-class 500 --pseudo-update-freq 10 --max-pseudo-iterations 20 --pseudo-warmup-epoch 30 --pseudo-conf-bar 0.5 --pseudo-bar-k 2.0 --pseudo-min-hi 10 --enable-novel-pseudo --novel-warmup-epoch 50 --novel-update-freq 10 --max-novel-iterations 6 --novel-max-samples 200 --novel-jaccard-th 0.6 --novel-agree-th 0.7 --novel-min-size 10
# C2. Multi-prototype (3 slots) + teacher-only, không pseudo
modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --seed 0 --memax-weight 1 --epochs 200 --exp-name-suffix Rerun_CARS_SimProtoT --use-parts --num-slots 3 --part-lambda 0.5 --tau-c 0.1 --use-momentum-teacher --teacher-m0 0.996
# D2. Full: dual + multi-prototype + teacher
modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --seed 0 --memax-weight 1 --epochs 200 --exp-name-suffix Rerun_CARS_SimFull --enable-pseudo-labeling --pseudo-mode 1 --confidence-threshold 0.9 --pseudo-top-ratio 0.8 --max-samples-per-class 500 --pseudo-update-freq 10 --max-pseudo-iterations 20 --pseudo-warmup-epoch 30 --pseudo-conf-bar 0.5 --pseudo-bar-k 2.0 --pseudo-min-hi 10 --enable-novel-pseudo --novel-warmup-epoch 50 --novel-update-freq 10 --max-novel-iterations 6 --novel-max-samples 200 --novel-jaccard-th 0.6 --novel-agree-th 0.7 --novel-min-size 10 --use-parts --num-slots 3 --part-lambda 0.5 --tau-c 0.1 --use-momentum-teacher --teacher-m0 0.996

# --- FGVC-Aircraft (50 known / 50 novel theo SSB split SimGCD) ---
# A3. Baseline
modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --seed 0 --memax-weight 1 --epochs 200 --exp-name-suffix Rerun_AIR_SimBase
# B3. Dual-only (known Mode-1 + novel)
modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --seed 0 --memax-weight 1 --epochs 200 --exp-name-suffix Rerun_AIR_SimDual --enable-pseudo-labeling --pseudo-mode 1 --confidence-threshold 0.9 --pseudo-top-ratio 0.8 --max-samples-per-class 500 --pseudo-update-freq 10 --max-pseudo-iterations 20 --pseudo-warmup-epoch 30 --pseudo-conf-bar 0.5 --pseudo-bar-k 2.0 --pseudo-min-hi 10 --enable-novel-pseudo --novel-warmup-epoch 50 --novel-update-freq 10 --max-novel-iterations 6 --novel-max-samples 200 --novel-jaccard-th 0.6 --novel-agree-th 0.7 --novel-min-size 10
# C3. Multi-prototype (3 slots) + teacher-only, không pseudo
modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --seed 0 --memax-weight 1 --epochs 200 --exp-name-suffix Rerun_AIR_SimProtoT --use-parts --num-slots 3 --part-lambda 0.5 --tau-c 0.1 --use-momentum-teacher --teacher-m0 0.996
# D3. Full: dual + multi-prototype + teacher
modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --seed 0 --memax-weight 1 --epochs 200 --exp-name-suffix Rerun_AIR_SimFull --enable-pseudo-labeling --pseudo-mode 1 --confidence-threshold 0.9 --pseudo-top-ratio 0.8 --max-samples-per-class 500 --pseudo-update-freq 10 --max-pseudo-iterations 20 --pseudo-warmup-epoch 30 --pseudo-conf-bar 0.5 --pseudo-bar-k 2.0 --pseudo-min-hi 10 --enable-novel-pseudo --novel-warmup-epoch 50 --novel-update-freq 10 --max-novel-iterations 6 --novel-max-samples 200 --novel-jaccard-th 0.6 --novel-agree-th 0.7 --novel-min-size 10 --use-parts --num-slots 3 --part-lambda 0.5 --tau-c 0.1 --use-momentum-teacher --teacher-m0 0.996
```

Theo dõi log: `modal app logs simgcd-train` (launcher có preflight check flag miễn phí trước khi submit).

### B. 12 lệnh server thường (flag gạch dưới `_`)

```bash
# Sửa 3 đường data cho đúng server trước khi chạy (CUB = PARENT của CUB_200_2011;
# scars = thư mục chứa devkit/cars_train...; aircraft = thư mục fgvc-aircraft-2013b/ chứa data/images):
export SIMGCD_CUB_ROOT=/data/cub
export SIMGCD_SCARS_ROOT=/data/cars
export SIMGCD_AIRCRAFT_ROOT=/data/aircraft/fgvc-aircraft-2013b

# --- CUB-200 ---
# A1. Baseline
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name cub --backbone dinov2_vitb14 --grad_from_block 11 --epochs 200 --num_workers 8 --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 --transform imagenet --lr 0.1 --eval_funcs v2 --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --memax_weight 2 --seed 0 --exp_name Rerun_CUB_SimBase
# B1. Dual-only (known Mode-1 + novel)
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name cub --backbone dinov2_vitb14 --grad_from_block 11 --epochs 200 --num_workers 8 --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 --transform imagenet --lr 0.1 --eval_funcs v2 --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --memax_weight 2 --seed 0 --exp_name Rerun_CUB_SimDual --enable_pseudo_labeling --pseudo_mode 1 --confidence_threshold 0.9 --pseudo_top_ratio 0.8 --max_samples_per_class 500 --pseudo_update_freq 10 --max_pseudo_iterations 20 --pseudo_warmup_epoch 30 --pseudo_conf_bar 0.5 --pseudo_bar_k 2.0 --pseudo_min_hi 10 --enable_novel_pseudo --novel_warmup_epoch 50 --novel_update_freq 10 --max_novel_iterations 6 --novel_max_samples 200 --novel_jaccard_th 0.6 --novel_agree_th 0.7 --novel_min_size 10
# C1. Multi-prototype (3 slots) + teacher-only, không pseudo
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name cub --backbone dinov2_vitb14 --grad_from_block 11 --epochs 200 --num_workers 8 --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 --transform imagenet --lr 0.1 --eval_funcs v2 --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --memax_weight 2 --seed 0 --exp_name Rerun_CUB_SimProtoT --use_parts --num_slots 3 --part_lambda 0.5 --tau_c 0.1 --use_momentum_teacher --teacher_m0 0.996
# D1. Full: dual + multi-prototype + teacher
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name cub --backbone dinov2_vitb14 --grad_from_block 11 --epochs 200 --num_workers 8 --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 --transform imagenet --lr 0.1 --eval_funcs v2 --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --memax_weight 2 --seed 0 --exp_name Rerun_CUB_SimFull --enable_pseudo_labeling --pseudo_mode 1 --confidence_threshold 0.9 --pseudo_top_ratio 0.8 --max_samples_per_class 500 --pseudo_update_freq 10 --max_pseudo_iterations 20 --pseudo_warmup_epoch 30 --pseudo_conf_bar 0.5 --pseudo_bar_k 2.0 --pseudo_min_hi 10 --enable_novel_pseudo --novel_warmup_epoch 50 --novel_update_freq 10 --max_novel_iterations 6 --novel_max_samples 200 --novel_jaccard_th 0.6 --novel_agree_th 0.7 --novel_min_size 10 --use_parts --num_slots 3 --part_lambda 0.5 --tau_c 0.1 --use_momentum_teacher --teacher_m0 0.996

# --- Stanford Cars (dataset_name=scars) ---
# A2. Baseline
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name scars --backbone dinov2_vitb14 --grad_from_block 11 --epochs 200 --num_workers 8 --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 --transform imagenet --lr 0.1 --eval_funcs v2 --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --memax_weight 1 --seed 0 --exp_name Rerun_CARS_SimBase
# B2. Dual-only (known Mode-1 + novel)
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name scars --backbone dinov2_vitb14 --grad_from_block 11 --epochs 200 --num_workers 8 --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 --transform imagenet --lr 0.1 --eval_funcs v2 --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --memax_weight 1 --seed 0 --exp_name Rerun_CARS_SimDual --enable_pseudo_labeling --pseudo_mode 1 --confidence_threshold 0.9 --pseudo_top_ratio 0.8 --max_samples_per_class 500 --pseudo_update_freq 10 --max_pseudo_iterations 20 --pseudo_warmup_epoch 30 --pseudo_conf_bar 0.5 --pseudo_bar_k 2.0 --pseudo_min_hi 10 --enable_novel_pseudo --novel_warmup_epoch 50 --novel_update_freq 10 --max_novel_iterations 6 --novel_max_samples 200 --novel_jaccard_th 0.6 --novel_agree_th 0.7 --novel_min_size 10
# C2. Multi-prototype (3 slots) + teacher-only, không pseudo
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name scars --backbone dinov2_vitb14 --grad_from_block 11 --epochs 200 --num_workers 8 --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 --transform imagenet --lr 0.1 --eval_funcs v2 --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --memax_weight 1 --seed 0 --exp_name Rerun_CARS_SimProtoT --use_parts --num_slots 3 --part_lambda 0.5 --tau_c 0.1 --use_momentum_teacher --teacher_m0 0.996
# D2. Full: dual + multi-prototype + teacher
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name scars --backbone dinov2_vitb14 --grad_from_block 11 --epochs 200 --num_workers 8 --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 --transform imagenet --lr 0.1 --eval_funcs v2 --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --memax_weight 1 --seed 0 --exp_name Rerun_CARS_SimFull --enable_pseudo_labeling --pseudo_mode 1 --confidence_threshold 0.9 --pseudo_top_ratio 0.8 --max_samples_per_class 500 --pseudo_update_freq 10 --max_pseudo_iterations 20 --pseudo_warmup_epoch 30 --pseudo_conf_bar 0.5 --pseudo_bar_k 2.0 --pseudo_min_hi 10 --enable_novel_pseudo --novel_warmup_epoch 50 --novel_update_freq 10 --max_novel_iterations 6 --novel_max_samples 200 --novel_jaccard_th 0.6 --novel_agree_th 0.7 --novel_min_size 10 --use_parts --num_slots 3 --part_lambda 0.5 --tau_c 0.1 --use_momentum_teacher --teacher_m0 0.996

# --- FGVC-Aircraft (dataset_name=aircraft) ---
# A3. Baseline
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name aircraft --backbone dinov2_vitb14 --grad_from_block 11 --epochs 200 --num_workers 8 --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 --transform imagenet --lr 0.1 --eval_funcs v2 --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --memax_weight 1 --seed 0 --exp_name Rerun_AIR_SimBase
# B3. Dual-only (known Mode-1 + novel)
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name aircraft --backbone dinov2_vitb14 --grad_from_block 11 --epochs 200 --num_workers 8 --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 --transform imagenet --lr 0.1 --eval_funcs v2 --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --memax_weight 1 --seed 0 --exp_name Rerun_AIR_SimDual --enable_pseudo_labeling --pseudo_mode 1 --confidence_threshold 0.9 --pseudo_top_ratio 0.8 --max_samples_per_class 500 --pseudo_update_freq 10 --max_pseudo_iterations 20 --pseudo_warmup_epoch 30 --pseudo_conf_bar 0.5 --pseudo_bar_k 2.0 --pseudo_min_hi 10 --enable_novel_pseudo --novel_warmup_epoch 50 --novel_update_freq 10 --max_novel_iterations 6 --novel_max_samples 200 --novel_jaccard_th 0.6 --novel_agree_th 0.7 --novel_min_size 10
# C3. Multi-prototype (3 slots) + teacher-only, không pseudo
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name aircraft --backbone dinov2_vitb14 --grad_from_block 11 --epochs 200 --num_workers 8 --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 --transform imagenet --lr 0.1 --eval_funcs v2 --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --memax_weight 1 --seed 0 --exp_name Rerun_AIR_SimProtoT --use_parts --num_slots 3 --part_lambda 0.5 --tau_c 0.1 --use_momentum_teacher --teacher_m0 0.996
# D3. Full: dual + multi-prototype + teacher
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name aircraft --backbone dinov2_vitb14 --grad_from_block 11 --epochs 200 --num_workers 8 --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 --transform imagenet --lr 0.1 --eval_funcs v2 --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --memax_weight 1 --seed 0 --exp_name Rerun_AIR_SimFull --enable_pseudo_labeling --pseudo_mode 1 --confidence_threshold 0.9 --pseudo_top_ratio 0.8 --max_samples_per_class 500 --pseudo_update_freq 10 --max_pseudo_iterations 20 --pseudo_warmup_epoch 30 --pseudo_conf_bar 0.5 --pseudo_bar_k 2.0 --pseudo_min_hi 10 --enable_novel_pseudo --novel_warmup_epoch 50 --novel_update_freq 10 --max_novel_iterations 6 --novel_max_samples 200 --novel_jaccard_th 0.6 --novel_agree_th 0.7 --novel_min_size 10 --use_parts --num_slots 3 --part_lambda 0.5 --tau_c 0.1 --use_momentum_teacher --teacher_m0 0.996
```

Lưu ý server Windows (PowerShell): thay `export X=...` bằng `$env:X="..."` và bỏ prefix `CUDA_VISIBLE_DEVICES=0`.
