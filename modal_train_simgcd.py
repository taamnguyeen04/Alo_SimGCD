"""Modal launcher for SimGCD (generality track, plansimgcd.md).

NOTE: this is a copy of gcd-hosts/modal_train_simgcd.py placed INSIDE SimGCD/
so the folder is self-contained (run everything from here). It mounts the
current dir (SimGCD/) to keep the image lean.

Reuses the bacon-storage volume (CUB images already there from the BaCon
track) and mirrors BaCon/modal_train.py patterns, kept lean: one train()
plus a foreground local entrypoint.

Data mapping:
  SIMGCD_CUB_ROOT=/bacon-storage/data/cub   (PARENT of CUB_200_2011;
  simgcd joins root/'CUB_200_2011'/... itself — see simgcd/config.py override)
SSB splits ship inside the repo mount (simgcd/data/ssb_splits, tiny).

Usage (from SimGCD/, Modal converts _ to - in flag names).
Experiment names are auto-readable: simgcd-cub-m1-dinov2b14-seed0-<stamp>[-suffix].
  # S0v1: DINOv1 baseline (anchor to reported paper numbers)
  modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dino-vitb16 --exp-name-suffix S0v1
  # S0: DINOv2 baseline (floor for the generality track)
  modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --exp-name-suffix S0
  # S1: + Mode-1 known pseudo (explicit flags, no extra-args needed)
  modal run -d modal_train_simgcd.py::launch --dataset-name cub --enable-pseudo-labeling --pseudo-mode 1 --exp-name-suffix S1
  # S2: S1 + novel door
  modal run -d modal_train_simgcd.py::launch --dataset-name cub --enable-pseudo-labeling --pseudo-mode 1 --enable-novel-pseudo --exp-name-suffix S2
  # S3: + AdaPart fused-logits (global + part)
  modal run -d modal_train_simgcd.py::launch --dataset-name cub --use-parts --num-slots 3 --exp-name-suffix S3
  # S1+teacher: S1 + momentum teacher (BaCon A1 graft: stable distill target)
  modal run -d modal_train_simgcd.py::launch --dataset-name cub --enable-pseudo-labeling --pseudo-mode 1 --use-momentum-teacher --exp-name-suffix S1teacher

Fine-grained generality batch (DA-GCD, balanced-SSB, DINOv2-B/14, seed0 probes):
  Step 0 (once): fetch aircraft raw data onto the volume (scars already there)
  modal run modal_train_simgcd.py::fetch_aircraft
  1. scars base:    modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --seed 0 --exp-name-suffix S0
  2. scars dual:    modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --enable-pseudo-labeling --pseudo-mode 1 --max-pseudo-iterations 4 --enable-novel-pseudo --max-novel-iterations 6 --novel-max-samples 200 --seed 0 --exp-name-suffix S2
  3. aircraft base: modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --seed 0 --exp-name-suffix S0
  4. aircraft dual: modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --enable-pseudo-labeling --pseudo-mode 1 --max-pseudo-iterations 4 --enable-novel-pseudo --max-novel-iterations 6 --novel-max-samples 200 --seed 0 --exp-name-suffix S2
  5. cub parts rerun (old S3 log truncated, missing Best lines):
  modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --use-parts --num-slots 3 --seed 0 --exp-name-suffix S3rerun
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path

import modal

APP_NAME = "simgcd-train"
# NOTE (self-contained copy): file lives INSIDE SimGCD/, so the repo IS the
# parent dir (the gcd-hosts/ original appends / "simgcd" instead).
LOCAL_REPO = Path(__file__).resolve().parent
PROJECT_DIR = "/root/simgcd"
STORAGE_DIR = "/bacon-storage"
DATA_DIR = f"{STORAGE_DIR}/data"
OUTPUT_DIR = f"{STORAGE_DIR}/outputs"
TORCH_CACHE_DIR = f"{STORAGE_DIR}/torch-cache"
SIMGCD_OUT = f"{OUTPUT_DIR}/simgcd"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "loguru",
        "numpy<2",
        "pandas",
        "scikit-learn",
        "scipy",
        "tqdm",
        "pillow",
        "matplotlib",
        "torch",
        "torchvision",
    )
    .add_local_dir(
        str(LOCAL_REPO),
        remote_path=PROJECT_DIR,
        ignore=["dev_outputs", "__pycache__", "venv", ".venv", ".git", "env"],
    )
)

storage_volume = modal.Volume.from_name("bacon-storage", create_if_missing=True)


def _run(cmd: list[str], cwd: str = PROJECT_DIR) -> None:
    log_path = Path(OUTPUT_DIR) / "last_simgcd_train.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # PARENT of CUB_200_2011 (see simgcd/config.py override + module docstring)
    env["SIMGCD_CUB_ROOT"] = f"{DATA_DIR}/cub"
    # NOTE (generality track, scars/aircraft): same override pattern; without
    # these, config.py's literal '${DATASET_DIR}/...' placeholders reach the
    # loaders unexpanded. Aircraft root points one level deeper because the
    # tarball extracts to fgvc-aircraft-2013b/data/images (see fetch_aircraft).
    env["SIMGCD_SCARS_ROOT"] = f"{DATA_DIR}/cars"
    env["SIMGCD_AIRCRAFT_ROOT"] = f"{DATA_DIR}/aircraft/fgvc-aircraft-2013b"
    print("$", " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd, cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        tail = []
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
            tail.append(line)
            if len(tail) > 200:
                tail.pop(0)
        return_code = process.wait()
    if return_code != 0:
        print("\nLast 200 training log lines:\n", flush=True)
        print("".join(tail), flush=True)
        raise subprocess.CalledProcessError(return_code, cmd)


def _known_train_flags() -> set[str]:
    """Parse simgcd/train.py add_argument flags from the LOCAL copy.

    Catches flag-spelling mismatches (dashes vs underscores) BEFORE spending
    a Modal run — argparse on the remote would fail in seconds anyway, but
    local validation is instant and free.
    """
    import re
    text = (LOCAL_REPO / "train.py").read_text(encoding="utf-8")
    return set(re.findall(r"add_argument\('([^']+)'", text))


def _validate_flags(command: list[str]) -> None:
    known = _known_train_flags()
    sent = [c for c in command if c.startswith("--")]
    bad = [c for c in sent if c not in known]
    if bad:
        raise ValueError(
            f"Unknown train.py flags (would fail remotely): {bad}\n"
            f"Known flags: {sorted(known)}")


_BACKBONE_TAGS = {"dino_vitb16": "dinov1b16", "dinov2_vitb14": "dinov2b14",
                 "dinov2_vitb14_reg": "dinov2b14reg"}

# Model tag đầu tên experiment: {model}-{data}-{config} (vd simgcd-cub-m1+teacher-dinov2b14-seed0-<stamp>).
MODEL_TAG = "simgcd"


def _variant_tag(enable_pseudo_labeling=False, pseudo_mode=1, enable_novel_pseudo=False,
                 use_parts=False, num_slots=3, use_teacher=False, imb_ratio=None):
    """Short readable variant: base / m1 / m1-novel / ... + +partsS3 when on.

    Slots use S (S2/S3) so they never collide with mode labels (M1/M2/M3).
    +teacher appended last when the momentum teacher graft is on.
    -imbN appended last when a precomputed BaCon split is used
    (1 = balanced, 10 = long-tailed); absent = legacy SSB/uniform path.
    """
    if not enable_pseudo_labeling:
        v = "base"
    else:
        v = {0: "novel-only", 1: "m1", 2: "m2", 3: "m3"}.get(int(pseudo_mode), f"m{pseudo_mode}")
        if v != "novel-only" and enable_novel_pseudo:
            v = f"{v}-novel"
    if use_parts:
        v = f"{v}+partsS{int(num_slots)}"
    if use_teacher:
        v = f"{v}+teacher"
    if imb_ratio is not None and int(imb_ratio) > 0:
        v = f"{v}-imb{int(imb_ratio)}"
    return v


def _pseudo_train_flags(enable_pseudo_labeling=False, pseudo_mode=1,
                        confidence_threshold=0.9, pseudo_top_ratio=0.8,
                        max_samples_per_class=500, pseudo_update_freq=10,
                        max_pseudo_iterations=20, pseudo_warmup_epoch=30,
                        pseudo_conf_bar=0.5, pseudo_bar_k=2.0, pseudo_min_hi=10,
                        enable_novel_pseudo=False, novel_warmup_epoch=50,
                        novel_update_freq=10, max_novel_iterations=2,
                        novel_max_samples=100, novel_jaccard_th=0.6,
                        novel_agree_th=0.7, novel_min_size=10):
    """Build simgcd/train.py pseudo flags (UNDERSCORES, exactly as defined)."""
    if not enable_pseudo_labeling:
        return []
    cmd = ["--enable_pseudo_labeling",
           "--pseudo_mode", str(pseudo_mode),
           "--confidence_threshold", str(confidence_threshold),
           "--pseudo_top_ratio", str(pseudo_top_ratio),
           "--max_samples_per_class", str(max_samples_per_class),
           "--pseudo_update_freq", str(pseudo_update_freq),
           "--max_pseudo_iterations", str(max_pseudo_iterations),
           "--pseudo_warmup_epoch", str(pseudo_warmup_epoch),
           "--pseudo_conf_bar", str(pseudo_conf_bar),
           "--pseudo_bar_k", str(pseudo_bar_k),
           "--pseudo_min_hi", str(pseudo_min_hi)]
    if enable_novel_pseudo:
        cmd += ["--enable_novel_pseudo",
                "--novel_warmup_epoch", str(novel_warmup_epoch),
                "--novel_update_freq", str(novel_update_freq),
                "--max_novel_iterations", str(max_novel_iterations),
                "--novel_max_samples", str(novel_max_samples),
                "--novel_jaccard_th", str(novel_jaccard_th),
                "--novel_agree_th", str(novel_agree_th),
                "--novel_min_size", str(novel_min_size)]
    return cmd


def _preflight_check(**kwargs) -> None:
    """Rebuild the remote command locally (minus exp_name/exp_root values)
    and validate flags + backbone value before submitting to Modal."""
    backbone = (kwargs.get("backbone") or "").replace("-", "_")
    if backbone not in ("dino_vitb16", "dinov2_vitb14", "dinov2_vitb14_reg"):
        raise ValueError(f"Unknown backbone value: {kwargs.get('backbone')}")
    command = [
        "python", "train.py",
        "--dataset_name", str(kwargs.get("dataset_name")),
        "--batch_size", str(kwargs.get("batch_size")),
        "--grad_from_block", str(kwargs.get("grad_from_block")),
        "--epochs", str(kwargs.get("epochs")),
        "--num_workers", str(kwargs.get("num_workers")),
        "--use_ssb_splits",
        "--sup_weight", str(kwargs.get("sup_weight")),
        "--weight_decay", "5e-5",
        "--transform", "imagenet",
        "--lr", str(kwargs.get("lr")),
        "--eval_funcs", "v2",
        "--warmup_teacher_temp", "0.07",
        "--teacher_temp", "0.04",
        "--warmup_teacher_temp_epochs",
        str(min(30, int(kwargs.get("epochs") or 0))),
        "--memax_weight", str(kwargs.get("memax_weight")),
        "--backbone", backbone,
        "--exp_name", "PREFLIGHT",
        "--exp_root", "/tmp/preflight",
    ]
    for c in (kwargs.get("extra_args") or []):
        if isinstance(c, str) and c.startswith("--"):
            command.append(c)
    command += _pseudo_train_flags(
        enable_pseudo_labeling=kwargs.get("enable_pseudo_labeling", False),
        pseudo_mode=kwargs.get("pseudo_mode", 1),
        confidence_threshold=kwargs.get("confidence_threshold", 0.9),
        pseudo_top_ratio=kwargs.get("pseudo_top_ratio", 0.8),
        max_samples_per_class=kwargs.get("max_samples_per_class", 500),
        pseudo_update_freq=kwargs.get("pseudo_update_freq", 10),
        max_pseudo_iterations=kwargs.get("max_pseudo_iterations", 20),
        pseudo_warmup_epoch=kwargs.get("pseudo_warmup_epoch", 30),
        pseudo_conf_bar=kwargs.get("pseudo_conf_bar", 0.5),
        pseudo_bar_k=kwargs.get("pseudo_bar_k", 2.0),
        pseudo_min_hi=kwargs.get("pseudo_min_hi", 10),
        enable_novel_pseudo=kwargs.get("enable_novel_pseudo", False),
        novel_warmup_epoch=kwargs.get("novel_warmup_epoch", 50),
        novel_update_freq=kwargs.get("novel_update_freq", 10),
        max_novel_iterations=kwargs.get("max_novel_iterations", 2),
        novel_max_samples=kwargs.get("novel_max_samples", 100),
        novel_jaccard_th=kwargs.get("novel_jaccard_th", 0.6),
        novel_agree_th=kwargs.get("novel_agree_th", 0.7),
        novel_min_size=kwargs.get("novel_min_size", 10))
    if kwargs.get("use_parts", False):
        command += ["--use_parts",
                    "--num_slots", str(kwargs.get("num_slots", 3)),
                    "--part_lambda", str(kwargs.get("part_lambda", 0.5)),
                    "--tau_c", str(kwargs.get("tau_c", 0.1))]
        if kwargs.get("ablate_confidence", False):
            command.append("--ablate_confidence")
    if kwargs.get("use_momentum_teacher", False):
        command += ["--use_momentum_teacher",
                    "--teacher_m0", str(kwargs.get("teacher_m0", 0.996))]
        if kwargs.get("teacher_fused", False):
            command.append("--teacher_fused")
        if int(kwargs.get("teacher_warmup_epoch", 0) or 0) > 0:
            command += ["--teacher_warmup_epoch",
                        str(int(kwargs.get("teacher_warmup_epoch")))]
    if kwargs.get("imb_ratio", None) is not None:
        command += ["--imb_ratio", str(int(kwargs.get("imb_ratio")))]
    if kwargs.get("early_stop_patience", 0) is not None \
            and int(kwargs.get("early_stop_patience", 0)) > 0:
        command += ["--early_stop_patience",
                    str(int(kwargs.get("early_stop_patience")))]
    if kwargs.get("fp16", False):
        command.append("--fp16")
    command += ["--gate_init", str(float(kwargs.get("gate_init", -1.0)))]
    if kwargs.get("resume", None):
        command += ["--resume", str(kwargs.get("resume"))]
    _validate_flags(command)
    print("[preflight] flags OK.", flush=True)


def _ensure_link(path: Path, target: Path) -> None:
    if path.exists() or path.is_symlink():
        return
    target.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target, target_is_directory=True)


@app.function(
    image=image,
    timeout=60 * 60 * 6,
    volumes={STORAGE_DIR: storage_volume},
)
def fetch_aircraft() -> dict:
    """One-time CPU fetch of FGVC-Aircraft 2013b onto the volume.

    Run ONCE before any --dataset-name aircraft job:
        modal run modal_train_simgcd.py::fetch_aircraft
    Idempotent (skips if data/images already present). Uses the same upstream
    URL as simgcd/data/fgvc_aircraft.py. If the host is unreachable, aircraft
    runs fall under the 3-day kill rule — report back instead of retrying forever.
    """
    import tarfile
    import urllib.request

    dest = Path(f"{DATA_DIR}/aircraft")
    dest.mkdir(parents=True, exist_ok=True)
    root = dest / "fgvc-aircraft-2013b"
    if (root / "data" / "images").is_dir():
        return {"status": "exists", "root": str(root)}
    url = ("http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/"
           "fgvc-aircraft-2013b.tar.gz")
    tar_path = dest / "fgvc-aircraft-2013b.tar.gz"
    print(f"Downloading {url} ...", flush=True)
    urllib.request.urlretrieve(url, tar_path)
    print("Extracting ...", flush=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest)
    tar_path.unlink()
    ok = (root / "data" / "images").is_dir() and \
        (root / "data" / "images_variant_trainval.txt").is_file()
    storage_volume.commit()
    if not ok:
        raise RuntimeError(f"fetch_aircraft: unexpected layout under {root}")
    return {"status": "fetched", "root": str(root)}


@app.function(
    image=image,
    # GPU selectable via env (default A100-80GB for deadline full runs):
    # $env:SIMGCD_GPU="A10G" (PowerShell) for cheap tests.
    gpu=os.environ.get("SIMGCD_GPU", "A100-80GB"),
    # 8 CPUs so the 8 DataLoader workers (+ KMeans novel-door on CPU) don't
    # starve a fast GPU. CPU is cheap next to GPU hours; drop to 4 if Modal
    # quotas complain.
    cpu=8,
    timeout=60 * 60 * 24,
    volumes={STORAGE_DIR: storage_volume},
)
def train(
    dataset_name: str = "cub",
    backbone: str = "dinov2_vitb14",
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 0.1,
    sup_weight: float = 0.35,
    memax_weight: float = 2,
    grad_from_block: int = 11,
    num_workers: int = 8,
    extra_args: list[str] | None = None,
    exp_name_suffix: str = "",
    # Train seed: -1 = legacy unseeded (matches S0/S1/S3 so far); >=0 explicit
    # repeat seed (passed as --seed; also stamped into exp name as seed{N},
    # same convention as the HypCD/DebGCD launchers).
    seed: int = -1,
    # Imbalance: None = legacy SSB/uniform path (default, keeps all old runs
    # identical); 1 = balanced precomputed split (cub200_k100_imb1);
    # 10 = long-tailed split (cub200_k100_imb10). CUB-only for now.
    imb_ratio: int | None = None,
    # Pseudo labeling (porter BaCon; launcher params use _ which Modal CLI
    # shows as -; train.py flags use _ and are built by _pseudo_train_flags)
    enable_pseudo_labeling: bool = False,
    pseudo_mode: int = 1,
    confidence_threshold: float = 0.9,
    pseudo_top_ratio: float = 0.8,
    max_samples_per_class: int = 500,
    pseudo_update_freq: int = 10,
    max_pseudo_iterations: int = 20,
    pseudo_warmup_epoch: int = 30,
    pseudo_conf_bar: float = 0.5,
    pseudo_bar_k: float = 2.0,
    pseudo_min_hi: int = 10,
    enable_novel_pseudo: bool = False,
    novel_warmup_epoch: int = 50,
    novel_update_freq: int = 10,
    max_novel_iterations: int = 2,
    novel_max_samples: int = 100,
    novel_jaccard_th: float = 0.6,
    novel_agree_th: float = 0.7,
    novel_min_size: int = 10,
    # AdaPart fused-logits (porter BaCon)
    use_parts: bool = False,
    num_slots: int = 3,
    part_lambda: float = 0.5,
    tau_c: float = 0.1,
    ablate_confidence: bool = False,
    # Momentum teacher (porter BaCon A1)
    use_momentum_teacher: bool = False,
    teacher_m0: float = 0.996,
    teacher_fused: bool = False,
    teacher_warmup_epoch: int = 0,
    # Early stopping: 0 = off (legacy run-all-epochs); >0 = epochs without
    # disjoint-test All improvement before stopping (best kept in best_test.pt).
    early_stop_patience: int = 0,
    # Mixed precision: big free speedup (~1.3-1.6x) on A100/H100, halves
    # activation memory (fits bigger batches). Validate vs fp32 once.
    fp16: bool = False,
    # Gate init constant: -1.0 = closed-start (legacy); 0.0 = open-start.
    gate_init: float = -1.0,
    # Resume from a volume checkpoint (preemption-safe continuation).
    # Pass the FULL /bacon-storage path to model.pt (not best.pt).
    resume: str | None = None,
) -> dict:
    """Train SimGCD (mirrors simgcd/scripts/run_cub.sh defaults unless overridden)."""
    backbone = (backbone or "").replace("-", "_")
    project_dir = Path(PROJECT_DIR)

    os.chdir(project_dir)
    _ensure_link(Path("/root/.cache/torch"), Path(TORCH_CACHE_DIR))

    # Readable names: {model}-{data}-{config}, e.g. simgcd-cub-m1-dinov2b14-seed0-<stamp>-S1,
    # simgcd-cub-base+partsS3-dinov2b14-seed0-<stamp>-S3rerun (suffix like S1/S2 still appended as-is).
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = _BACKBONE_TAGS.get(backbone, backbone)
    variant = _variant_tag(enable_pseudo_labeling, pseudo_mode, enable_novel_pseudo,
                           use_parts, num_slots, use_momentum_teacher, imb_ratio)
    seed_tag = f"-seed{int(seed)}" if seed is not None and int(seed) >= 0 else ""
    exp_name = f"{MODEL_TAG}-{dataset_name}-{variant}-{tag}{seed_tag}-{stamp}"
    if exp_name_suffix:
        exp_name = f"{exp_name}-{exp_name_suffix}"
    exp_root = Path(SIMGCD_OUT) / exp_name
    exp_root.mkdir(parents=True, exist_ok=True)
    print(f"Experiment: {exp_name}\nRoot: {exp_root}", flush=True)

    # NOTE: simgcd/train.py uses UNDERSCORES for almost everything — match each
    # flag EXACTLY as in add_argument (verified by grep; the usage readout is
    # easy to misread). Dashes only: epochs/transform/lr/backbone.
    command = [
        "python", "train.py",
        "--dataset_name", dataset_name,
        "--batch_size", str(batch_size),
        "--grad_from_block", str(grad_from_block),
        "--epochs", str(epochs),
        "--num_workers", str(num_workers),
        "--use_ssb_splits",
        "--sup_weight", str(sup_weight),
        "--weight_decay", "5e-5",
        "--transform", "imagenet",
        "--lr", str(lr),
        "--eval_funcs", "v2",
        "--warmup_teacher_temp", "0.07",
        "--teacher_temp", "0.04",
        "--warmup_teacher_temp_epochs", str(min(30, int(epochs))),
        "--memax_weight", str(memax_weight),
        "--backbone", backbone,
        "--exp_name", exp_name,
        "--exp_root", str(exp_root),
    ]
    command += _pseudo_train_flags(
        enable_pseudo_labeling=enable_pseudo_labeling, pseudo_mode=pseudo_mode,
        confidence_threshold=confidence_threshold, pseudo_top_ratio=pseudo_top_ratio,
        max_samples_per_class=max_samples_per_class, pseudo_update_freq=pseudo_update_freq,
        max_pseudo_iterations=max_pseudo_iterations, pseudo_warmup_epoch=pseudo_warmup_epoch,
        pseudo_conf_bar=pseudo_conf_bar, pseudo_bar_k=pseudo_bar_k,
        pseudo_min_hi=pseudo_min_hi, enable_novel_pseudo=enable_novel_pseudo,
        novel_warmup_epoch=novel_warmup_epoch, novel_update_freq=novel_update_freq,
        max_novel_iterations=max_novel_iterations, novel_max_samples=novel_max_samples,
        novel_jaccard_th=novel_jaccard_th, novel_agree_th=novel_agree_th,
        novel_min_size=novel_min_size)
    if use_parts:
        command += ["--use_parts",
                    "--num_slots", str(num_slots),
                    "--part_lambda", str(part_lambda),
                    "--tau_c", str(tau_c)]
        if ablate_confidence:
            command.append("--ablate_confidence")
    if use_momentum_teacher:
        command += ["--use_momentum_teacher",
                    "--teacher_m0", str(teacher_m0)]
        if teacher_fused:
            command.append("--teacher_fused")
        if int(teacher_warmup_epoch or 0) > 0:
            command += ["--teacher_warmup_epoch", str(int(teacher_warmup_epoch))]
    if seed is not None and int(seed) >= 0:
        command += ["--seed", str(int(seed))]
    if imb_ratio is not None:
        command += ["--imb_ratio", str(int(imb_ratio))]
    if early_stop_patience is not None and int(early_stop_patience) > 0:
        command += ["--early_stop_patience", str(int(early_stop_patience))]
    if fp16:
        command.append("--fp16")
    command += ["--gate_init", str(float(gate_init))]
    if resume:
        command += ["--resume", str(resume)]
    if extra_args:
        command += extra_args

    # Preemption-safe: train.py saves model.pt every epoch, so at most 1
    # epoch is lost. Always commit the volume even when Modal preempts
    # the container, otherwise checkpoints die with the container.
    try:
        _run(command)
    finally:
        try:
            storage_volume.commit()
        except Exception as e:
            print(f'[volume] commit failed (non-fatal): {e}', flush=True)
    return {"experiment_name": exp_name, "experiment_dir": str(exp_root)}


_PSEUDO_PARAMS = dict(
    enable_pseudo_labeling=False, pseudo_mode=1, confidence_threshold=0.9,
    pseudo_top_ratio=0.8, max_samples_per_class=500, pseudo_update_freq=10,
    max_pseudo_iterations=20, pseudo_warmup_epoch=30, pseudo_conf_bar=0.5,
    pseudo_bar_k=2.0, pseudo_min_hi=10, enable_novel_pseudo=False,
    novel_warmup_epoch=50, novel_update_freq=10, max_novel_iterations=2,
    novel_max_samples=100, novel_jaccard_th=0.6, novel_agree_th=0.7,
    novel_min_size=10)


@app.local_entrypoint()
def main(
    dataset_name: str = "cub",
    backbone: str = "dinov2_vitb14",
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 0.1,
    sup_weight: float = 0.35,
    memax_weight: float = 2,
    grad_from_block: int = 11,
    num_workers: int = 8,
    extra_args: str = "",
    exp_name_suffix: str = "",
    seed: int = -1,
    # Imbalance: None = legacy SSB/uniform path; 1 = balanced precomputed
    # split; 10 = long-tailed split (CUB-only for now).
    imb_ratio: int | None = None,
    enable_pseudo_labeling: bool = False,
    pseudo_mode: int = 1,
    confidence_threshold: float = 0.9,
    pseudo_top_ratio: float = 0.8,
    max_samples_per_class: int = 500,
    pseudo_update_freq: int = 10,
    max_pseudo_iterations: int = 20,
    pseudo_warmup_epoch: int = 30,
    pseudo_conf_bar: float = 0.5,
    pseudo_bar_k: float = 2.0,
    pseudo_min_hi: int = 10,
    enable_novel_pseudo: bool = False,
    novel_warmup_epoch: int = 50,
    novel_update_freq: int = 10,
    max_novel_iterations: int = 2,
    novel_max_samples: int = 100,
    novel_jaccard_th: float = 0.6,
    novel_agree_th: float = 0.7,
    novel_min_size: int = 10,
    # AdaPart fused-logits
    use_parts: bool = False,
    num_slots: int = 3,
    part_lambda: float = 0.5,
    tau_c: float = 0.1,
    ablate_confidence: bool = False,
    # Momentum teacher (porter BaCon A1)
    use_momentum_teacher: bool = False,
    teacher_m0: float = 0.996,
    teacher_fused: bool = False,
    teacher_warmup_epoch: int = 0,
    # Early stopping: 0 = off (legacy run-all-epochs); >0 = patience.
    early_stop_patience: int = 0,
    # Mixed precision: big free speedup on A100/H100. Validate vs fp32 once.
    fp16: bool = False,
    # Gate init constant: -1.0 = closed-start (legacy); 0.0 = open-start.
    gate_init: float = -1.0,
    resume: str | None = None,
) -> None:
    """Foreground run (streams logs; keep machine on) or add -d to detach."""
    # Local pre-flight: rebuild the exact command the remote would run and
    # validate every flag against simgcd/train.py. Fails HERE (free) instead
    # of on Modal (wastes a run).
    _preflight_check(
        dataset_name=dataset_name, backbone=backbone, epochs=epochs,
        batch_size=batch_size, lr=lr, sup_weight=sup_weight,
        memax_weight=memax_weight, grad_from_block=grad_from_block,
        num_workers=num_workers,
        extra_args=extra_args.split() if extra_args else [],
        enable_pseudo_labeling=enable_pseudo_labeling, pseudo_mode=pseudo_mode,
        confidence_threshold=confidence_threshold, pseudo_top_ratio=pseudo_top_ratio,
        max_samples_per_class=max_samples_per_class, pseudo_update_freq=pseudo_update_freq,
        max_pseudo_iterations=max_pseudo_iterations, pseudo_warmup_epoch=pseudo_warmup_epoch,
        pseudo_conf_bar=pseudo_conf_bar, pseudo_bar_k=pseudo_bar_k,
        pseudo_min_hi=pseudo_min_hi, enable_novel_pseudo=enable_novel_pseudo,
        novel_warmup_epoch=novel_warmup_epoch, novel_update_freq=novel_update_freq,
        max_novel_iterations=max_novel_iterations, novel_max_samples=novel_max_samples,
        novel_jaccard_th=novel_jaccard_th, novel_agree_th=novel_agree_th,
        novel_min_size=novel_min_size, seed=seed,
        use_momentum_teacher=use_momentum_teacher, teacher_m0=teacher_m0,
        teacher_fused=teacher_fused, teacher_warmup_epoch=teacher_warmup_epoch,
        imb_ratio=imb_ratio, early_stop_patience=early_stop_patience, fp16=fp16,
        gate_init=gate_init, resume=resume,
    )
    result = train.remote(
        dataset_name=dataset_name, backbone=backbone, epochs=epochs,
        batch_size=batch_size, lr=lr, sup_weight=sup_weight,
        memax_weight=memax_weight, grad_from_block=grad_from_block,
        num_workers=num_workers,
        extra_args=extra_args.split() if extra_args else [],
        exp_name_suffix=exp_name_suffix,
        enable_pseudo_labeling=enable_pseudo_labeling, pseudo_mode=pseudo_mode,
        confidence_threshold=confidence_threshold, pseudo_top_ratio=pseudo_top_ratio,
        max_samples_per_class=max_samples_per_class, pseudo_update_freq=pseudo_update_freq,
        max_pseudo_iterations=max_pseudo_iterations, pseudo_warmup_epoch=pseudo_warmup_epoch,
        pseudo_conf_bar=pseudo_conf_bar, pseudo_bar_k=pseudo_bar_k,
        pseudo_min_hi=pseudo_min_hi, enable_novel_pseudo=enable_novel_pseudo,
        novel_warmup_epoch=novel_warmup_epoch, novel_update_freq=novel_update_freq,
        max_novel_iterations=max_novel_iterations, novel_max_samples=novel_max_samples,
        novel_jaccard_th=novel_jaccard_th, novel_agree_th=novel_agree_th,
        novel_min_size=novel_min_size, use_parts=use_parts, num_slots=num_slots,
        part_lambda=part_lambda, tau_c=tau_c, ablate_confidence=ablate_confidence,
        use_momentum_teacher=use_momentum_teacher, teacher_m0=teacher_m0,
        teacher_fused=teacher_fused, teacher_warmup_epoch=teacher_warmup_epoch,
        seed=seed, imb_ratio=imb_ratio,
        early_stop_patience=early_stop_patience, fp16=fp16,
        gate_init=gate_init, resume=resume,
    )
    print(f"\nDone: {result['experiment_name']}\nDir: {result['experiment_dir']}")


@app.local_entrypoint()
def launch(
    dataset_name: str = "cub",
    backbone: str = "dinov2_vitb14",
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 0.1,
    sup_weight: float = 0.35,
    memax_weight: float = 2,
    grad_from_block: int = 11,
    num_workers: int = 8,
    extra_args: str = "",
    exp_name_suffix: str = "",
    seed: int = -1,
    # Imbalance: None = legacy SSB/uniform path; 1 = balanced precomputed
    # split; 10 = long-tailed split (CUB-only for now).
    imb_ratio: int | None = None,
    enable_pseudo_labeling: bool = False,
    pseudo_mode: int = 1,
    confidence_threshold: float = 0.9,
    pseudo_top_ratio: float = 0.8,
    max_samples_per_class: int = 500,
    pseudo_update_freq: int = 10,
    max_pseudo_iterations: int = 20,
    pseudo_warmup_epoch: int = 30,
    pseudo_conf_bar: float = 0.5,
    pseudo_bar_k: float = 2.0,
    pseudo_min_hi: int = 10,
    enable_novel_pseudo: bool = False,
    novel_warmup_epoch: int = 50,
    novel_update_freq: int = 10,
    max_novel_iterations: int = 2,
    novel_max_samples: int = 100,
    novel_jaccard_th: float = 0.6,
    novel_agree_th: float = 0.7,
    novel_min_size: int = 10,
    # AdaPart fused-logits
    use_parts: bool = False,
    num_slots: int = 3,
    part_lambda: float = 0.5,
    tau_c: float = 0.1,
    ablate_confidence: bool = False,
    # Momentum teacher (porter BaCon A1)
    use_momentum_teacher: bool = False,
    teacher_m0: float = 0.996,
    teacher_fused: bool = False,
    teacher_warmup_epoch: int = 0,
    # Early stopping: 0 = off (legacy run-all-epochs); >0 = patience.
    early_stop_patience: int = 0,
    # Mixed precision: big free speedup on A100/H100. Validate vs fp32 once.
    fp16: bool = False,
    # Gate init constant: -1.0 = closed-start (legacy); 0.0 = open-start.
    gate_init: float = -1.0,
    resume: str | None = None,
) -> None:
    """Fire-and-forget launch: spawns the run detached on Modal and exits.

    IMPORTANT: invoke with the --detach flag so Modal keeps the app running
    after this local process exits (without it the app is torn down as soon
    as the entrypoint returns) — same pattern as BaCon/modal_train.py::launch:

        modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dino-vitb16 --exp-name-suffix S0v1

    Follow progress with:
        modal app logs simgcd-train
        modal app list
    """
    # Same free pre-flight as main(): bad flags fail HERE, not on Modal.
    _preflight_check(
        dataset_name=dataset_name, backbone=backbone, epochs=epochs,
        batch_size=batch_size, lr=lr, sup_weight=sup_weight,
        memax_weight=memax_weight, grad_from_block=grad_from_block,
        num_workers=num_workers,
        extra_args=extra_args.split() if extra_args else [],
        enable_pseudo_labeling=enable_pseudo_labeling, pseudo_mode=pseudo_mode,
        confidence_threshold=confidence_threshold, pseudo_top_ratio=pseudo_top_ratio,
        max_samples_per_class=max_samples_per_class, pseudo_update_freq=pseudo_update_freq,
        max_pseudo_iterations=max_pseudo_iterations, pseudo_warmup_epoch=pseudo_warmup_epoch,
        pseudo_conf_bar=pseudo_conf_bar, pseudo_bar_k=pseudo_bar_k,
        pseudo_min_hi=pseudo_min_hi, enable_novel_pseudo=enable_novel_pseudo,
        novel_warmup_epoch=novel_warmup_epoch, novel_update_freq=novel_update_freq,
        max_novel_iterations=max_novel_iterations, novel_max_samples=novel_max_samples,
        novel_jaccard_th=novel_jaccard_th, novel_agree_th=novel_agree_th,
        novel_min_size=novel_min_size, use_parts=use_parts, num_slots=num_slots,
        part_lambda=part_lambda, tau_c=tau_c, ablate_confidence=ablate_confidence,
        use_momentum_teacher=use_momentum_teacher, teacher_m0=teacher_m0,
        teacher_fused=teacher_fused, teacher_warmup_epoch=teacher_warmup_epoch,
        seed=seed, imb_ratio=imb_ratio,
        early_stop_patience=early_stop_patience, fp16=fp16,
        gate_init=gate_init, resume=resume,
    )
    call = train.spawn(
        dataset_name=dataset_name, backbone=backbone, epochs=epochs,
        batch_size=batch_size, lr=lr, sup_weight=sup_weight,
        memax_weight=memax_weight, grad_from_block=grad_from_block,
        num_workers=num_workers,
        extra_args=extra_args.split() if extra_args else [],
        exp_name_suffix=exp_name_suffix,
        enable_pseudo_labeling=enable_pseudo_labeling, pseudo_mode=pseudo_mode,
        confidence_threshold=confidence_threshold, pseudo_top_ratio=pseudo_top_ratio,
        max_samples_per_class=max_samples_per_class, pseudo_update_freq=pseudo_update_freq,
        max_pseudo_iterations=max_pseudo_iterations, pseudo_warmup_epoch=pseudo_warmup_epoch,
        pseudo_conf_bar=pseudo_conf_bar, pseudo_bar_k=pseudo_bar_k,
        pseudo_min_hi=pseudo_min_hi, enable_novel_pseudo=enable_novel_pseudo,
        novel_warmup_epoch=novel_warmup_epoch, novel_update_freq=novel_update_freq,
        max_novel_iterations=max_novel_iterations, novel_max_samples=novel_max_samples,
        novel_jaccard_th=novel_jaccard_th, novel_agree_th=novel_agree_th,
        novel_min_size=novel_min_size, use_parts=use_parts, num_slots=num_slots,
        part_lambda=part_lambda, tau_c=tau_c, ablate_confidence=ablate_confidence,
        use_momentum_teacher=use_momentum_teacher, teacher_m0=teacher_m0,
        teacher_fused=teacher_fused, teacher_warmup_epoch=teacher_warmup_epoch,
        seed=seed, imb_ratio=imb_ratio,
        early_stop_patience=early_stop_patience, fp16=fp16,
        gate_init=gate_init, resume=resume,
    )
    print("\n" + "=" * 60)
    print(f"Spawned detached Modal run (call id: {call.object_id})")
    print("It keeps running when you close this terminal / shut down.")
    print("=" * 60)
    print("Follow logs:   modal app logs simgcd-train")
    print("List apps:     modal app list")
    print("Stop manually: modal app stop simgcd-train")
    print()
    print("NOTE: if you did not pass --detach (-d) to 'modal run', this app")
    print("was already torn down when the entrypoint returned. Re-run with:")
    print("  modal run -d modal_train_simgcd.py::launch ...")


if __name__ == "__main__":
    main()
