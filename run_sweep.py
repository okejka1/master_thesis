#!/usr/bin/env python3
"""
run_sweep.py — Local machine-unlearning sweep runner.

Replaces sweep_kaggle.ipynb and sweep_classwise_kaggle.ipynb.
Crash-safe: completed runs (status='complete' in their JSON) are skipped on
re-run, so you can interrupt and resume freely.

Checkpoint layout produced
--------------------------
checkpoints/
└── seed_<N>/
    └── <dataset>/                         e.g. cifar10/
        ├── base/                          ← train.py output
        │   ├── resnet18_cifar10_best.pth
        │   └── resnet18_cifar10_final.pth
        ├── sisa/                          ← train_sisa.py output (shared across both sweeps)
        │   └── sisa_cifar10/
        │       ├── ensemble_meta.json
        │       ├── shard_assignments.json
        │       └── shard_0/ … shard_4/
        ├── sample_wise/
        │   ├── naive/
        │   │   ├── frac_01pct/naive_cifar10_results.json
        │   │   └── frac_05pct/…
        │   ├── grad_tau/
        │   │   └── frac_01pct/grad_tau_cifar10_results.json
        │   └── sisa/
        │       └── frac_01pct/unlearn_results.json
        └── class_wise/
            ├── naive/
            │   └── class_0/naive_cifar10_results.json
            ├── grad_tau/
            └── sisa/
                └── class_0/unlearn_results.json

Collected results per seed
--------------------------
checkpoints/seed_<N>/seed_<N>_cifar10_sample_results.json
checkpoints/seed_<N>/seed_<N>_cifar10_class_results.json

Usage
-----
    # Both sweeps, seeds 0-2, CIFAR-10 (default):
    python run_sweep.py

    # Sample-wise only, single seed:
    python run_sweep.py --mode sample --seeds 0

    # Class-wise only, skip naive (fastest):
    python run_sweep.py --mode class --no-naive

    # CIFAR-100, specific fractions:
    python run_sweep.py --dataset cifar100 --fractions 0.01 0.05

    # Collect results without re-running anything:
    python run_sweep.py --collect-only

    # Skip SISA entirely:
    python run_sweep.py --no-sisa
"""

import argparse
import json
import os
import subprocess
import sys
import time

# ── Repo root (script lives next to train.py etc.) ────────────────────────────
REPO_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Crash-safe sentinel ───────────────────────────────────────────────────────

def is_complete(json_path: str) -> bool:
    """True only when the JSON exists AND contains status='complete'."""
    if not os.path.exists(json_path):
        return False
    try:
        with open(json_path) as f:
            return json.load(f).get("status") == "complete"
    except Exception:
        return False


def _get_status(json_path: str) -> str:
    if not os.path.exists(json_path):
        return "missing"
    try:
        return json.load(open(json_path)).get("status", "unknown")
    except Exception:
        return "corrupt"


# ── Subprocess runner ─────────────────────────────────────────────────────────

def run_script(script_args: list, sentinel: str | None = None, label: str = "") -> bool:
    """
    Run  python <script_args>  from REPO_DIR.
    Skips silently if sentinel JSON already has status='complete'.
    Returns True on success / skip, False on non-zero exit.
    """
    tag = f"[{label}]" if label else ""
    if sentinel and is_complete(sentinel):
        print(f"  {tag} SKIP — already complete")
        return True
    cmd = [sys.executable] + [str(a) for a in script_args]
    print(f"\n  {tag} RUN  {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=REPO_DIR).returncode
    elapsed = time.time() - t0
    status = "OK" if rc == 0 else f"ERROR rc={rc}"
    print(f"  {tag} {status}  ({elapsed:.0f}s)")
    return rc == 0


# ── Directory helpers ─────────────────────────────────────────────────────────

def base_dir(ckpt_root: str, seed: int, dataset: str) -> str:
    return os.path.join(ckpt_root, f"seed_{seed}", dataset, "base")


def sisa_base_dir(ckpt_root: str, seed: int, dataset: str) -> str:
    """Root passed to train_sisa.py / unlearn_sisa.py --checkpoint-dir."""
    return os.path.join(ckpt_root, f"seed_{seed}", dataset, "sisa")


def ovr_base_dir(ckpt_root: str, seed: int, dataset: str) -> str:
    """Root passed to ovr.py --checkpoint-dir (holds the ovr_<dataset>/ tree)."""
    return os.path.join(ckpt_root, f"seed_{seed}", dataset, "ovr")


def method_out_dir(ckpt_root: str, seed: int, dataset: str,
                   sweep: str, method: str, key: str) -> str:
    """
    sweep  : "sample_wise" | "class_wise"
    method : "naive" | "grad_tau" | "sisa"
    key    : "frac_01pct" | "class_0" …
    """
    return os.path.join(ckpt_root, f"seed_{seed}", dataset, sweep, method, key)


def frac_key(frac: float) -> str:
    """
    Produce a unique, filesystem-safe directory name for any forget fraction.

    Examples
    --------
    0.0001 → frac_0p01pct
    0.001  → frac_0p1pct
    0.005  → frac_0p5pct
    0.01   → frac_01pct
    0.05   → frac_05pct
    0.10   → frac_10pct
    """
    pct = round(frac * 100, 6)          # avoid floating-point drift
    if pct >= 1.0 and pct == int(pct):  # whole-number percentage: use zero-padded int
        return f"frac_{int(pct):02d}pct"
    # Sub-1% or fractional: replace '.' with 'p' so the name is filesystem-safe
    s = f"{pct:.4g}".replace(".", "p")
    return f"frac_{s}pct"


def class_key(cls: int) -> str:
    return f"class_{cls}"


# ── Result JSON path per method ───────────────────────────────────────────────

def result_json(ckpt_root: str, seed: int, dataset: str,
                sweep: str, method: str, key: str) -> str:
    d = method_out_dir(ckpt_root, seed, dataset, sweep, method, key)
    if method == "naive":
        return os.path.join(d, f"naive_{dataset}_results.json")
    if method == "grad_tau":
        return os.path.join(d, f"grad_tau_{dataset}_results.json")
    if method == "sisa":
        return os.path.join(d, "unlearn_results.json")
    if method == "ovr":
        return os.path.join(d, "unlearn_results.json")
    raise ValueError(f"Unknown method: {method!r}")


# ── Training ──────────────────────────────────────────────────────────────────

def maybe_train_base(ckpt_root: str, seed: int, dataset: str,
                     data_root: str) -> None:
    bd = base_dir(ckpt_root, seed, dataset)
    os.makedirs(bd, exist_ok=True)
    sentinel = os.path.join(bd, f"resnet18_{dataset}_best.pth")
    if os.path.exists(sentinel):
        print(f"  [train base] SKIP — checkpoint exists")
        return
    run_script(
        ["train.py",
         "--config",         f"configs/{dataset}.yaml",
         "--data-root",      data_root,
         "--checkpoint-dir", bd,
         "--seed",           seed],
        label=f"train base  seed={seed}  {dataset}",
    )


def maybe_train_sisa(ckpt_root: str, seed: int, dataset: str,
                     data_root: str) -> None:
    sd = sisa_base_dir(ckpt_root, seed, dataset)
    os.makedirs(sd, exist_ok=True)
    sentinel = os.path.join(sd, f"sisa_{dataset}", "ensemble_meta.json")
    if os.path.exists(sentinel):
        print(f"  [train sisa] SKIP — ensemble_meta.json exists")
        return
    run_script(
        ["train_sisa.py",
         "--config",         f"configs/{dataset}.yaml",
         "--data-root",      data_root,
         "--checkpoint-dir", sd,
         "--seed",           seed],
        label=f"train sisa  seed={seed}  {dataset}",
    )


def maybe_train_ovr(ckpt_root: str, seed: int, dataset: str,
                    data_root: str) -> None:
    od = ovr_base_dir(ckpt_root, seed, dataset)
    os.makedirs(od, exist_ok=True)
    sentinel = os.path.join(od, f"ovr_{dataset}", "ensemble_meta.json")
    if os.path.exists(sentinel):
        print(f"  [train ovr] SKIP — ensemble_meta.json exists")
        return
    run_script(
        ["ovr.py", "--mode", "train",
         "--config",         f"configs/{dataset}.yaml",
         "--data-root",      data_root,
         "--checkpoint-dir", od,
         "--seed",           seed],
        label=f"train ovr  seed={seed}  {dataset}",
    )


# ── Sample-wise sweep ─────────────────────────────────────────────────────────

def run_sample_sweep(ckpt_root: str, seed: int, dataset: str, data_root: str,
                     fractions: list[float], methods: list[str],
                     save_unlearned_ckpts: bool = False) -> None:
    original_ckpt = os.path.join(
        base_dir(ckpt_root, seed, dataset), f"resnet18_{dataset}_best.pth")
    sd = sisa_base_dir(ckpt_root, seed, dataset)
    od = ovr_base_dir(ckpt_root, seed, dataset)

    for frac in fractions:
        key = frac_key(frac)
        pct = f"{frac * 100:.4g}%"
        print(f"\n{'='*62}")
        print(f"  sample_wise  frac={pct}  seed={seed}  {dataset}")
        print(f"{'='*62}")

        if "naive" in methods:
            out = method_out_dir(ckpt_root, seed, dataset, "sample_wise", "naive", key)
            os.makedirs(out, exist_ok=True)
            run_script(
                ["unlearn_naive.py",
                 "--config",          f"configs/{dataset}.yaml",
                 "--data-root",       data_root,
                 "--checkpoint-dir",  out,
                 "--original-ckpt",   original_ckpt,
                 "--seed",            seed,
                 "--forget-fraction", frac],
                sentinel=result_json(ckpt_root, seed, dataset, "sample_wise", "naive", key),
                label=f"naive     frac={pct}",
            )

        if "grad_tau" in methods:
            out = method_out_dir(ckpt_root, seed, dataset, "sample_wise", "grad_tau", key)
            os.makedirs(out, exist_ok=True)
            run_script(
                ["unlearn_grad_tau.py",
                 "--config",          f"configs/{dataset}.yaml",
                 "--data-root",       data_root,
                 "--checkpoint-dir",  out,
                 "--original-ckpt",   original_ckpt,
                 "--seed",            seed,
                 "--forget-fraction", frac],
                sentinel=result_json(ckpt_root, seed, dataset, "sample_wise", "grad_tau", key),
                label=f"grad_tau  frac={pct}",
            )

        if "sisa" in methods:
            out = method_out_dir(ckpt_root, seed, dataset, "sample_wise", "sisa", key)
            os.makedirs(out, exist_ok=True)
            sisa_args = ["unlearn_sisa.py",
                         "--config",          f"configs/{dataset}.yaml",
                         "--data-root",       data_root,
                         "--checkpoint-dir",  sd,
                         "--output-dir",      out,
                         "--seed",            seed,
                         "--forget-fraction", frac]
            if save_unlearned_ckpts:
                sisa_args.append("--save-unlearned-ckpts")
            run_script(
                sisa_args,
                sentinel=result_json(ckpt_root, seed, dataset, "sample_wise", "sisa", key),
                label=f"sisa      frac={pct}",
            )

        if "ovr" in methods:
            out = method_out_dir(ckpt_root, seed, dataset, "sample_wise", "ovr", key)
            os.makedirs(out, exist_ok=True)
            run_script(
                ["ovr.py", "--mode", "unlearn",
                 "--config",          f"configs/{dataset}.yaml",
                 "--data-root",       data_root,
                 "--checkpoint-dir",  od,
                 "--output-dir",      out,
                 "--seed",            seed,
                 "--forget-fraction", frac],   # random strategy → variant auto = slice_resume
                sentinel=result_json(ckpt_root, seed, dataset, "sample_wise", "ovr", key),
                label=f"ovr       frac={pct}",
            )


# ── Class-wise sweep ──────────────────────────────────────────────────────────

def run_class_sweep(ckpt_root: str, seed: int, dataset: str, data_root: str,
                    classes: list[int], methods: list[str],
                    save_unlearned_ckpts: bool = False) -> None:
    original_ckpt = os.path.join(
        base_dir(ckpt_root, seed, dataset), f"resnet18_{dataset}_best.pth")
    sd = sisa_base_dir(ckpt_root, seed, dataset)
    od = ovr_base_dir(ckpt_root, seed, dataset)

    for cls in classes:
        key = class_key(cls)
        print(f"\n{'='*62}")
        print(f"  class_wise  class={cls}  seed={seed}  {dataset}")
        print(f"{'='*62}")

        if "naive" in methods:
            out = method_out_dir(ckpt_root, seed, dataset, "class_wise", "naive", key)
            os.makedirs(out, exist_ok=True)
            run_script(
                ["unlearn_naive.py",
                 "--config",          f"configs/{dataset}.yaml",
                 "--data-root",       data_root,
                 "--checkpoint-dir",  out,
                 "--original-ckpt",   original_ckpt,
                 "--seed",            seed,
                 "--forget-strategy", "class",
                 "--forget-class",    cls],
                sentinel=result_json(ckpt_root, seed, dataset, "class_wise", "naive", key),
                label=f"naive     class={cls}",
            )

        if "grad_tau" in methods:
            out = method_out_dir(ckpt_root, seed, dataset, "class_wise", "grad_tau", key)
            os.makedirs(out, exist_ok=True)
            run_script(
                ["unlearn_grad_tau.py",
                 "--config",          f"configs/{dataset}.yaml",
                 "--data-root",       data_root,
                 "--checkpoint-dir",  out,
                 "--original-ckpt",   original_ckpt,
                 "--seed",            seed,
                 "--forget-strategy", "class",
                 "--forget-class",    cls],
                sentinel=result_json(ckpt_root, seed, dataset, "class_wise", "grad_tau", key),
                label=f"grad_tau  class={cls}",
            )

        if "sisa" in methods:
            out = method_out_dir(ckpt_root, seed, dataset, "class_wise", "sisa", key)
            os.makedirs(out, exist_ok=True)
            sisa_args = ["unlearn_sisa.py",
                         "--config",          f"configs/{dataset}.yaml",
                         "--data-root",       data_root,
                         "--checkpoint-dir",  sd,
                         "--output-dir",      out,
                         "--seed",            seed,
                         "--forget-strategy", "class",
                         "--forget-class",    cls]
            if save_unlearned_ckpts:
                sisa_args.append("--save-unlearned-ckpts")
            run_script(
                sisa_args,
                sentinel=result_json(ckpt_root, seed, dataset, "class_wise", "sisa", key),
                label=f"sisa      class={cls}",
            )

        if "ovr" in methods:
            out = method_out_dir(ckpt_root, seed, dataset, "class_wise", "ovr", key)
            os.makedirs(out, exist_ok=True)
            run_script(
                ["ovr.py", "--mode", "unlearn",
                 "--config",          f"configs/{dataset}.yaml",
                 "--data-root",       data_root,
                 "--checkpoint-dir",  od,
                 "--output-dir",      out,
                 "--seed",            seed,
                 "--forget-strategy", "class",
                 "--forget-class",    cls],   # class strategy → variant auto = drop
                sentinel=result_json(ckpt_root, seed, dataset, "class_wise", "ovr", key),
                label=f"ovr       class={cls}",
            )


# ── Collect ───────────────────────────────────────────────────────────────────

def collect(ckpt_root: str, seed: int, dataset: str,
            fractions: list[float], classes: list[int],
            methods: list[str], do_sample: bool, do_class: bool) -> None:
    """Gather all complete results for this seed into per-sweep JSON files."""

    seed_root = os.path.join(ckpt_root, f"seed_{seed}")
    os.makedirs(seed_root, exist_ok=True)

    for sweep, keys, label_field in [
        ("sample_wise", [frac_key(f) for f in fractions], "frac"),
        ("class_wise",  [class_key(c) for c in classes],  "class"),
    ]:
        if sweep == "sample_wise" and not do_sample:
            continue
        if sweep == "class_wise" and not do_class:
            continue

        records, missing = [], []
        raw_keys = fractions if sweep == "sample_wise" else classes

        for raw_key, key in zip(raw_keys, keys):
            for method in methods:
                fp = result_json(ckpt_root, seed, dataset, sweep, method, key)
                if is_complete(fp):
                    with open(fp) as f:
                        records.append(json.load(f))
                else:
                    missing.append({
                        "method": method,
                        label_field: raw_key,
                        "status": _get_status(fp),
                    })

        out_path = os.path.join(seed_root,
                                f"seed_{seed}_{dataset}_{sweep}_results.json")
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"\n  ✓ {len(records)} complete runs  →  {out_path}")

        if missing:
            print(f"  ✗ {len(missing)} incomplete:")
            for m in missing:
                val = m.get("frac", m.get("class"))
                print(f"      {m['method']:<12}  {label_field}={val}  "
                      f"status={m['status']}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Local machine-unlearning sweep runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", default="both",
                   choices=["sample", "class", "both"],
                   help="Which sweep(s) to run.")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                   help="List of seeds to run.")
    p.add_argument("--dataset", default="cifar10",
                   choices=["cifar10", "cifar100"],
                   help="Dataset to use.")
    p.add_argument("--fractions", nargs="+", type=float,
                   default=[0.005, 0.01, 0.05, 0.10],
                   help="Forget fractions for sample-wise sweep.")
    p.add_argument("--classes", nargs="+", type=int, default=None,
                   help="Class indices for class-wise sweep. "
                        "Defaults to all classes (10 for CIFAR-10, 100 for CIFAR-100).")
    p.add_argument("--no-naive",    dest="no_naive",    action="store_true",
                   help="Skip naive retrain.")
    p.add_argument("--no-grad-tau", dest="no_grad_tau", action="store_true",
                   help="Skip ∇τ unlearning.")
    p.add_argument("--no-sisa",     dest="no_sisa",     action="store_true",
                   help="Skip SISA unlearning (also skips SISA training).")
    p.add_argument("--ovr",         dest="ovr",         action="store_true",
                   help="Enable OvR (One-vs-Rest ensemble) unlearning. OFF by "
                        "default — training c independent ResNet-18s (100 for "
                        "CIFAR-100) is far heavier than the other methods.")
    p.add_argument("--ckpt-dir",  default="./checkpoints",
                   help="Root checkpoint directory.")
    p.add_argument("--data-dir",  default="./data",
                   help="Dataset cache directory.")
    p.add_argument("--collect-only", action="store_true",
                   help="Skip all training/unlearning; only (re-)collect results.")
    p.add_argument("--save-unlearned-ckpts", dest="save_unlearned_ckpts",
                   action="store_true", default=False,
                   help="Save slice_RR_unlearned.pth after each SISA shard retrain. "
                        "Off by default — per-class accuracy and forget-sample "
                        "confidences are captured in the results JSON instead. "
                        "The original slice_RR.pth files are never affected.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve active methods
    methods = []
    if not args.no_naive:    methods.append("naive")
    if not args.no_grad_tau: methods.append("grad_tau")
    if not args.no_sisa:     methods.append("sisa")
    if args.ovr:             methods.append("ovr")
    if not methods:
        print("ERROR: all methods disabled — nothing to run.")
        sys.exit(1)

    # Mode flags
    do_sample = args.mode in ("sample", "both")
    do_class  = args.mode in ("class",  "both")

    # Default classes
    n_classes = 100 if args.dataset == "cifar100" else 10
    classes   = args.classes if args.classes is not None else list(range(n_classes))

    ckpt_root = os.path.abspath(args.ckpt_dir)
    data_root = os.path.abspath(args.data_dir)

    print(f"\n{'#'*65}")
    print(f"  run_sweep.py")
    print(f"  Dataset   : {args.dataset}")
    print(f"  Seeds     : {args.seeds}")
    print(f"  Methods   : {methods}")
    if do_sample:
        print(f"  Fractions : {args.fractions}")
    if do_class:
        print(f"  Classes   : {classes}")
    print(f"  Checkpoints → {ckpt_root}")
    print(f"{'#'*65}\n")

    for seed in args.seeds:
        print(f"\n{'*'*65}")
        print(f"  SEED {seed}")
        print(f"{'*'*65}")

        if not args.collect_only:
            # ── Training ──────────────────────────────────────────────────────
            maybe_train_base(ckpt_root, seed, args.dataset, data_root)
            if "sisa" in methods:
                maybe_train_sisa(ckpt_root, seed, args.dataset, data_root)
            if "ovr" in methods:
                maybe_train_ovr(ckpt_root, seed, args.dataset, data_root)

            # ── Unlearning sweeps ─────────────────────────────────────────────
            if do_sample:
                run_sample_sweep(ckpt_root, seed, args.dataset, data_root,
                                 args.fractions, methods,
                                 save_unlearned_ckpts=args.save_unlearned_ckpts)
            if do_class:
                run_class_sweep(ckpt_root, seed, args.dataset, data_root,
                                classes, methods,
                                save_unlearned_ckpts=args.save_unlearned_ckpts)

        # ── Collect ───────────────────────────────────────────────────────────
        print(f"\n{'─'*62}")
        print(f"  Collecting results  seed={seed}  {args.dataset}")
        print(f"{'─'*62}")
        collect(ckpt_root, seed, args.dataset,
                args.fractions, classes, methods,
                do_sample, do_class)

    print(f"\n{'#'*65}")
    print(f"  Sweep done.")
    print(f"{'#'*65}\n")


if __name__ == "__main__":
    main()
