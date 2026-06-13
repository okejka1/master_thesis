#!/usr/bin/env python3
"""
run_sweep_mucac.py — Sweep runner for MUCAC instance-level unlearning.

Analogiczny do run_sweep.py dla CIFAR — każdy seed trenuje własny model bazowy
i losuje własny D_f (inny podzbiór tożsamości dla tej samej frakcji).

Frakcje (plan, sekcja 6):  0.1%, 1%, 5%, 10%
Seedów:                    5  (domyślnie 0 1 2 3 4)
Metody:                    naive (domyślnie), --sisa, --grad-tau

Układ katalogów:
    checkpoints/
    └── seed_<N>/
        └── mucac/
            ├── base/
            │   ├── mucac_best.pth
            │   └── mucac_final.pth
            ├── sisa/
            │   └── sisa_mucac/          # shardy + ensemble_meta.json
            └── instance_wise/
                ├── naive/
                │   ├── frac_0p1pct/naive_mucac_results.json
                │   ├── frac_01pct/naive_mucac_results.json
                │   ├── frac_05pct/naive_mucac_results.json
                │   └── frac_10pct/naive_mucac_results.json
                ├── sisa/
                │   └── frac_*/sisa_mucac_results.json
                └── grad_tau/
                    └── frac_*/grad_tau_mucac_results.json

    checkpoints/seed_<N>/seed_<N>_mucac_instance_results.json  # agregat per seed

Użycie:
    python run_sweep_mucac.py                              # pełny sweep (naive)
    python run_sweep_mucac.py --seeds 0 1                 # tylko 2 seedy
    python run_sweep_mucac.py --fractions 0.01 0.05       # tylko wybrane frakcje
    python run_sweep_mucac.py --sisa                      # naive + SISA
    python run_sweep_mucac.py --grad-tau --no-naive       # tylko ∇τ
    python run_sweep_mucac.py --sisa --grad-tau           # wszystkie metody
    python run_sweep_mucac.py --train-only                # tylko trenuj modele bazowe
    python run_sweep_mucac.py --collect-only              # tylko zbierz wyniki
"""

import argparse
import json
import os
import subprocess
import sys
import time

REPO_DIR = os.path.dirname(os.path.abspath(__file__))



# helpers

def is_complete(json_path: str) -> bool:
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


# subprocess runner

def run_script(script_args: list, sentinel: str | None = None,
               label: str = "") -> bool:
    tag = f"[{label}]" if label else ""
    if sentinel and is_complete(sentinel):
        print(f"  {tag} SKIP — already complete")
        return True
    cmd = [sys.executable] + [str(a) for a in script_args]
    print(f"\n  {tag} RUN  {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=REPO_DIR).returncode
    elapsed = time.time() - t0
    print(f"  {tag} {'OK' if rc == 0 else f'ERROR rc={rc}'}  ({elapsed:.0f}s)")
    return rc == 0


# directory layout

def frac_key(frac: float) -> str:
    """0.001 → frac_0p1pct,  0.01 → frac_01pct,  0.10 → frac_10pct"""
    pct = round(frac * 100, 6)
    if pct >= 1.0 and pct == int(pct):
        return f"frac_{int(pct):02d}pct"
    s = f"{pct:.4g}".replace(".", "p")
    return f"frac_{s}pct"



def base_dir(ckpt_root: str, seed: int) -> str:
    return os.path.join(ckpt_root, f"seed_{seed}", "mucac", "base")


def sisa_base_dir(ckpt_root: str, seed: int) -> str:
    return os.path.join(ckpt_root, f"seed_{seed}", "mucac", "sisa")


def method_dir(ckpt_root: str, seed: int, method: str, frac: float) -> str:
    return os.path.join(ckpt_root, f"seed_{seed}", "mucac",
                        "instance_wise", method, frac_key(frac))


def result_json(ckpt_root: str, seed: int, method: str, frac: float) -> str:
    filename = ("sisa_mucac_results.json" if method == "sisa"
                else f"{method}_mucac_results.json")
    return os.path.join(method_dir(ckpt_root, seed, method, frac), filename)



# training

def maybe_train_sisa(ckpt_root: str, seed: int, data_root: str) -> None:
    sd       = sisa_base_dir(ckpt_root, seed)
    sentinel = os.path.join(sd, "sisa_mucac", "ensemble_meta.json")
    os.makedirs(sd, exist_ok=True)
    if os.path.exists(sentinel):
        print(f"  [train sisa seed={seed}] SKIP — ensemble_meta.json exists")
        return
    run_script(
        ["train_sisa_mucac.py",
         "--config",         "configs/mucac.yaml",
         "--data-root",      data_root,
         "--checkpoint-dir", sd,
         "--seed",           seed],
        label=f"train sisa  seed={seed}",
    )


def maybe_train_base(ckpt_root: str, seed: int, data_root: str) -> None:
    bd       = base_dir(ckpt_root, seed)
    sentinel = os.path.join(bd, "mucac_best.pth")
    os.makedirs(bd, exist_ok=True)
    if os.path.exists(sentinel):
        print(f"  [train base seed={seed}] SKIP — mucac_best.pth exists")
        return
    run_script(
        ["train_mucac.py",
         "--config",         "configs/mucac.yaml",
         "--data-root",      data_root,
         "--checkpoint-dir", bd,
         "--seed",           seed],
        label=f"train base  seed={seed}",
    )



# sweeps

def run_sisa_sweep(ckpt_root: str, seed: int, data_root: str,
                   fractions: list[float]) -> None:
    sd = sisa_base_dir(ckpt_root, seed)

    for frac in fractions:
        pct = f"{frac * 100:.4g}%"
        out = method_dir(ckpt_root, seed, "sisa", frac)
        os.makedirs(out, exist_ok=True)
        run_script(
            ["unlearn_sisa_mucac.py",
             "--config",          "configs/mucac.yaml",
             "--data-root",       data_root,
             "--checkpoint-dir",  sd,
             "--output-dir",      out,
             "--forget-fraction", frac,
             "--seed",            seed,
             "--save-unlearned-ckpts"],
            sentinel=result_json(ckpt_root, seed, "sisa", frac),
            label=f"sisa   frac={pct}  seed={seed}",
        )


def run_naive_sweep(ckpt_root: str, seed: int, data_root: str,
                    fractions: list[float]) -> None:
    base_ckpt = os.path.join(base_dir(ckpt_root, seed), "mucac_best.pth")

    for frac in fractions:
        pct = f"{frac * 100:.4g}%"
        out = method_dir(ckpt_root, seed, "naive", frac)
        os.makedirs(out, exist_ok=True)
        run_script(
            ["unlearn_naive_mucac.py",
             "--config",          "configs/mucac.yaml",
             "--data-root",       data_root,
             "--checkpoint-dir",  out,
             "--base-ckpt",       base_ckpt,
             "--forget-fraction", frac,
             "--seed",            seed],
            sentinel=result_json(ckpt_root, seed, "naive", frac),
            label=f"naive  frac={pct}  seed={seed}",
        )


def run_grad_tau_sweep(ckpt_root: str, seed: int, data_root: str,
                       fractions: list[float]) -> None:
    base_ckpt = os.path.join(base_dir(ckpt_root, seed), "mucac_best.pth")

    for frac in fractions:
        pct = f"{frac * 100:.4g}%"
        out = method_dir(ckpt_root, seed, "grad_tau", frac)
        os.makedirs(out, exist_ok=True)
        run_script(
            ["unlearn_grad_tau_mucac.py",
             "--config",          "configs/mucac.yaml",
             "--data-root",       data_root,
             "--checkpoint-dir",  out,
             "--base-ckpt",       base_ckpt,
             "--forget-fraction", frac,
             "--seed",            seed],
            sentinel=result_json(ckpt_root, seed, "grad_tau", frac),
            label=f"grad_tau frac={pct}  seed={seed}",
        )



# collect results

def collect(ckpt_root: str, seeds: list[int],
            fractions: list[float], methods: list[str]) -> None:
    all_missing = []

    for seed in seeds:
        records = []
        missing = []
        for method in methods:
            for frac in fractions:
                fp = result_json(ckpt_root, seed, method, frac)
                if is_complete(fp):
                    with open(fp) as f:
                        records.append(json.load(f))
                else:
                    missing.append({
                        "seed":   seed,
                        "method": method,
                        "frac":   frac,
                        "status": _get_status(fp),
                    })

        seed_root = os.path.join(ckpt_root, f"seed_{seed}")
        os.makedirs(seed_root, exist_ok=True)
        out_path = os.path.join(seed_root, f"seed_{seed}_mucac_instance_results.json")
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"\n  ✓ seed={seed}  {len(records)} complete runs → {out_path}")
        all_missing.extend(missing)

    if all_missing:
        print(f"  ✗ {len(all_missing)} incomplete:")
        for m in all_missing:
            print(f"      seed={m['seed']}  {m['method']:<12}  "
                  f"frac={m['frac']}  status={m['status']}")



# cli

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--seeds", nargs="+", type=int,
                   default=[0, 1, 2, 3, 4],
                   help="Seed indices — each trains its own base model and "
                        "draws its own D_f identity set.")
    p.add_argument("--fractions", nargs="+", type=float,
                   default=[0.001, 0.01, 0.05, 0.10],
                   help="Forget fractions (plan: 0.1%%, 1%%, 5%%, 10%%).")
    p.add_argument("--no-naive",     dest="no_naive",    action="store_true")
    p.add_argument("--sisa",         dest="sisa",        action="store_true",
                   help="Enable SISA (trains S×R shard models per seed).")
    p.add_argument("--grad-tau",     dest="grad_tau",    action="store_true",
                   help="Enable ∇τ approximate unlearning.")
    p.add_argument("--ckpt-dir",     default="./checkpoints")
    p.add_argument("--data-dir",     default="./data")
    p.add_argument("--train-only",   action="store_true",
                   help="Only train base models, skip unlearning.")
    p.add_argument("--collect-only", action="store_true",
                   help="Only collect results, skip training/unlearning.")
    return p.parse_args()


# main

def main():
    args      = parse_args()
    ckpt_root = os.path.abspath(args.ckpt_dir)
    data_root = os.path.abspath(args.data_dir)

    methods = []
    if not args.no_naive:
        methods.append("naive")
    if args.sisa:
        methods.append("sisa")
    if args.grad_tau:
        methods.append("grad_tau")
    if not methods and not args.collect_only and not args.train_only:
        print("ERROR: all methods disabled.")
        sys.exit(1)

    print(f"\n{'#'*62}")
    print(f"  run_sweep_mucac.py")
    print(f"  Seeds     : {args.seeds}")
    print(f"  Methods   : {methods}")
    print(f"  Fractions : {[f'{f*100:.4g}%' for f in args.fractions]}")
    print(f"  Checkpoints → {ckpt_root}")
    print(f"{'#'*62}\n")

    if not args.collect_only:
        for seed in args.seeds:
            print(f"\n{'*'*62}")
            print(f"  SEED {seed}")
            print(f"{'*'*62}")

            maybe_train_base(ckpt_root, seed, data_root)
            if "sisa" in methods:
                maybe_train_sisa(ckpt_root, seed, data_root)

            if not args.train_only:
                if "naive" in methods:
                    run_naive_sweep(ckpt_root, seed, data_root, args.fractions)
                if "sisa" in methods:
                    run_sisa_sweep(ckpt_root, seed, data_root, args.fractions)
                if "grad_tau" in methods:
                    run_grad_tau_sweep(ckpt_root, seed, data_root, args.fractions)

    print(f"\n{'─'*62}")
    print(f"  Collecting results")
    print(f"{'─'*62}")
    collect(ckpt_root, args.seeds, args.fractions, methods)

    print(f"\n{'#'*62}")
    print(f"  Sweep done.")
    print(f"{'#'*62}\n")


if __name__ == "__main__":
    main()
