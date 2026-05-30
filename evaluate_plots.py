"""
evaluate_plots.py — Generuje wykresy wyników oduczania maszynowego do pracy magisterskiej.

Wczytuje dane z plików JSON (class-wise i sample-wise) oraz CSV w results/.
Wykresy zapisywane jako PDF + PNG do results/plots/.

Wygenerowane pliki:
  fig_mia                          — MIA przed i po oduczeniu (2×2)
  fig_utility_forget               — dokładność testowa/zapomnienia vs frakcja (2×3)
  fig_speedup                      — przyspieszenie log-log vs frakcja
  fig_tradeoff                     — scatter: A(Df) vs A(Dt)
  fig_time                         — czasy bezwzględne (skala log)
  fig_classwise_{ds}               — słupki po oduczeniu: test/retain/forget, klasowe
  fig_ba_class_{ds}                — przed i po: forget + test acc, klasowe
  fig_ba_sample_{ds}               — przed i po: forget + test acc, próbkowe (per frakcja)
  fig_forget_conf_kde_{ds}         — KDE pewności na zbiorze zapominanym
  fig_per_class_heatmap_{ds}       — heatmapa zmiany dokładności per klasa
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

CHECKPOINTS = Path("checkpoints")
OUT_DIR = Path("results/plots")

COLORS = {
    "naive_retrain": "#1f77b4",
    "grad_tau":      "#ff7f0e",
    "sisa":          "#2ca02c",
}

METHOD_LABELS = {
    "naive_retrain": "Naiwne retrenowanie",
    "grad_tau":      "Grad-Tau",
    "sisa":          "SISA",
}

METHOD_ORDER = {"naive_retrain": 0, "grad_tau": 1, "sisa": 2}
METHODS = ["naive_retrain", "grad_tau", "sisa"]

MIA_UNRELIABLE = 50
MIA_LIMITED = 500


# ---------------------------------------------------------------------------
# Pomocnicze
# ---------------------------------------------------------------------------

def mia_reliability(forget_size: int) -> str:
    if forget_size < MIA_UNRELIABLE:
        return "unreliable"
    if forget_size < MIA_LIMITED:
        return "limited"
    return "ok"


def get_forget_class(r: dict) -> int:
    for i, v in enumerate(r["before"]["per_class_acc_retain"]):
        if v < 1.0:
            return i
    return -1


def _bar_offsets(bar_w: float = 0.12, within_gap: float = 0.02,
                 between_gap: float = 0.07) -> tuple:
    """
    Zwraca (offsets_before, offsets_after, bar_w, total_width) dla 3 metod × 2 stany.
    offsets[i] = przesunięcie środka słupka względem centrum grupy.
    """
    pair_w = 2 * bar_w + within_gap
    total_w = 3 * pair_w + 2 * between_gap
    before_offsets, after_offsets = [], []
    for i in range(3):
        start = -total_w / 2 + i * (pair_w + between_gap)
        before_offsets.append(start + bar_w / 2)
        after_offsets.append(start + bar_w + within_gap + bar_w / 2)
    return before_offsets, after_offsets, bar_w, total_w


def _kde(values, xs):
    """Gaussowska KDE bez scipy."""
    vals = np.array(values, dtype=float)
    n = len(vals)
    if n == 0:
        return np.zeros_like(xs)
    bw = 1.06 * vals.std(ddof=1) * n ** (-0.2) if vals.std() > 0 else 0.01
    return np.mean(
        np.exp(-0.5 * ((xs[:, None] - vals[None, :]) / bw) ** 2), axis=1
    ) / (bw * np.sqrt(2 * np.pi))


def _save(fig, name: str) -> None:
    fig.savefig(OUT_DIR / f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {name}.pdf / {name}.png")


# ---------------------------------------------------------------------------
# Ładowanie danych
# ---------------------------------------------------------------------------

def load_sample_runs(dataset_key: str) -> list:
    path = CHECKPOINTS / "seed_0" / f"seed_0_{dataset_key}_sample_wise_results.json"
    with open(path) as f:
        runs = [r for r in json.load(f) if r.get("status") == "complete"]

    naive_time = {
        r["forget_fraction"]: r["unlearn_time_s"]
        for r in runs if r["method"] == "naive_retrain"
    }

    out = []
    for r in runs:
        b, a = r["before"], r["after"]
        method, t = r["method"], r.get("unlearn_time_s")

        if method == "naive_retrain":
            speedup = 1.0
        elif method == "grad_tau":
            base = naive_time.get(r["forget_fraction"])
            speedup = base / t if base and t else None
        elif method == "sisa":
            full = r.get("sisa_train_time_s")
            speedup = full / t if full and t else None
        else:
            speedup = None

        out.append({
            "method": method,
            "forget_fraction": r["forget_fraction"],
            "forget_size": int(r["forget_size"]),
            "unlearn_time_s": t,
            "sisa_train_time_s": r.get("sisa_train_time_s"),
            "b_test_acc":   b["test_acc"],
            "b_retain_acc": b["retain_acc"],
            "b_forget_acc": b["forget_acc"],
            "b_mia_l": b["mia_l"] * 100,
            "b_mia_e": b["mia_e"] * 100,
            "a_test_acc":   a["test_acc"],
            "a_retain_acc": a["retain_acc"],
            "a_forget_acc": a["forget_acc"],
            "a_mia_l": a["mia_l"] * 100,
            "a_mia_e": a["mia_e"] * 100,
            "speedup": speedup,
            "mia_reliability": mia_reliability(int(r["forget_size"])),
        })

    out.sort(key=lambda r: (r["forget_fraction"], METHOD_ORDER.get(r["method"], 9)))
    return out


def load_class_runs(dataset_key: str) -> list:
    path = CHECKPOINTS / "seed_0" / f"seed_0_{dataset_key}_class_wise_results.json"
    with open(path) as f:
        runs = [r for r in json.load(f) if r.get("status") == "complete"]

    for r in runs:
        r["_forget_class"] = get_forget_class(r)

    naive_time = {
        r["_forget_class"]: r["unlearn_time_s"]
        for r in runs if r["method"] == "naive_retrain"
    }

    out = []
    for r in runs:
        b, a = r["before"], r["after"]
        method, t = r["method"], r.get("unlearn_time_s")
        fc = r["_forget_class"]

        if method == "naive_retrain":
            speedup = 1.0
        elif method == "grad_tau":
            base = naive_time.get(fc)
            speedup = base / t if base and t else None
        elif method == "sisa":
            full = r.get("sisa_train_time_s")
            speedup = full / t if full and t else None
        else:
            speedup = None

        out.append({
            "method": method,
            "forget_class": fc,
            "forget_size": int(r["forget_size"]),
            "unlearn_time_s": t,
            "sisa_train_time_s": r.get("sisa_train_time_s"),
            "b_test_acc":   b["test_acc"],
            "b_retain_acc": b["retain_acc"],
            "b_forget_acc": b["forget_acc"],
            "b_mia_l": b["mia_l"] * 100,
            "b_mia_e": b["mia_e"] * 100,
            "a_test_acc":   a["test_acc"],
            "a_retain_acc": a["retain_acc"],
            "a_forget_acc": a["forget_acc"],
            "a_mia_l": a["mia_l"] * 100,
            "a_mia_e": a["mia_e"] * 100,
            "speedup": speedup,
            "mia_reliability": mia_reliability(int(r["forget_size"])),
            "b_per_class_acc_test":   b.get("per_class_acc_test", []),
            "a_per_class_acc_test":   a.get("per_class_acc_test", []),
            "b_per_class_acc_retain": b.get("per_class_acc_retain", []),
            "a_per_class_acc_retain": a.get("per_class_acc_retain", []),
            "b_per_class_acc_forget": b.get("per_class_acc_forget", []),
            "a_per_class_acc_forget": a.get("per_class_acc_forget", []),
            "b_forget_conf": b.get("forget_conf", []),
            "a_forget_conf": a.get("forget_conf", []),
        })

    out.sort(key=lambda r: (r["forget_class"], METHOD_ORDER.get(r["method"], 9)))
    return out


# ---------------------------------------------------------------------------
# Wykresy próbkowe (sample-wise)
# ---------------------------------------------------------------------------

def plot_mia(sample_runs_c10, sample_runs_c100):
    """MIA-Loss i MIA-Entropy przed i po oduczeniu — siatka 2×2."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey="row")

    mia_configs = [
        ("b_mia_l", "a_mia_l", "MIA-Loss"),
        ("b_mia_e", "a_mia_e", "MIA-Entropy"),
    ]
    datasets = [("CIFAR-10", sample_runs_c10), ("CIFAR-100", sample_runs_c100)]

    for row_idx, (dataset_name, runs) in enumerate(datasets):
        for col_idx, (bkey, akey, mia_name) in enumerate(mia_configs):
            ax = axes[row_idx][col_idx]

            for method in METHODS:
                rows = [r for r in runs if r["method"] == method]
                fracs = sorted(set(r["forget_fraction"] for r in rows))
                fracs_pct = [f * 100 for f in fracs]
                before = [next((r[bkey] for r in rows if r["forget_fraction"] == f), None) for f in fracs]
                after  = [next((r[akey] for r in rows if r["forget_fraction"] == f), None) for f in fracs]
                color  = COLORS[method]
                ax.plot(fracs_pct, before, linestyle="--", color=color, linewidth=1.5,
                        alpha=0.55, marker="o", markersize=5, markerfacecolor="none")
                ax.plot(fracs_pct, after,  linestyle="-",  color=color, linewidth=2,
                        marker="o", markersize=6, label=METHOD_LABELS[method])

            ax.axhline(50, color="black", linestyle=":", linewidth=1.8)
            ax.set_xscale("log")
            ax.set_xlabel("Rozmiar $D_f$ [% zbioru treningowego, skala log]", fontsize=10)
            ax.set_ylabel("Dokładność ataku MIA [%]", fontsize=10)
            ax.set_title(f"{dataset_name} — {mia_name}", fontsize=11, fontweight="bold")
            ax.set_ylim([30, 80])
            ax.grid(True, alpha=0.3)

            if row_idx == 0 and col_idx == 0:
                method_h = [plt.Line2D([0], [0], color=COLORS[m], lw=2, label=METHOD_LABELS[m])
                            for m in METHODS]
                style_h = [
                    plt.Line2D([0], [0], color="gray", lw=1.5, ls="--",
                               marker="o", ms=5, mfc="none", label="Przed oduczeniem"),
                    plt.Line2D([0], [0], color="gray", lw=2, ls="-",
                               marker="o", ms=6, label="Po oduczeniu"),
                    plt.Line2D([0], [0], color="black", lw=1.8, ls=":", label="Ideał (50%)"),
                ]
                ax.legend(handles=method_h + style_h, fontsize=9, loc="best")

    fig.suptitle("Dokładność ataku MIA przed i po oduczeniu",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig_mia")


def plot_utility_forget(sample_runs_c10, sample_runs_c100):
    """Dokładność testowa, zapomnienia i ich różnica vs frakcja — siatka 2×3."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    col_configs = [
        ("b_test_acc",   "a_test_acc",   "Dokładność testowa $A(D_t)$ [%]",       (60, 100), False),
        ("b_forget_acc", "a_forget_acc", "Dokładność na $D_f$: $A(D_f)$ [%]",     (0, 105),  False),
        (None,           None,           "$A(D_t) - A(D_f)$ [pp]",                None,      True),
    ]
    datasets = [("CIFAR-10", sample_runs_c10), ("CIFAR-100", sample_runs_c100)]

    for row_idx, (dataset_name, runs) in enumerate(datasets):
        for col_idx, (bkey, akey, ylabel, ylim, is_diff) in enumerate(col_configs):
            ax = axes[row_idx][col_idx]

            for method in METHODS:
                rows = [r for r in runs if r["method"] == method]
                fracs = sorted(set(r["forget_fraction"] for r in rows))
                fracs_pct = [f * 100 for f in fracs]
                color = COLORS[method]

                if is_diff:
                    at = [next((r["a_test_acc"]   for r in rows if r["forget_fraction"] == f), None) for f in fracs]
                    af = [next((r["a_forget_acc"] for r in rows if r["forget_fraction"] == f), None) for f in fracs]
                    bt = [next((r["b_test_acc"]   for r in rows if r["forget_fraction"] == f), None) for f in fracs]
                    bf = [next((r["b_forget_acc"] for r in rows if r["forget_fraction"] == f), None) for f in fracs]
                    da = [t - g if t and g else None for t, g in zip(at, af)]
                    db = [t - g if t and g else None for t, g in zip(bt, bf)]
                    ax.plot(fracs_pct, db, ls="--", color=color, lw=1.5, alpha=0.55,
                            marker="o", ms=5, mfc="none")
                    ax.plot(fracs_pct, da, ls="-",  color=color, lw=2,
                            marker="o", ms=6, label=METHOD_LABELS[method])
                    ax.axhline(0, color="black", ls=":", lw=1.5, alpha=0.6)
                else:
                    before = [next((r[bkey] for r in rows if r["forget_fraction"] == f), None) for f in fracs]
                    after  = [next((r[akey] for r in rows if r["forget_fraction"] == f), None) for f in fracs]
                    ax.plot(fracs_pct, before, ls="--", color=color, lw=1.5, alpha=0.55,
                            marker="o", ms=5, mfc="none")
                    ax.plot(fracs_pct, after,  ls="-",  color=color, lw=2,
                            marker="o", ms=6, label=METHOD_LABELS[method])

            ax.set_xscale("log")
            ax.set_xlabel("Rozmiar $D_f$ [% zbioru treningowego, skala log]", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            title_str = ylabel.replace("[%]", "").replace("[pp]", "").strip()
            ax.set_title(f"{dataset_name} — {title_str}", fontsize=10, fontweight="bold")
            if ylim:
                ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3)

            if row_idx == 0 and col_idx == 0:
                method_h = [plt.Line2D([0], [0], color=COLORS[m], lw=2, label=METHOD_LABELS[m])
                            for m in METHODS]
                style_h = [
                    plt.Line2D([0], [0], color="gray", lw=1.5, ls="--",
                               marker="o", ms=5, mfc="none", label="Przed oduczeniem"),
                    plt.Line2D([0], [0], color="gray", lw=2, ls="-",
                               marker="o", ms=6, label="Po oduczeniu"),
                ]
                ax.legend(handles=method_h + style_h, fontsize=9, loc="best")

    axes[0][1].text(0.03, 0.04, "niżej = lepiej", transform=axes[0][1].transAxes,
                    fontsize=8, color="gray", style="italic")
    axes[1][1].text(0.03, 0.04, "niżej = lepiej", transform=axes[1][1].transAxes,
                    fontsize=8, color="gray", style="italic")
    axes[0][2].text(0.03, 0.04, "≈ 0 → $D_f$ ≈ $D_t$", transform=axes[0][2].transAxes,
                    fontsize=8, color="gray", style="italic")
    axes[1][2].text(0.03, 0.04, "≈ 0 → $D_f$ ≈ $D_t$", transform=axes[1][2].transAxes,
                    fontsize=8, color="gray", style="italic")

    fig.suptitle("Użyteczność modelu i efektywność zapomnienia vs rozmiar $D_f$",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig_utility_forget")


def plot_speedup(sample_runs_c10, sample_runs_c100):
    """Przyspieszenie (log-log) vs frakcja zapomnienia."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for dataset_name, ax, runs in [("CIFAR-10", ax1, sample_runs_c10),
                                    ("CIFAR-100", ax2, sample_runs_c100)]:
        ax.axhline(1.0, color="gray", ls="--", lw=2,
                   label="Naiwne retrenowanie (1×)")

        for method in ["grad_tau", "sisa"]:
            rows = [r for r in runs if r["method"] == method]
            fracs = sorted(set(r["forget_fraction"] for r in rows))
            speedups = [next((r["speedup"] for r in rows if r["forget_fraction"] == f), None)
                        for f in fracs]
            ax.plot(fracs, speedups, "o-", label=METHOD_LABELS[method],
                    color=COLORS[method], lw=2, ms=7)
            for f, s in zip(fracs, speedups):
                if s is not None:
                    ax.text(f, s * 1.08, f"{s:.1f}×", ha="center", fontsize=9)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Rozmiar $D_f$ [% zbioru treningowego, skala log]", fontsize=11)
        ax.set_ylabel("Przyspieszenie (skala log)", fontsize=11)
        ax.set_title(dataset_name, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=10, loc="best")

    fig.suptitle("Przyspieszenie oduczania względem rozmiaru $D_f$",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    _save(fig, "fig_speedup")


def plot_tradeoff(sample_runs_c10, sample_runs_c100):
    """Scatter kompromisu: $A(D_f)$ vs $A(D_t)$."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    fractions = [0.0001, 0.001, 0.01, 0.1]
    marker_map = {f: m for f, m in zip(fractions, ["o", "s", "^", "D"])}

    for dataset_name, ax, runs in [("CIFAR-10", ax1, sample_runs_c10),
                                    ("CIFAR-100", ax2, sample_runs_c100)]:
        for method in METHODS:
            rows = [r for r in runs if r["method"] == method]
            for row in rows:
                frac = row["forget_fraction"]
                ax.scatter(row["a_forget_acc"], row["a_test_acc"],
                           color=COLORS[method], marker=marker_map.get(frac, "o"),
                           s=150, alpha=0.7,
                           label=METHOD_LABELS[method] if row == rows[0] else "")

        for row in [r for r in runs if r["method"] == "naive_retrain"]:
            ax.annotate(f'{row["forget_fraction"]*100:.2g}%',
                        (row["a_forget_acc"], row["a_test_acc"]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)

        ax.set_xlabel("$A(D_f)$ [%] — niżej = lepiej", fontsize=11)
        ax.set_ylabel("$A(D_t)$ [%] — wyżej = lepiej", fontsize=11)
        ax.set_title(dataset_name, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.95, "Strefa idealna\n(wysoka użyteczność,\nniskie $A(D_f)$)",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=COLORS[m], ms=8, label=METHOD_LABELS[m])
               for m in METHODS]
    fig.legend(handles=handles, loc="upper left",
               bbox_to_anchor=(0.08, 0.97), fontsize=10)
    fig.suptitle("Kompromis użyteczność–zapominanie",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    _save(fig, "fig_tradeoff")


def plot_time(sample_runs_c10, sample_runs_c100):
    """Czasy bezwzględne oduczania (oś Y w skali log)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for dataset_name, ax, runs in [("CIFAR-10", ax1, sample_runs_c10),
                                    ("CIFAR-100", ax2, sample_runs_c100)]:
        fracs = sorted(set(r["forget_fraction"] for r in runs))
        x_pos = np.arange(len(fracs))
        width = 0.25

        for i, method in enumerate(METHODS):
            times = [next((r["unlearn_time_s"] for r in runs
                           if r["method"] == method and r["forget_fraction"] == f), None)
                     for f in fracs]
            ax.bar(x_pos + i * width, times, width, label=METHOD_LABELS[method],
                   color=COLORS[method], alpha=0.8)

            if method != "naive_retrain":
                speedups = [next((r["speedup"] for r in runs
                                  if r["method"] == method and r["forget_fraction"] == f), None)
                            for f in fracs]
                for j, (t, s) in enumerate(zip(times, speedups)):
                    if t and s:
                        ax.text(j + i * width, t * 1.3, f"{s:.1f}×",
                                ha="center", fontsize=8, fontweight="bold")

        ax.set_yscale("log")
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels([f"{f*100:.2g}%" for f in fracs])
        ax.set_xlabel("Rozmiar $D_f$ [% zbioru treningowego]", fontsize=11)
        ax.set_ylabel("Czas [s] (skala log)", fontsize=11)
        ax.set_title(dataset_name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Czas oduczania — porównanie metod",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    _save(fig, "fig_time")


# ---------------------------------------------------------------------------
# Wykresy klasowe (class-wise)
# ---------------------------------------------------------------------------

def plot_classwise(class_runs, dataset_name, filename_base):
    """Słupki: dokładność testowa/retencyjna/zapomnienia po oduczeniu, per zapomniana klasa."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    classes = sorted(set(r["forget_class"] for r in class_runs))
    x_pos = np.arange(len(classes))
    width = 0.25

    metrics = [
        ("a_test_acc",   "Dokładność testowa $A(D_t)$ [%]",        ax1),
        ("a_retain_acc", "Dokładność retencyjna $A(D_r)$ [%]",      ax2),
        ("a_forget_acc", "Dokładność na $D_f$: $A(D_f)$ [%]",       ax3),
    ]

    for metric_key, ylabel, ax in metrics:
        for i, method in enumerate(METHODS):
            vals = [next((r[metric_key] for r in class_runs
                          if r["method"] == method and r["forget_class"] == c), None)
                    for c in classes]
            ax.bar(x_pos + i * width, vals, width, label=METHOD_LABELS[method],
                   color=COLORS[method], alpha=0.8)

        if metric_key == "a_forget_acc":
            ax.axhline(0, color="red", ls="--", lw=1.5, alpha=0.5, label="Ideał (0%)")

        ax.set_xticks(x_pos + width)
        ax.set_xticklabels([f"Klasa {c}" for c in classes])
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Oduczanie klasowe — {dataset_name} (ziarno 0)",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    _save(fig, filename_base)


def _plot_before_after(ax, groups, group_labels, runs_or_class_runs,
                       bkey, akey, ylabel, ideal_line,
                       lookup_key, title):
    """
    Rysuje 6 słupków (3 metody × przed/po) per grupę bez nakładania się.
    Słupki PRZED: kreskowane (hatch='///'), pełne wypełnienie jasnym odcieniem.
    Słupki PO:    pełne, ciemny odcień — jednoznaczna różnica wizualna.
    """
    b_offsets, a_offsets, bar_w, _ = _bar_offsets()
    x_pos = np.arange(len(groups))

    for i, method in enumerate(METHODS):
        b_vals = [next((r[bkey] for r in runs_or_class_runs
                        if r["method"] == method and r[lookup_key] == g), None)
                  for g in groups]
        a_vals = [next((r[akey] for r in runs_or_class_runs
                        if r["method"] == method and r[lookup_key] == g), None)
                  for g in groups]

        # przed: kreskowanie + jasne wypełnienie
        ax.bar(x_pos + b_offsets[i], b_vals, bar_w,
               color=COLORS[method], alpha=0.40,
               hatch="///", edgecolor=COLORS[method], linewidth=0.5)
        # po: pełny słupek
        ax.bar(x_pos + a_offsets[i], a_vals, bar_w,
               color=COLORS[method], alpha=0.90, edgecolor="none")

    if ideal_line is not None:
        ax.axhline(ideal_line, color="red", ls="--", lw=1.5, alpha=0.6)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(group_labels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")

    # Legenda: metody (kolory) + styl (kreskowany = przed, pełny = po)
    method_handles = [mpatches.Patch(color=COLORS[m], label=METHOD_LABELS[m])
                      for m in METHODS]
    style_handles = [
        mpatches.Patch(facecolor="gray", alpha=0.40, hatch="///",
                       edgecolor="gray", label="Przed oduczeniem"),
        mpatches.Patch(facecolor="gray", alpha=0.90,
                       edgecolor="none", label="Po oduczeniu"),
    ]
    if ideal_line is not None:
        style_handles.append(
            plt.Line2D([0], [0], color="red", ls="--", lw=1.5, label="Ideał (0%)")
        )
    ax.legend(handles=method_handles + style_handles, fontsize=9, loc="best")


def plot_ba_class(class_runs, dataset_name, filename_base):
    """Przed i po oduczeniu: $A(D_f)$ i $A(D_t)$ per zapomniana klasa."""
    classes = sorted(set(r["forget_class"] for r in class_runs))
    group_labels = [f"Klasa {c}" for c in classes]

    fig, (ax_f, ax_t) = plt.subplots(1, 2, figsize=(14, 5))

    _plot_before_after(ax_f, classes, group_labels, class_runs,
                       "b_forget_acc", "a_forget_acc",
                       "Dokładność na $D_f$: $A(D_f)$ [%]",
                       ideal_line=0.0, lookup_key="forget_class",
                       title="Dokładność na $D_f$ — przed i po")

    _plot_before_after(ax_t, classes, group_labels, class_runs,
                       "b_test_acc", "a_test_acc",
                       "Dokładność testowa $A(D_t)$ [%]",
                       ideal_line=None, lookup_key="forget_class",
                       title="Dokładność testowa — przed i po")

    fig.suptitle(f"Oduczanie klasowe: przed i po oduczeniu — {dataset_name} (ziarno 0)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, filename_base)


def plot_ba_sample(sample_runs, dataset_name, filename_base):
    """Przed i po oduczeniu: $A(D_f)$ i $A(D_t)$ per frakcja zapomnienia."""
    fracs = sorted(set(r["forget_fraction"] for r in sample_runs))
    group_labels = [f"{f*100:.2g}%" for f in fracs]

    fig, (ax_f, ax_t) = plt.subplots(1, 2, figsize=(14, 5))

    _plot_before_after(ax_f, fracs, group_labels, sample_runs,
                       "b_forget_acc", "a_forget_acc",
                       "Dokładność na $D_f$: $A(D_f)$ [%]",
                       ideal_line=None, lookup_key="forget_fraction",
                       title="Dokładność na $D_f$ — przed i po")

    _plot_before_after(ax_t, fracs, group_labels, sample_runs,
                       "b_test_acc", "a_test_acc",
                       "Dokładność testowa $A(D_t)$ [%]",
                       ideal_line=None, lookup_key="forget_fraction",
                       title="Dokładność testowa — przed i po")

    ax_f.set_xlabel("Frakcja zapomnienia $|D_f|$ / $|D_{train}|$", fontsize=11)
    ax_t.set_xlabel("Frakcja zapomnienia $|D_f|$ / $|D_{train}|$", fontsize=11)

    fig.suptitle(f"Oduczanie próbkowe: przed i po oduczeniu — {dataset_name} (ziarno 0)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, filename_base)


# ---------------------------------------------------------------------------
# Wykresy z danych bogatych (JSON class-wise)
# ---------------------------------------------------------------------------

def plot_forget_conf_kde(class_runs, dataset_name, filename_base):
    """KDE rozkładu pewności modelu na zbiorze zapominanym — przed i po."""
    classes = sorted(set(r["forget_class"] for r in class_runs))
    n_cls = len(classes)
    xs = np.linspace(0.0, 1.0, 400)

    fig, axes = plt.subplots(3, n_cls, figsize=(5 * n_cls, 4 * 3), sharey=False)
    if n_cls == 1:
        axes = [[axes[row]] for row in range(3)]

    for row, method in enumerate(METHODS):
        for col, fc in enumerate(classes):
            ax = axes[row][col]
            run = next((r for r in class_runs
                        if r["method"] == method and r["forget_class"] == fc), None)
            if run is None:
                ax.set_visible(False)
                continue

            b_conf, a_conf = run["b_forget_conf"], run["a_forget_conf"]

            if b_conf:
                dens_b = _kde(b_conf, xs)
                ax.plot(xs, dens_b, color="steelblue", lw=2, ls="--",
                        label="Przed oduczeniem")
                ax.fill_between(xs, dens_b, alpha=0.15, color="steelblue")
            if a_conf:
                dens_a = _kde(a_conf, xs)
                ax.plot(xs, dens_a, color=COLORS[method], lw=2, ls="-",
                        label="Po oduczeniu")
                ax.fill_between(xs, dens_a, alpha=0.20, color=COLORS[method])

            ax.axvline(0.5, color="gray", ls=":", lw=1.2, alpha=0.7)
            ax.set_xlabel("Pewność modelu (zapomniana klasa)", fontsize=9)
            ax.set_ylabel("Gęstość", fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_title(f"{METHOD_LABELS[method]} — Klasa {fc}",
                         fontsize=10, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"Rozkład pewności modelu na zbiorze $D_f$ — {dataset_name} (ziarno 0)\n"
        f"(kolaps w lewo = głębokie zapomnienie)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, filename_base)


def plot_per_class_heatmap(class_runs, dataset_name, filename_base):
    """Heatmapa zmiany $A(D_t)$ per klasa: po − przed oduczeniem."""
    classes_forgotten = sorted(set(r["forget_class"] for r in class_runs))
    sample_run = next((r for r in class_runs if r["b_per_class_acc_test"]), None)
    if sample_run is None:
        print(f"  (pomijam {filename_base}: brak per_class_acc_test)")
        return

    n_model_cls = len(sample_run["b_per_class_acc_test"])
    n_forgotten  = len(classes_forgotten)

    fig, axes = plt.subplots(
        1, 3, figsize=(6 * 3, max(6, n_model_cls * 0.35 + 2))
    )
    vmax = 15.0

    for ax_idx, method in enumerate(METHODS):
        ax = axes[ax_idx]
        mat = np.full((n_model_cls, n_forgotten), np.nan)

        for col, fc in enumerate(classes_forgotten):
            run = next((r for r in class_runs
                        if r["method"] == method and r["forget_class"] == fc), None)
            if run and run["b_per_class_acc_test"] and run["a_per_class_acc_test"]:
                mat[:, col] = (np.array(run["a_per_class_acc_test"])
                               - np.array(run["b_per_class_acc_test"]))

        im = ax.imshow(mat, aspect="auto", cmap="RdBu",
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(n_forgotten))
        ax.set_xticklabels([f"Zapom. {fc}" for fc in classes_forgotten], fontsize=9)
        ax.set_yticks(range(n_model_cls))
        ax.set_yticklabels([f"K{i}" for i in range(n_model_cls)], fontsize=7)
        ax.set_xlabel("Zapomniana klasa", fontsize=10)
        ax.set_ylabel("Klasa modelu", fontsize=10)
        ax.set_title(METHOD_LABELS[method], fontsize=11, fontweight="bold")

        for col, fc in enumerate(classes_forgotten):
            ax.add_patch(plt.Rectangle((col - 0.5, fc - 0.5), 1, 1,
                                       fill=False, edgecolor="yellow", lw=2))

        plt.colorbar(im, ax=ax, label="Δ dokładność [pp]",
                     fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Zmiana dokładności per klasa (po − przed) — {dataset_name}\n"
        f"Żółta ramka = zapomniana klasa. Niebieski = wzrost, Czerwony = spadek.",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, filename_base)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerowanie wykresów → {OUT_DIR}/\n")

    sr10  = load_sample_runs("cifar10")
    sr100 = load_sample_runs("cifar100")
    cr10  = load_class_runs("cifar10")
    cr100 = load_class_runs("cifar100")

    # próbkowe
    plot_mia(sr10, sr100)
    plot_utility_forget(sr10, sr100)
    plot_speedup(sr10, sr100)
    plot_tradeoff(sr10, sr100)
    plot_time(sr10, sr100)

    # przed/po — próbkowe (osobno per dataset)
    plot_ba_sample(sr10,  "CIFAR-10",  "fig_ba_sample_cifar10")
    plot_ba_sample(sr100, "CIFAR-100", "fig_ba_sample_cifar100")

    # klasowe — CIFAR-10
    plot_classwise(cr10,  "CIFAR-10",  "fig_classwise_cifar10")
    plot_ba_class(cr10,   "CIFAR-10",  "fig_ba_class_cifar10")
    plot_forget_conf_kde(cr10,  "CIFAR-10",  "fig_forget_conf_kde_cifar10")
    plot_per_class_heatmap(cr10, "CIFAR-10", "fig_per_class_heatmap_cifar10")

    # klasowe — CIFAR-100
    plot_classwise(cr100, "CIFAR-100", "fig_classwise_cifar100")
    plot_ba_class(cr100,  "CIFAR-100", "fig_ba_class_cifar100")
    plot_forget_conf_kde(cr100, "CIFAR-100", "fig_forget_conf_kde_cifar100")
    plot_per_class_heatmap(cr100, "CIFAR-100", "fig_per_class_heatmap_cifar100")

    print(f"\n✓ Wszystkie wykresy zapisane do {OUT_DIR}/")


if __name__ == "__main__":
    main()
