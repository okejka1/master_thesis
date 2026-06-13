"""
evaluate_plots.py — Script to generate plots from the exctracted results

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


def _save(fig, name: str) -> None:
    fig.savefig(OUT_DIR / f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {name}.pdf / {name}.png")


def _shade_mia_reliability(ax, runs) -> list:
    """Cieniuje strefy niewiarygodnego/ograniczonego MIA i zwraca uchwyty legendy."""
    fracs = sorted(set(r["forget_fraction"] for r in runs))
    rel_map = {r["forget_fraction"]: r["mia_reliability"]
               for r in runs if r["forget_fraction"] not in {}}
    # deduplicate: first occurrence per frac
    rel_map = {}
    for r in runs:
        f = r["forget_fraction"]
        if f not in rel_map:
            rel_map[f] = r["mia_reliability"]

    # Granice stref: geometryczne środki między sąsiednimi frakcjami (skala log)
    bounds = []
    for i, f in enumerate(fracs):
        l = (f * fracs[i - 1]) ** 0.5 if i > 0          else f * 0.4
        r = (f * fracs[i + 1]) ** 0.5 if i < len(fracs) - 1 else f * 2.5
        bounds.append((l * 100, r * 100, rel_map[f]))

    # Scalanie sąsiednich przedziałów tej samej klasy
    merged = []
    for l, r, rel in bounds:
        if merged and merged[-1][2] == rel:
            merged[-1][1] = r
        else:
            merged.append([l, r, rel])

    ZONE = {
        "unreliable": ("#d62728", 0.10, f"MIA niewiarygodne ($|D_f|<{MIA_UNRELIABLE}$)"),
        "limited":    ("#ff7f0e", 0.07, f"MIA ograniczone ($|D_f|<{MIA_LIMITED}$)"),
    }
    added = set()
    legend_handles = []
    for l, r, rel in merged:
        if rel not in ZONE:
            continue
        color, alpha, label = ZONE[rel]
        ax.axvspan(l, r, alpha=alpha, color=color, zorder=0, lw=0)
        if rel not in added:
            legend_handles.append(
                mpatches.Patch(facecolor=color, alpha=alpha + 0.15, label=label)
            )
            added.add(rel)
    return legend_handles



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
            base = naive_time.get(r["forget_fraction"])
            speedup = base / t if base and t else None
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
            base = naive_time.get(fc)
            speedup = base / t if base and t else None
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


def _mia_fig_legend(fig, handles, ncol=4):
    """Umieszcza wspólną legendę pod panelami wykresu MIA."""
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=ncol, fontsize=9, frameon=True,
               borderaxespad=0.5, columnspacing=1.2)
    plt.subplots_adjust(bottom=0.20)


def plot_mia_dumbbell(sample_runs_c10, sample_runs_c100, mode="po_przed"):
    """Dumbbell MIA — pionowe segmenty z kropkami, osobna figura per dataset.

    mode='po_50'    → segment: ideał (50) → MIA_after
    mode='po_przed' → segment: MIA_before → MIA_after
    """
    if mode == "po_50":
        suptitle = "MIA po oduczeniu vs ideał (50%)"
        fig_base = "fig_mia_dumbbell_po50"
    else:
        suptitle = "MIA przed i po oduczeniu"
        fig_base = "fig_mia_dumbbell_poprzed"

    mia_configs = [
        ("b_mia_l", "a_mia_l", "MIA-Loss"),
        ("b_mia_e", "a_mia_e", "MIA-Entropy"),
    ]
    datasets = [
        ("CIFAR-10",  sample_runs_c10,  f"{fig_base}_cifar10"),
        ("CIFAR-100", sample_runs_c100, f"{fig_base}_cifar100"),
    ]

    jitter = dict(zip(METHODS, [0.80, 1.0, 1.25]))

    for dataset_name, runs, fig_name in datasets:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        rel_h = []

        for col_idx, (bkey, akey, mia_name) in enumerate(mia_configs):
            ax = axes[col_idx]

            for method in METHODS:
                rows  = [r for r in runs if r["method"] == method]
                fracs = sorted(set(r["forget_fraction"] for r in rows))
                color = COLORS[method]
                j     = jitter[method]

                for f in fracs:
                    row = next((r for r in rows if r["forget_fraction"] == f), None)
                    if row is None:
                        continue
                    b_val = row[bkey]
                    a_val = row[akey]
                    x = f * 100 * j

                    if mode == "po_50":
                        lo, hi = (a_val, 50) if a_val < 50 else (50, a_val)
                        ax.plot([x, x], [lo, hi], color=color, lw=1.8, alpha=0.7, zorder=3)
                        ax.scatter([x], [a_val], color=color, s=55, zorder=5)
                        ax.scatter([x], [50],    color=color, s=35, marker="D",
                                   facecolors="none", linewidths=1.5, zorder=4)
                    else:
                        ax.plot([x, x], [b_val, a_val], color=color, lw=1.8, alpha=0.7, zorder=3)
                        ax.scatter([x], [a_val], color=color, s=55, zorder=5)
                        ax.scatter([x], [b_val], color=color, s=35, marker="o",
                                   facecolors="none", linewidths=1.5, zorder=4)

            h = _shade_mia_reliability(ax, runs)
            if not rel_h:
                rel_h = h
            ax.axhline(50, color="black", ls=":", lw=1.8)
            ax.set_xscale("log")
            ax.set_xlabel("Rozmiar $D_f$ [% zbioru treningowego, skala log]", fontsize=10)
            ax.set_ylabel("Dokładność ataku MIA [%]", fontsize=10)
            ax.set_title(mia_name, fontsize=11, fontweight="bold")
            ax.set_ylim([30, 80])
            ax.grid(True, alpha=0.3)

        method_h = [mpatches.Patch(color=COLORS[m], label=METHOD_LABELS[m])
                    for m in METHODS]
        if mode == "po_50":
            style_h = [
                plt.Line2D([0], [0], color="gray", lw=0, marker="o", ms=7,
                           label="Po oduczeniu"),
                plt.Line2D([0], [0], color="gray", lw=0, marker="D", ms=7,
                           mfc="none", mew=1.5, label="Ideał (50%)"),
            ]
        else:
            style_h = [
                plt.Line2D([0], [0], color="gray", lw=0, marker="o", ms=7,
                           label="Po oduczeniu"),
                plt.Line2D([0], [0], color="gray", lw=0, marker="o", ms=7,
                           mfc="none", mew=1.5, label="Przed oduczeniem"),
            ]
        ref_h = [plt.Line2D([0], [0], color="black", lw=1.8, ls=":", label="Ideał (50%)")]
        _mia_fig_legend(fig, method_h + style_h + ref_h + rel_h)
        fig.suptitle(f"{suptitle} — {dataset_name}", fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0.18, 1, 1])
        _save(fig, fig_name)


SPEEDUP_LABELS = {
    "grad_tau": "Grad-Tau",
    "sisa":     "SISA",
}


def _draw_at_af_panel(ax, runs, dataset_name, title=None):
    """Draw A(Dt)/A(Df) lines on ax; returns (method_h, style_h) legend handles."""
    for method in METHODS:
        rows  = sorted([r for r in runs if r["method"] == method],
                       key=lambda r: r["forget_fraction"])
        fracs = [r["forget_fraction"] * 100 for r in rows]
        at    = [r["a_test_acc"]   for r in rows]
        af    = [r["a_forget_acc"] for r in rows]
        color = COLORS[method]
        ax.plot(fracs, at, "o-",  color=color, lw=2, ms=7)
        ax.plot(fracs, af, "o--", color=color, lw=2, ms=7, alpha=0.75)

    ax.set_xscale("log")
    ax.set_xlabel("Rozmiar $D_f$ [% zbioru treningowego, skala log]", fontsize=11)
    ax.set_ylabel("Dokładność po oduczeniu [%]", fontsize=11)
    ax.set_title(title if title is not None else dataset_name, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, which="both")

    method_h = [mpatches.Patch(color=COLORS[m], label=METHOD_LABELS[m]) for m in METHODS]
    style_h  = [
        plt.Line2D([0], [0], color="gray", lw=2, ls="-",  label="$A(D_t)$ — test"),
        plt.Line2D([0], [0], color="gray", lw=2, ls="--", label="$A(D_f)$ — zapomnienie"),
    ]
    return method_h, style_h


def plot_at_af_sample(sample_runs_c10, sample_runs_c100):
    """A(Dt) i A(Df) po oduczeniu vs frakcja zapomnienia.

    Generuje osobną figurę per dataset oraz łączoną figurę dwupanelową.
    Linia ciągła = A(Dt), linia przerywana = A(Df), kolor per metoda.
    """
    # per-dataset figures
    for dataset_name, runs, suffix in [
        ("CIFAR-10",  sample_runs_c10,  "cifar10"),
        ("CIFAR-100", sample_runs_c100, "cifar100"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        full_title = f"$A(D_t)$ i $A(D_f)$ po oduczeniu próbkowym — {dataset_name}"
        method_h, style_h = _draw_at_af_panel(ax, runs, dataset_name, title=full_title)
        ax.legend(handles=method_h + style_h, fontsize=10, loc="best")
        plt.tight_layout()
        _save(fig, f"fig_at_af_sample_{suffix}")

    # combined two-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    method_h, style_h = _draw_at_af_panel(ax1, sample_runs_c10,  "CIFAR-10")
    _draw_at_af_panel(ax2, sample_runs_c100, "CIFAR-100")
    fig.suptitle("$A(D_t)$ i $A(D_f)$ po oduczeniu próbkowym",
                 fontsize=13, fontweight="bold")
    fig.legend(handles=method_h + style_h, loc="lower center",
               ncol=len(method_h) + len(style_h), fontsize=10,
               bbox_to_anchor=(0.5, 0.0), frameon=True)
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    _save(fig, "fig_at_af_sample")


def plot_speedup(sample_runs_c10, sample_runs_c100):
    """Przyspieszenie (log-log) vs frakcja zapomnienia — osobno per dataset."""
    for dataset_name, runs, suffix in [
        ("CIFAR-10",  sample_runs_c10,  "cifar10"),
        ("CIFAR-100", sample_runs_c100, "cifar100"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 7))

        ax.axhline(1.0, color="gray", ls="--", lw=2,
                   label="Naiwne retrenowanie 1×")

        for method in ["grad_tau", "sisa"]:
            rows = [r for r in runs if r["method"] == method]
            fracs = sorted(set(r["forget_fraction"] for r in rows))
            speedups = [next((r["speedup"] for r in rows if r["forget_fraction"] == f), None)
                        for f in fracs]
            ax.plot(fracs, speedups, "o-", label=SPEEDUP_LABELS[method],
                    color=COLORS[method], lw=2, ms=8)
            for f, s in zip(fracs, speedups):
                if s is not None:
                    ax.text(f, s * 1.10, f"{s:.1f}×", ha="center", fontsize=10)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Rozmiar $D_f$ [% zbioru treningowego, skala log]", fontsize=12)
        ax.set_ylabel("Przyspieszenie (skala log)", fontsize=12)
        ax.set_title(f"Przyspieszenie oduczania — {dataset_name}",
                     fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=11, loc="best")

        plt.tight_layout()
        _save(fig, f"fig_speedup_{suffix}")


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

    fig.suptitle(f"Oduczanie klasowe — {dataset_name} ",
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

    
    method_handles = [mpatches.Patch(color=COLORS[m], label=METHOD_LABELS[m])
                      for m in METHODS]
    style_handles = [
        mpatches.Patch(facecolor="gray", alpha=0.40, hatch="///",
                       edgecolor="gray", label="Przed oduczeniem (kreskowanie)"),
        mpatches.Patch(facecolor="gray", alpha=0.90,
                       edgecolor="none", label="Po oduczeniu (pełny)"),
    ]
    if ideal_line is not None:
        style_handles.append(
            plt.Line2D([0], [0], color="red", ls="--", lw=1.5, label="Ideał (0%)")
        )
    return method_handles + style_handles


def plot_ba_class(class_runs, dataset_name, filename_base):
    """Przed i po oduczeniu: $A(D_f)$ i $A(D_t)$ per zapomniana klasa — osobne figury."""
    classes = sorted(set(r["forget_class"] for r in class_runs))
    group_labels = [f"Klasa {c}" for c in classes]

    # --- Df ---
    fig_f, ax_f = plt.subplots(figsize=(9, 6))
    handles = _plot_before_after(ax_f, classes, group_labels, class_runs,
                                 "b_forget_acc", "a_forget_acc",
                                 "Dokładność na $D_f$ [%]",
                                 ideal_line=0.0, lookup_key="forget_class",
                                 title=f"Oduczanie klasowe: $A(D_f)$ przed i po — {dataset_name}")
    fig_f.legend(handles=handles, loc="lower center", ncol=len(handles),
                 fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig_f, f"{filename_base}_df")

    # --- Dt ---
    fig_t, ax_t = plt.subplots(figsize=(9, 6))
    handles = _plot_before_after(ax_t, classes, group_labels, class_runs,
                                 "b_test_acc", "a_test_acc",
                                 "Dokładność na $D_t$ [%]",
                                 ideal_line=None, lookup_key="forget_class",
                                 title=f"Oduczanie klasowe: $A(D_t)$ przed i po — {dataset_name}")
    fig_t.legend(handles=handles, loc="lower center", ncol=len(handles),
                 fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig_t, f"{filename_base}_dt")


def plot_ba_sample(sample_runs, dataset_name, filename_base):
    """Przed i po oduczeniu: $A(D_f)$ i $A(D_t)$ per frakcja zapomnienia — osobne figury."""
    fracs = sorted(set(r["forget_fraction"] for r in sample_runs))
    group_labels = [f"{f*100:.2g}%" for f in fracs]
    xlabel = "Frakcja zapomnienia $|D_f|$ / $|D_{train}|$"

    # --- Df ---
    fig_f, ax_f = plt.subplots(figsize=(9, 6))
    handles = _plot_before_after(ax_f, fracs, group_labels, sample_runs,
                                 "b_forget_acc", "a_forget_acc",
                                 "Dokładność na $D_f$ [%]",
                                 ideal_line=None, lookup_key="forget_fraction",
                                 title=f"Oduczanie próbkowe: $A(D_f)$ przed i po — {dataset_name}")
    ax_f.set_xlabel(xlabel, fontsize=11)
    fig_f.legend(handles=handles, loc="lower center", ncol=len(handles),
                 fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig_f, f"{filename_base}_df")

    # --- Dt ---
    fig_t, ax_t = plt.subplots(figsize=(9, 6))
    handles = _plot_before_after(ax_t, fracs, group_labels, sample_runs,
                                 "b_test_acc", "a_test_acc",
                                 "Dokładność na $D_t$ [%]",
                                 ideal_line=None, lookup_key="forget_fraction",
                                 title=f"Oduczanie próbkowe: $A(D_t)$ przed i po — {dataset_name}")
    ax_t.set_xlabel(xlabel, fontsize=11)
    fig_t.legend(handles=handles, loc="lower center", ncol=len(handles),
                 fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig_t, f"{filename_base}_dt")



def plot_per_class_heatmap(class_runs, dataset_name, filename_base,
                           max_rows: int = 20):
    """Heatmapa zmiany $A(D_t)$ per klasa: po − przed oduczeniem.

    Dla dużych datasetów (>max_rows klas) pokazuje tylko zapomniane klasy
    + top-(max_rows - n_forgotten) wierszy wg maksymalnej zmiany.
    """
    classes_forgotten = sorted(set(r["forget_class"] for r in class_runs))
    sample_run = next((r for r in class_runs if r["b_per_class_acc_test"]), None)
    if sample_run is None:
        print(f"  (pomijam {filename_base}: brak per_class_acc_test)")
        return

    n_model_cls = len(sample_run["b_per_class_acc_test"])
    n_forgotten  = len(classes_forgotten)

    # Zbuduj pełne macierze dla każdej metody
    all_mats = {}
    for method in METHODS:
        mat = np.full((n_model_cls, n_forgotten), np.nan)
        for col, fc in enumerate(classes_forgotten):
            run = next((r for r in class_runs
                        if r["method"] == method and r["forget_class"] == fc), None)
            if run and run["b_per_class_acc_test"] and run["a_per_class_acc_test"]:
                mat[:, col] = (np.array(run["a_per_class_acc_test"])
                               - np.array(run["b_per_class_acc_test"]))
        all_mats[method] = mat

    # Wybór wierszy do wyświetlenia
    if n_model_cls > max_rows:
        # Maksymalna zmiana w wierszu (po wszystkich metodach i zapomnianych klasach)
        stacked = np.nanmax(np.abs(np.stack(list(all_mats.values()))), axis=0)
        max_per_row = np.nanmax(stacked, axis=1)
        forced = set(classes_forgotten)
        remaining = [i for i in range(n_model_cls) if i not in forced]
        n_extra = max(0, max_rows - len(forced))
        top_extra = sorted(remaining, key=lambda i: max_per_row[i], reverse=True)[:n_extra]
        rows_shown = sorted(forced | set(top_extra))
        filter_note = (f"pokazano {len(rows_shown)}/{n_model_cls} klas "
                       f"(zapomniane + {n_extra} najsilniej dotkniętych)")
    else:
        rows_shown = list(range(n_model_cls))
        filter_note = None

    n_shown = len(rows_shown)
    fig, axes = plt.subplots(
        1, 3, figsize=(6 * 3, max(6, n_shown * 0.40 + 2))
    )
    vmax = 15.0

    for ax_idx, method in enumerate(METHODS):
        ax = axes[ax_idx]
        mat = all_mats[method][rows_shown, :]

        im = ax.imshow(mat, aspect="auto", cmap="RdBu",
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(n_forgotten))
        ax.set_xticklabels([f"Zapom. {fc}" for fc in classes_forgotten], fontsize=9)
        ax.set_yticks(range(n_shown))
        ax.set_yticklabels([f"K{rows_shown[i]}" for i in range(n_shown)], fontsize=7)
        ax.set_xlabel("Zapomniana klasa", fontsize=10)
        ax.set_ylabel("Klasa modelu (zbiór testowy)", fontsize=10)
        ax.set_title(METHOD_LABELS[method], fontsize=11, fontweight="bold")

        for col, fc in enumerate(classes_forgotten):
            if fc in rows_shown:
                row_idx = rows_shown.index(fc)
                ax.add_patch(plt.Rectangle((col - 0.5, row_idx - 0.5), 1, 1,
                                           fill=False, edgecolor="yellow", lw=2))

        plt.colorbar(im, ax=ax, label="Δ dokładność [pp]",
                     fraction=0.046, pad=0.04)

    note_line = f"\n{filter_note}" if filter_note else ""
    fig.suptitle(
        f"Zmiana dokładności testowej per klasa (po − przed) — {dataset_name}\n"
        f"Żółta ramka = zapomniana klasa. Niebieski = wzrost, Czerwony = spadek.{note_line}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, filename_base)




# ---------------------------------------------------------------------------
# Confidence distribution plots (class-wise unlearning)
# ---------------------------------------------------------------------------

def plot_confidence_distributions(class_runs, dataset_name, filename_base, num_classes):
    """
    Two separate 1×3 figures:
      {filename_base}_before  — histogram per method BEFORE unlearning, x=[0.7,1.0]
      {filename_base}_after   — histogram per method AFTER  unlearning, x=[0,1]

    Y-axis = number of forget-set samples.  Splitting before/after lets each
    figure use the full page width in the thesis without crowding.
    """
    random_chance = 1.0 / num_classes

    before_by_method = {m: [] for m in METHODS}
    after_by_method  = {m: [] for m in METHODS}

    for r in class_runs:
        b_conf = r.get("b_forget_conf", [])
        a_conf = r.get("a_forget_conf", [])
        if not b_conf or not a_conf:
            continue
        before_by_method[r["method"]].extend(b_conf)
        after_by_method[r["method"]].extend(a_conf)

    if not any(before_by_method.values()):
        print(f"  (pomijam {filename_base}: brak forget_conf)")
        return

    n_forget_classes = len(set(r["forget_class"] for r in class_runs))
    n_samples = sum(len(v) for v in before_by_method.values()) // len(METHODS)
    subtitle = f"(agregacja {n_forget_classes} klas zapomnienia, N={n_samples:,} próbek)"

    fig_b, axes_b = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for col_idx, method in enumerate(METHODS):
        ax = axes_b[col_idx]
        vals = before_by_method[method]
        if vals:
            ax.hist(vals, bins=15, range=(0.7, 1.0),
                    color=COLORS[method], alpha=0.78, edgecolor="white", lw=0.5)
            mean_v = float(np.mean(vals))
            ax.axvline(mean_v, color="black", ls="--", lw=1.8,
                       label=f"Średnia = {mean_v:.3f}")
        ax.set_xlim(0.7, 1.0)
        ax.set_xlabel("Pewność modelu na praw. etykiecie $D_f$", fontsize=11)
        ax.set_title(METHOD_LABELS[method], fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
    axes_b[0].set_ylabel("Liczba próbek", fontsize=12)
    fig_b.suptitle(
        f"Pewność modelu na $D_f$ PRZED oduczeniem — {dataset_name}\n{subtitle}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig_b, f"{filename_base}_before")

    fig_a, axes_a = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for col_idx, method in enumerate(METHODS):
        ax = axes_a[col_idx]
        vals = after_by_method[method]
        if vals:
            ax.hist(vals, bins=25, range=(0.0, 1.0),
                    color=COLORS[method], alpha=0.78, edgecolor="white", lw=0.5)
            ax.axvline(random_chance, color="black", ls=":", lw=2,
                       label=f"Losowe (1/{num_classes})")
            mean_v = float(np.mean(vals))
            ax.axvline(mean_v, color=COLORS[method], ls="--", lw=1.8,
                       label=f"Średnia = {mean_v:.3f}")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Pewność modelu na praw. etykiecie $D_f$", fontsize=11)
        ax.set_title(METHOD_LABELS[method], fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
    axes_a[0].set_ylabel("Liczba próbek", fontsize=12)
    fig_a.suptitle(
        f"Pewność modelu na $D_f$ PO oduczeniu — {dataset_name}\n{subtitle}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig_a, f"{filename_base}_after")


def plot_confidence_distributions_perclass(class_runs, dataset_name, filename_base,
                                           num_classes):
    """
    Per-forgotten-class histograms after unlearning.

    Layout: rows = methods, cols = forgotten classes.
    Y-axis = number of forget-set samples in each confidence bin (0–1).

    For naive_retrain / SISA the confidence collapses to ~0, so the histogram
    shows a single tall bar on the left — replaced here by a plain annotation
    box to avoid scale distortion while still communicating the result clearly.

    grad_tau row shows the actual histogram, making class-to-class variation
    in forgetting quality directly visible.
    """
    random_chance = 1.0 / num_classes
    forget_classes = sorted(set(r["forget_class"] for r in class_runs))
    n_cols = len(forget_classes)
    n_rows = len(METHODS)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             sharex=False)

    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[row] for row in axes]

    DEGENERATE_STD = 0.02

    for row_idx, method in enumerate(METHODS):
        for col_idx, fc in enumerate(forget_classes):
            ax = axes[row_idx][col_idx]
            run = next((r for r in class_runs
                        if r["method"] == method and r["forget_class"] == fc), None)

            if run is None:
                ax.axis("off")
                continue

            b_conf = np.asarray(run.get("b_forget_conf", []), dtype=float)
            a_conf = np.asarray(run.get("a_forget_conf", []), dtype=float)
            if len(b_conf) == 0 or len(a_conf) == 0:
                ax.axis("off")
                continue

            degenerate = float(a_conf.std()) < DEGENERATE_STD

            if degenerate:
                frac_near_zero = float((a_conf < 0.01).mean()) * 100
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.text(0.5, 0.52,
                        f"Pewność → 0\n\nśr. = {a_conf.mean():.5f}\n"
                        f"{frac_near_zero:.0f}% próbek < 0.01",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.6", fc=COLORS[method],
                                  alpha=0.15, ec=COLORS[method], lw=1.5))
                ax.tick_params(left=False, bottom=False,
                               labelleft=False, labelbottom=False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
            else:
                ax.hist(a_conf, bins=25, range=(0, 1),
                        color=COLORS[method], alpha=0.75,
                        edgecolor="white", lw=0.5)
                ax.axvline(random_chance, color="black", ls=":", lw=1.8,
                           label=f"Losowe (1/{num_classes})")
                mean_a = float(a_conf.mean())
                ax.axvline(mean_a, color=COLORS[method], ls="--", lw=1.8,
                           label=f"Średnia = {mean_a:.3f}")
                ax.set_xlim(0, 1)
                ax.set_ylim(bottom=0)
                ax.legend(fontsize=8, loc="upper right")
                ax.grid(True, alpha=0.25, axis="y")

            if col_idx == 0:
                ax.set_ylabel(f"{METHOD_LABELS[method]}\nLiczba próbek",
                              fontsize=9, fontweight="bold")
            if row_idx == 0:
                ax.set_title(f"Zapomniana klasa {fc}", fontsize=10, fontweight="bold")
            if row_idx == n_rows - 1 and not degenerate:
                ax.set_xlabel("Pewność modelu na praw. etykiecie", fontsize=9)

    fig.suptitle(
        f"Liczba próbek $D_f$ wg pewności modelu po oduczeniu — {dataset_name}\n"
        f"(metody z zapadaniem do 0 pokazane jako adnotacja)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, filename_base)


def plot_mia_classwise(class_runs, dataset_name, filename_base):
    """
    Dumbbell MIA for class-wise unlearning — one figure, two panels (MIA-L / MIA-E).

    X-axis: which class was forgotten (discrete).
    Each method has a small x-jitter.  A segment connects before (hollow circle)
    to after (filled circle).  Reference line at 50% (ideal for sample-wise MIA).

    Key story: after class-wise unlearning, naive_retrain and SISA push MIA
    UP to ~95% — NOT toward 50%.  The model has never seen the forgotten class,
    so its loss is huge on those samples and the MIA classifier trivially separates
    them.  grad_tau on CIFAR-100 does the opposite: MIA drops below 50%.
    This shows MIA is a flawed metric for class-wise unlearning evaluation.
    """
    forget_classes = sorted(set(r["forget_class"] for r in class_runs))
    n_cls = len(forget_classes)
    x_pos = {fc: i for i, fc in enumerate(forget_classes)}

    jitter = {"naive_retrain": -0.18, "grad_tau": 0.0, "sisa": +0.18}
    mia_pairs = [("b_mia_l", "a_mia_l", "MIA-Loss"),
                 ("b_mia_e", "a_mia_e", "MIA-Entropy")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for col_idx, (bkey, akey, mia_name) in enumerate(mia_pairs):
        ax = axes[col_idx]

        for method in METHODS:
            color = COLORS[method]
            j = jitter[method]
            rows = [r for r in class_runs if r["method"] == method]

            for r in rows:
                fc = r["forget_class"]
                x = x_pos[fc] + j
                b_val = r[bkey]
                a_val = r[akey]

                ax.plot([x, x], [b_val, a_val], color=color, lw=2.0, alpha=0.7, zorder=3)
                ax.scatter([x], [b_val], color=color, s=60, zorder=5,
                           facecolors="none", linewidths=1.8)   # hollow = before
                ax.scatter([x], [a_val], color=color, s=60, zorder=5)  # filled = after

        ax.axhline(50, color="black", ls=":", lw=1.8, label="Ideał MIA (50%)")
        ax.set_xticks(range(n_cls))
        ax.set_xticklabels([f"Klasa {fc}" for fc in forget_classes], fontsize=11)
        ax.set_xlim(-0.5, n_cls - 0.5)
        ax.set_ylim(30, 105)
        ax.set_ylabel("Dokładność ataku MIA [%]", fontsize=11)
        ax.set_title(mia_name, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    method_h = [mpatches.Patch(color=COLORS[m], label=METHOD_LABELS[m])
                for m in METHODS]
    style_h = [
        plt.Line2D([0], [0], color="gray", lw=0, marker="o", ms=8,
                   mfc="none", mew=1.8, label="Przed oduczeniem"),
        plt.Line2D([0], [0], color="gray", lw=0, marker="o", ms=8,
                   label="Po oduczeniu"),
        plt.Line2D([0], [0], color="black", lw=1.8, ls=":", label="Ideał (50%)"),
    ]
    fig.legend(handles=method_h + style_h, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=len(method_h + style_h),
               fontsize=10, frameon=True)
    fig.suptitle(f"MIA przed i po oduczeniu klasowym — {dataset_name}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig, filename_base)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerowanie wykresów → {OUT_DIR}/\n")

    sr10  = load_sample_runs("cifar10")
    sr100 = load_sample_runs("cifar100")
    cr10  = load_class_runs("cifar10")
    cr100 = load_class_runs("cifar100")

    # próbkowe — MIA dumbbell (osobno per dataset)
    plot_mia_dumbbell(sr10, sr100, mode="po_50")
    plot_mia_dumbbell(sr10, sr100, mode="po_przed")

    # próbkowe — przyspieszenie, kompromis, czasy, luka A(Dt)-A(Df)
    plot_speedup(sr10, sr100)
    plot_tradeoff(sr10, sr100)
    plot_time(sr10, sr100)
    plot_at_af_sample(sr10, sr100)  # osobno per dataset

    # przed/po — próbkowe (osobno per dataset)
    plot_ba_sample(sr10,  "CIFAR-10",  "fig_ba_sample_cifar10")
    plot_ba_sample(sr100, "CIFAR-100", "fig_ba_sample_cifar100")

    # klasowe — CIFAR-10
    plot_classwise(cr10,  "CIFAR-10",  "fig_classwise_cifar10")
    plot_ba_class(cr10,   "CIFAR-10",  "fig_ba_class_cifar10")
    plot_per_class_heatmap(cr10, "CIFAR-10", "fig_per_class_heatmap_cifar10")

    # klasowe — CIFAR-100
    plot_classwise(cr100, "CIFAR-100", "fig_classwise_cifar100")
    plot_ba_class(cr100,  "CIFAR-100", "fig_ba_class_cifar100")
    plot_per_class_heatmap(cr100, "CIFAR-100", "fig_per_class_heatmap_cifar100")

    # rozkłady pewności — klasowe
    plot_confidence_distributions(cr10,  "CIFAR-10",  "fig_conf_dist_cifar10",  num_classes=10)
    plot_confidence_distributions(cr100, "CIFAR-100", "fig_conf_dist_cifar100", num_classes=100)
    plot_confidence_distributions_perclass(cr10,  "CIFAR-10",  "fig_conf_dist_perclass_cifar10",  num_classes=10)
    plot_confidence_distributions_perclass(cr100, "CIFAR-100", "fig_conf_dist_perclass_cifar100", num_classes=100)

    # MIA klasowe — dumbbell przed/po
    plot_mia_classwise(cr10,  "CIFAR-10",  "fig_mia_classwise_cifar10")
    plot_mia_classwise(cr100, "CIFAR-100", "fig_mia_classwise_cifar100")

    print(f"\n Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
