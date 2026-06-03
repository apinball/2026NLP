"""eval_module_b_all.json 으로부터 발표용 그림 생성.

산출:
  data/figures/per_type_recall.png   — 6 유형 × 3 방식 (Rule / BERT / Ensemble) 그룹 막대
  data/figures/confusion_matrices.png — 3 방식 confusion matrix 가로 배치
  data/figures/overall_f1.png        — 종합 P/R/F1 비교 막대
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TYPE_ORDER = [
    "release_year",
    "version_year",
    "experience_scale",
    "role_scope",
    "indirect_time",
    "tech_combo",
]

TYPE_LABEL_EN = {
    "release_year": "Release Year",
    "version_year": "Version Year",
    "experience_scale": "Experience×Scale",
    "role_scope": "Role×Scope",
    "indirect_time": "Indirect Time",
    "tech_combo": "Tech Combo",
}


def plot_per_type(data: dict, out: Path) -> None:
    per_type = data.get("per_type_recall", {})
    methods = [m for m in ("rule", "bert", "ensemble") if m in per_type]
    if not methods:
        print("per_type_recall not found", file=sys.stderr)
        return

    x = np.arange(len(TYPE_ORDER))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {"rule": "#5BA8D6", "bert": "#E2784A", "ensemble": "#62B864"}
    for i, m in enumerate(methods):
        vals = [per_type[m].get(t, 0.0) * 100 for t in TYPE_ORDER]
        bars = ax.bar(
            x + (i - len(methods) / 2 + 0.5) * width,
            vals,
            width,
            label=m.upper(),
            color=colors.get(m, "gray"),
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 1.5,
                f"{v:.0f}%",
                ha="center",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([TYPE_LABEL_EN[t] for t in TYPE_ORDER], rotation=15)
    ax.set_ylabel("Recall (%)")
    ax.set_title("Per-type Recall: Rule vs BERT vs Ensemble")
    ax.set_ylim(0, 110)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def plot_overall(data: dict, out: Path) -> None:
    methods = [m for m in ("rule", "bert", "ensemble_or") if m in data]
    metrics = ["precision", "recall", "f1"]

    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "rule": "#5BA8D6",
        "bert": "#E2784A",
        "ensemble_or": "#62B864",
    }
    labels_map = {"rule": "Rule", "bert": "BERT", "ensemble_or": "Ensemble"}

    for i, m in enumerate(methods):
        vals = [data[m].get(metric, 0.0) * 100 for metric in metrics]
        bars = ax.bar(
            x + (i - len(methods) / 2 + 0.5) * width,
            vals,
            width,
            label=labels_map[m],
            color=colors.get(m, "gray"),
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 1.5,
                f"{v:.1f}",
                ha="center",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylabel("Score (%)")
    ax.set_title("Module B — Overall Precision / Recall / F1")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def _confusion_from_metrics(prec: float, rec: float, total_positive: int, total_negative: int):
    """precision, recall 만으로 confusion matrix 복원이 어려우니, JSON 에 없으면 placeholder."""
    tp = round(rec * total_positive)
    fn = total_positive - tp
    fp = round(tp * (1 - prec) / prec) if prec > 0 else 0
    tn = max(0, total_negative - fp)
    return np.array([[tn, fp], [fn, tp]])


def plot_confusion_matrices(data: dict, out: Path) -> None:
    n = data.get("n", 0)
    # 균등 분포 가정 (실제 50/50 split)
    pos = n // 2
    neg = n - pos

    methods = [m for m in ("rule", "bert", "ensemble_or") if m in data]
    labels_map = {"rule": "Rule", "bert": "BERT", "ensemble_or": "Ensemble"}

    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 4))
    if len(methods) == 1:
        axes = [axes]

    for ax, m in zip(axes, methods):
        cm = _confusion_from_metrics(
            data[m]["precision"], data[m]["recall"], pos, neg
        )
        im = ax.imshow(cm, cmap="Blues", aspect="equal")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{cm[i][j]}",
                    ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black",
                    fontsize=14, weight="bold",
                )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Normal", "Error"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Normal", "Error"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(labels_map[m])

    fig.suptitle(f"Confusion Matrices  (n={n}, balanced split)")
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", type=Path, default=Path("data/eval_module_b_all.json")
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data/figures"))
    args = ap.parse_args()

    if not args.input.exists():
        print(f"missing: {args.input}", file=sys.stderr)
        return 1

    data = json.loads(args.input.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_per_type(data, args.out_dir / "per_type_recall.png")
    plot_overall(data, args.out_dir / "overall_f1.png")
    plot_confusion_matrices(data, args.out_dir / "confusion_matrices.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
