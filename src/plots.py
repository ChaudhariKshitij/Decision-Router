"""Matplotlib figures (headless)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

COLORS = {"fast": "#2a9d8f", "accurate": "#e9c46a", "deep": "#e76f51"}


def curves(y, probs: dict, path_prefix) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for n, p in probs.items():
        pr, rc, _ = precision_recall_curve(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(rc, pr, label=n, color=COLORS.get(n))
        ax[1].plot(fpr, tpr, label=n, color=COLORS.get(n))
    ax[0].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (test)")
    ax[1].set(xlabel="FPR", ylabel="TPR", title="ROC (test)")
    ax[1].plot([0, 1], [0, 1], "k:", lw=0.8)
    for a in ax:
        a.legend(); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{path_prefix}_pr_roc.png", dpi=130); plt.close(fig)


def calibration_plot(tables: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], "k:", label="perfect")
    for n, t in tables.items():
        ax.plot(t["mean_predicted"], t["observed_rate"], "o-", label=n, color=COLORS.get(n))
    ax.set(xlabel="Mean predicted P(fraud)", ylabel="Observed fraud rate", title="Reliability diagram (test, calibrated)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def confusion_plot(cms: dict, path) -> None:
    fig, ax = plt.subplots(1, len(cms), figsize=(4 * len(cms), 3.6))
    for a, (n, (tn, fp, fn, tp)) in zip(np.atleast_1d(ax), cms.items()):
        a.imshow([[tn, fp], [fn, tp]], cmap="Blues")
        for (i, j), v in np.ndenumerate([[tn, fp], [fn, tp]]):
            a.text(j, i, str(v), ha="center", va="center")
        a.set(title=n, xticks=[0, 1], yticks=[0, 1], xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"])
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def pareto_plot(df, xcol: str, xlabel: str, path, sweep=None) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    if sweep is not None:
        for name, g in sweep.groupby("policy"):
            ax.plot(g[xcol], g["pr_auc"], "-", alpha=0.35, label=f"{name} threshold sweep")
    for r in df.itertuples():
        ax.scatter(getattr(r, xcol), r.pr_auc, s=90 if r.on_pareto else 45, marker="*" if r.on_pareto else "o", zorder=3)
        ax.annotate(r.policy, (getattr(r, xcol), r.pr_auc), textcoords="offset points", xytext=(5, 5), fontsize=8)
    pf = df[df.on_pareto].sort_values(xcol)
    ax.step(pf[xcol], pf["pr_auc"], where="post", color="k", lw=1, ls="--", label="Pareto frontier")
    ax.set(xlabel=xlabel, ylabel="PR-AUC (test)", title="Cost / quality trade-off (stars = Pareto-optimal)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
