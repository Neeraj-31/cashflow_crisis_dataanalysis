"""NMC Health figures (refactored from nmc_eda.py; output pixel-identical to the original)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def make_all(raw: pd.DataFrame, r: pd.DataFrame, x: pd.DataFrame, cfg: dict, out: Path) -> None:
    event = cfg["event_marker"]
    az = cfg["altman_z2"]
    yr = raw.index.year
    r_plot = r.copy(); r_plot.index = r_plot.index.year

    def mark(ax):
        ax.axvline(event, color="red", ls="--", lw=1)
        ax.set_xlim(2012.5, 2020.5); ax.set_xticks(range(2013, 2021))

    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    for a, (cols, t) in zip(ax.ravel(), [
            (["ocf_to_cl", "cf_to_debt"], "Cash flow ratios"),
            (["current_ratio", "quick_ratio", "cash_ratio"], "Liquidity"),
            (["dso", "dio", "dpo", "ccc"], "Working-capital days"),
            (["debt_to_equity", "netdebt_ebitda", "interest_cov"], "Leverage / coverage"),
            (["sloan_accruals"], "Sloan accruals (neg = cash > earnings)"),
            (["altman_z2"], "Altman Z''")]):
        for c in cols:
            a.plot(r_plot.index, r_plot[c], marker="o", label=c)
        if t.startswith("Altman"):
            a.axhline(az["safe"], color="green", ls=":"); a.axhline(az["distress"], color="red", ls=":")
        mark(a); a.set_title(t); a.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(out / "01_ratio_panels.png", dpi=140); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].bar(yr - 0.2, raw.revenue / 1e3, 0.4, label="Revenue"); ax[0].bar(yr + 0.2, raw.ebitda / 1e3, 0.4, label="EBITDA")
    ax[0].bar(yr + 0.2, raw.ocf_ytd / 1e3, 0.4, color="k", alpha=.5, label="OCF"); ax[0].set_title("US$m (revenue, EBITDA, OCF)"); ax[0].legend()
    ax[1].plot(yr, x.rev_growth * 100, marker="o", label="Revenue"); ax[1].plot(yr, x.recv_growth * 100, marker="o", label="Receivables")
    ax[1].plot(yr, x.debt_growth * 100, marker="o", label="Total debt"); ax[1].plot(yr, x.ebitda_growth * 100, marker="o", label="EBITDA")
    ax[1].set_xticks(list(yr)); ax[1].set_title("YoY growth (%)"); ax[1].legend(fontsize=8)
    ax[2].plot(yr, x.liquid_to_st_debt, marker="o", label="(cash+deposits)/ST debt"); ax[2].axhline(1, color="red", ls=":")
    ax[2].plot(yr, x.ocf_to_ebitda, marker="o", label="OCF/EBITDA"); ax[2].set_title("Short-term liquidity cover"); ax[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out / "02_growth_redflags.png", dpi=140); plt.close()

    c = r.corr(); fig, a = plt.subplots(figsize=(9, 8)); im = a.imshow(c, cmap="coolwarm", vmin=-1, vmax=1)
    a.set_xticks(range(len(c))); a.set_xticklabels(c.columns, rotation=90); a.set_yticks(range(len(c))); a.set_yticklabels(c.columns)
    plt.colorbar(im); a.set_title(f"NMC ratio correlations (n={len(r)})"); plt.tight_layout(); plt.savefig(out / "03_corr.png", dpi=140); plt.close()
