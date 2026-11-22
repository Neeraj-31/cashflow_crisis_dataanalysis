"""Figures for Layer 1 (event-time panels) and Layer 2 (rule x year heatmap)."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.ratios import ORIENTATION  # noqa: E402
from src.reporting.style import C, EVENT, STYLE, fy_label  # noqa: E402


def layer1_event_time(panel: pd.DataFrame, cols: list[str], firm: str, event_date: str, frac: float,
                      missing_fy: set[int], band: dict | None, out: Path) -> None:
    """panel: index tau, column fy + ratio columns. Event line at tau = -0.5 + frac (inside FY0)."""
    with plt.rc_context(STYLE):
        n = len(cols); nc = 3; nr = math.ceil(n / nc)
        fig, axes = plt.subplots(nr, nc, figsize=(4.6 * nc, 3.0 * nr), sharex=True, squeeze=False)
        taus = panel.index.to_numpy(); ev = -0.5 + frac
        for ax, c in zip(axes.ravel(), cols):
            if band is not None and c in band:
                b = band[c].reindex(panel.index)
                ax.fill_between(taus, b.lo, b.hi, color=C["band"], alpha=.5, lw=0, label="control IQR")
                ax.plot(taus, b["median"], color=C["slate"], ls=":", lw=1.4, label="control median")
            ax.plot(taus, panel[c], "-o", color=C["navy"], lw=2, ms=4.5, label=firm)
            for t, fy in zip(taus, panel.fy):
                if fy in missing_fy and panel[c].notna().any():
                    ax.text(t, .5, "no\ndata", transform=ax.get_xaxis_transform(), ha="center", va="center", fontsize=7.5, color=C["slate"])
            ax.axvline(ev, **EVENT)
            if panel[c].isna().all():
                ax.text(.5, .5, "not available\nfor this firm", transform=ax.transAxes, ha="center", va="center", fontsize=9, color=C["slate"])
                ax.set_yticks([])
            ax.set_title(ORIENTATION.get(c, ("", c, 1))[1] + ("  (negated)" if ORIENTATION.get(c, ("", "", 1))[2] < 0 else ""), fontsize=9.5)
            ax.set_xticks(taus); ax.set_xticklabels([f"{t}\n{fy_label(fy)}" for t, fy in zip(taus, panel.fy)], fontsize=8)
            ax.set_xlim(taus.min() - .4, taus.max() + .4)
        for ax in axes.ravel()[n:]:
            ax.set_visible(False)
        h, l = axes[0, 0].get_legend_handles_labels()
        h.append(plt.Line2D([], [], **EVENT)); l.append(f"event {event_date}")
        fig.suptitle(f"Layer 1 - {firm}: oriented ratios in event time (lower = worse; tau = fiscal years to event)",
                     fontsize=13, weight="bold", x=.01, ha="left", y=.995)
        fig.legend(h, l, loc="upper left", ncol=len(l), bbox_to_anchor=(.005, .965), fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, .93))
        fig.savefig(out); plt.close(fig)


def layer2_heatmap(F: pd.DataFrame, firm: str, efy: int, event_date: str, frac: float,
                   missing_fy: set[int], first: pd.DataFrame, out: Path) -> None:
    """Rules x fiscal years. Columns extended to the event FY so the event line is always visible."""
    with plt.rc_context(STYLE):
        yrs = list(range(int(F.index.min()), max(int(F.index.max()), efy) + 1))
        G = F.reindex(yrs)
        fig, ax = plt.subplots(figsize=(max(7, .62 * len(yrs) + 4), .62 * G.shape[1] + 1.9))
        ax.set_facecolor(C["nodata"]); ax.grid(False)
        ax.imshow(np.ma.masked_invalid(G.T.values.astype(float)), cmap=ListedColormap([C["ok"], C["coral"]]),
                  aspect="auto", vmin=0, vmax=1, extent=[yrs[0] - .5, yrs[-1] + .5, G.shape[1], 0])
        for i in range(G.shape[1] + 1): ax.axhline(i, color="white", lw=2)
        for y in yrs: ax.axvline(y - .5, color="white", lw=2)
        for y in yrs:
            if y in missing_fy or y > int(F.index.max()):
                ax.text(y, G.shape[1] / 2, "no data", rotation=90, ha="center", va="center", fontsize=8, color=C["slate"])
        ev = efy - .5 + frac
        ax.axvline(ev, **EVENT)
        ax.text(ev, -0.15, f"event {event_date}", color=C["coral"], ha="center", va="bottom", fontsize=8.5, weight="bold")
        labels = []
        for r in G.columns:
            fy = first.loc[r, "first_trip_fy"]
            labels.append(f"{r}\nfirst {fy_label(fy)}, lead {first.loc[r, 'lead_years']:.1f}y" if pd.notna(fy) else f"{r}\nno pre-event trip")
        ax.set_yticks(np.arange(G.shape[1]) + .5); ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xticks(yrs); ax.set_xticklabels([fy_label(y) for y in yrs], fontsize=8.5)
        ax.set_xlim(yrs[0] - .5, yrs[-1] + .5)
        ax.legend(handles=[Patch(color=C["coral"], label="tripped"), Patch(color=C["ok"], label="not tripped"),
                           Patch(color=C["nodata"], label="not evaluable"), plt.Line2D([], [], **EVENT, label="event")],
                  loc="upper center", bbox_to_anchor=(.5, -.12), ncol=4)
        ax.set_title(f"Layer 2 - {firm}: pre-registered rule flags by fiscal year", fontsize=12, pad=30)
        fig.savefig(out); plt.close(fig)
