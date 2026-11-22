"""
Jyoti Structures - detailed EDA (consolidated, INR crore, FY2013-FY2026).  Run: python jyoti_eda_full.py
Outputs: fig1..fig5 PNGs, jyoti_metrics_full.csv, jyoti_flags.csv
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

d = pd.read_csv("jyoti_full.csv", index_col=0)
C = dict(navy="#1d3557", blue="#457b9d", sky="#a8dadc", coral="#e63946", orange="#f4a261", amber="#e9c46a",
         teal="#2a9d8f", slate="#6c757d", green="#52b788", purple="#7b2cbf", ink="#212529")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": .25, "axes.titleweight": "bold", "axes.titlesize": 11.5, "axes.titlelocation": "left",
                     "axes.edgecolor": "#adb5bd", "figure.facecolor": "white", "legend.frameon": False, "font.family": "DejaVu Sans"})
YRS = list(d.index); XT = [y for y in YRS]; XL = [f"FY{str(y)[2:]}" for y in YRS]
PH = [(2012.5, 2013.5, "", "#f8f9fa"), (2013.5, 2016.5, "Cash crunch", "#ffe5e7"), (2016.5, 2017.5, "", "#e9ecef"),
      (2017.5, 2021.5, "Insolvency years", "#dee2e6"), (2021.5, 2026.5, "Post-resolution", "#d8f3dc")]

def shade(ax, labels=False, lo=2012.5, hi=2026.5):
    for a, b, t, c in PH:
        if b < lo or a > hi: continue
        ax.axvspan(max(a, lo), min(b, hi), color=c, alpha=.75, lw=0, zorder=0)
        if labels and t: ax.text((max(a, lo) + min(b, hi)) / 2, 1.02, t, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9, color=C["slate"], weight="bold")
    if lo <= 2017 <= hi: ax.text(2017, 0.04, "FY17\nn/a", transform=ax.get_xaxis_transform(), ha="center", fontsize=7.5, color=C["slate"])
def xax(ax, lo=2012.5, hi=2026.5):
    ax.set_xlim(lo, hi); ax.set_xticks([y for y in YRS if lo <= y <= hi]); ax.set_xticklabels([f"FY{str(y)[2:]}" for y in YRS if lo <= y <= hi], fontsize=8.5)
def lab(ax, x, y, s, dy=0, dx=0, color=None, ha="center", va=None, fs=8.5):
    ax.annotate(s, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va or ("bottom" if dy >= 0 else "top"), fontsize=fs, color=color or C["ink"], weight="bold", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=.72))

# ------------------------------------------------------------------ metrics
m = pd.DataFrame(index=d.index)
sales = d.sales.where(d.sales > 0)
m["sales_growth"] = d.sales.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
m["op_margin"] = d.op_profit / sales
m["net_margin"] = d.net_profit / sales
m["ebitda"] = d.op_profit; m["cfo"] = d.cfo
m["ebitda_minus_cfo"] = d.op_profit - d.cfo
m["gap_to_sales"] = m.ebitda_minus_cfo / sales
m["cfo_to_ebitda"] = d.cfo / d.op_profit.where(d.op_profit > 0)
m["cfo_to_sales"] = d.cfo / sales
m["cfo_to_borrowings"] = d.cfo / d.borrowings
m["ebitda_interest_cover"] = d.op_profit / d.interest.where(d.interest > 0)
m["interest_to_sales"] = d.interest / sales
m["interest_growth"] = d.interest.pct_change(fill_method=None)
m["avg_borrowings"] = (d.borrowings + d.borrowings.shift(1)) / 2
m["implied_interest_rate"] = d.interest / m.avg_borrowings
m["net_worth"] = d.equity_capital + d.reserves
m["debt_to_net_worth"] = (d.borrowings / m.net_worth).where(m.net_worth > 0)
m["borrowings_to_assets"] = d.borrowings / d.total_assets
m["borrowings_growth"] = d.borrowings.pct_change(fill_method=None)
m["other_liab_to_assets"] = d.other_liabilities / d.total_assets
m["other_assets_to_sales"] = d.other_assets / sales
m["fcf"] = d.cfo + d.cfi
m["external_funding_share_of_burn"] = d.cff / (-m.fcf).where(m.fcf < 0)
m["sloan_accruals"] = (d.net_profit - d.cfo) / d.total_assets
for c in ["debtor_days", "inventory_days", "payable_days", "ccc_days"]: m[c] = d[c]
m.round(3).to_csv("jyoti_metrics_full.csv")

# ------------------------------------------------------------------ warning flags
def fl(cond, valid): return pd.Series(np.where(valid, cond.astype(float), np.nan), index=d.index)
F = pd.DataFrame({
    "Operating cash flow < 0": fl(d.cfo < 0, d.cfo.notna()),
    "Cash conversion < 50% of EBITDA": fl(d.cfo < 0.5 * d.op_profit.clip(lower=0), d.cfo.notna() & d.op_profit.notna()),
    "EBITDA / interest < 1.2": fl((d.op_profit / d.interest.where(d.interest > 0)).fillna(9) < 1.2, d.op_profit.notna()),
    "Debtor days > 270": fl(d.debtor_days > 270, d.debtor_days.notna()),
    "Cash conversion cycle > 180 days": fl(d.ccc_days > 180, d.ccc_days.notna()),
    "Borrowings growth > 50%": fl(m.borrowings_growth > .5, m.borrowings_growth.notna()),
    "Sales fall > 10%": fl(m.sales_growth < -.10, m.sales_growth.notna()),
    "Net worth < 0": fl(m.net_worth < 0, m.net_worth.notna()),
})
F.to_csv("jyoti_flags.csv")

# ================================================================== FIG 1: lifecycle
fig, ax = plt.subplots(3, 1, figsize=(14, 12.5), sharex=True, gridspec_kw={"hspace": .22})
a = ax[0]; shade(a, labels=True)
a.plot(YRS, d.op_profit, "-o", color=C["orange"], lw=2, label="EBITDA (operating profit)")
a.plot(YRS, d.net_profit, "--s", color=C["slate"], lw=1.5, ms=4, label="Net profit")
a.plot(YRS, d.cfo, "-o", color=C["coral"], lw=2.6, label="Cash from operations")
a.axhline(0, color=C["ink"], lw=.8); a.set_yscale("symlog", linthresh=200); a.set_ylabel("INR crore (symlog scale)")
a.set_title("1. Earnings vs cash: the company bled cash while EBITDA was still positive", pad=24); a.legend(loc="upper right", fontsize=9); a.set_ylim(-6000, 6000)
lab(a, 2013, 51, "CFO +51", dy=10, color=C["coral"]); lab(a, 2014, -88, "CFO turns\nnegative", dy=-12, color=C["coral"])
lab(a, 2016, -1764, "-1,764", dy=-12, color=C["coral"]); lab(a, 2018, 1190, "+1,190 (one-off,\nunexplained)", dx=12, dy=-4, ha="left", va="top", color=C["coral"])
lab(a, 2026, -269, "-269", dy=-12, color=C["coral"]); lab(a, 2025, -177, "-177", dy=-12, color=C["coral"]); lab(a, 2014, 311, "EBITDA 311", dy=10, color=C["orange"])
b = ax[1]; shade(b)
nw = m.net_worth
b.bar(YRS, nw, color=[C["teal"] if (v or 0) >= 0 else C["coral"] for v in nw], alpha=.85, label="Net worth (equity + reserves)")
b.plot(YRS, d.borrowings, "-o", color=C["navy"], lw=2.2, label="Borrowings"); b.axhline(0, color=C["ink"], lw=.8)
b.set_ylabel("INR crore"); b.set_title("2. Net worth stayed positive for two years after cash flow turned negative; debt tripled"); b.legend(loc="lower right", ncol=2)
lab(b, 2014, 0, "NW +681", dy=-8, color=C["teal"]); lab(b, 2016, -276, "NW turns\nnegative", dy=-6, color=C["coral"])
lab(b, 2016, 5190, "5,190", dy=6, color=C["navy"]); lab(b, 2021, -11384, "NW -11,384", dy=10, color="white", va="bottom"); lab(b, 2022, 1691, "debt cut\nto 1,691", dy=10, color=C["navy"])
c = ax[2]; shade(c)
c.plot(YRS, d.debtor_days, "-o", color=C["purple"], lw=2, label="Debtor days"); c.plot(YRS, d.ccc_days, "-s", color=C["blue"], lw=2, label="Cash conversion cycle (days)")
c.set_yscale("log"); c.set_ylabel("days (log scale)"); c.set_title("3. Collections broke down: debtor days 285 to 542 before the insolvency years")
c.legend(loc="upper left"); lab(c, 2014, 285, "285", dy=-14, color=C["purple"]); lab(c, 2016, 542, "542", dy=-14, color=C["purple"]); lab(c, 2026, 1085, "1,085", dy=-14, color=C["purple"])
xax(c); fig.suptitle("Jyoti Structures: full lifecycle FY2013-FY2026 (consolidated, INR crore)", fontsize=15, weight="bold", x=.06, ha="left", y=.945)
plt.savefig("fig1_lifecycle.png", dpi=150, bbox_inches="tight"); plt.close()

# ================================================================== FIG 2: pre-failure deep dive
Y = [2013, 2014, 2015, 2016]; X = np.arange(4); L = [f"FY{str(y)[2:]}" for y in Y]
fig, ax = plt.subplots(2, 3, figsize=(16, 9.5)); fig.subplots_adjust(hspace=.42, wspace=.34)
a = ax[0, 0]; a.bar(X - .19, d.op_profit[Y], .38, color=C["orange"], label="EBITDA"); a.bar(X + .19, d.cfo[Y], .38, color=C["coral"], label="Cash from operations"); a.axhline(0, color=C["ink"], lw=.8)
for i, y in enumerate(Y):
    if not np.isnan(d.op_profit[y]): lab(a, i - .19, d.op_profit[y], f"{d.op_profit[y]:,.0f}", dy=4, color=C["orange"])
    lab(a, i + .19, d.cfo[y], f"{d.cfo[y]:,.0f}", dy=4 if d.cfo[y] > 0 else -4, color=C["coral"])
a.set_xticks(X); a.set_xticklabels(L); a.set_title("A. EBITDA vs operating cash flow"); a.legend(fontsize=8.5, loc="lower left")
a.set_xlabel("EBITDA minus CFO gap: FY14 399 | FY15 1,217 | FY16 1,785 (INR cr)", fontsize=8.5, color=C["slate"]); a.set_ylim(-2100, 500)
a = ax[0, 1]; Z = [2014, 2015, 2016]; XZ = np.arange(3)
a.fill_between(XZ, d.payable_days[Z], d.debtor_days[Z], color=C["coral"], alpha=.12, label="collection-payment squeeze")
for col, nm, cc in [("debtor_days", "Debtor days (DSO)", C["purple"]), ("inventory_days", "Inventory days (DIO)", C["amber"]), ("payable_days", "Payable days (DPO)", C["teal"]), ("ccc_days", "Cash conversion cycle", C["navy"])]:
    a.plot(XZ, d[col][Z], "-o", lw=2.4 if col == "ccc_days" else 1.8, color=cc, label=nm)
    lab(a, 2, d[col][2016], f"{d[col][2016]:.0f}", dx=14, color=cc, ha="left", va="center")
a.set_xticks(XZ); a.set_xticklabels(["FY14", "FY15", "FY16"]); a.set_xlim(-.2, 2.5); a.set_xlabel("customers pay slower while suppliers get paid faster", fontsize=8.5, color=C["slate"]); a.set_title("B. Cash conversion cycle (days)"); a.legend(fontsize=8, loc="upper left")
a = ax[0, 2]; src = [3201, -2982, -205, 14]; names = ["Financing\ninflow", "Operating\ncash burn", "Investing", "Net cash\nchange"]
bottoms, run = [], 0
for i, v in enumerate(src[:3]):
    bottoms.append(run if v > 0 else run + v); run += v
a.bar(range(3), [abs(v) for v in src[:3]], bottom=bottoms, color=[C["teal"], C["coral"], C["slate"]], width=.6)
a.bar(3, src[3], color=C["navy"], width=.6)
for i, v in enumerate(src): lab(a, i, (bottoms[i] + abs(v)) if i < 3 else v, f"{v:+,}", dy=4, color=C["ink"])
a.set_xticks(range(4)); a.set_xticklabels(names, fontsize=8.5); a.set_title("C. Sources & uses, FY14-16 (INR cr)")
a.set_xlabel("new financing covered 100% of the operating burn", fontsize=8.5, color=C["slate"]); a.set_ylim(0, 3700)
a = ax[1, 0]; a.bar(XZ, m.ebitda_interest_cover[Z], .5, color=[C["amber"], C["orange"], C["coral"]]); a.axhline(1, color=C["coral"], ls="--", lw=1.4); a.text(-.42, 1.03, "cover = 1.0x", color=C["coral"], fontsize=8.5, ha="left")
for i, y in enumerate(Z): lab(a, i, m.ebitda_interest_cover[y], f"{m.ebitda_interest_cover[y]:.2f}x", dy=4)
a2 = a.twinx(); a2.plot(XZ, m.interest_to_sales[Z] * 100, "-o", color=C["navy"], lw=2); a2.set_ylabel("interest / sales (%)", color=C["navy"]); a2.grid(False); a2.set_ylim(0, 30)
for i, y in enumerate(Z): lab(a2, i, m.interest_to_sales[y] * 100, f"{m.interest_to_sales[y]*100:.0f}%", dy=7, color=C["navy"])
a.set_xticks(XZ); a.set_xticklabels(["FY14", "FY15", "FY16"]); a.set_ylim(0, 1.25); a.set_xlabel("EBITDA / interest never reached 1.0x (bars); interest/sales rising (line)", fontsize=8.5, color=C["slate"]); a.set_title("D. Interest burden")
a = ax[1, 1]; a.bar(XZ, d.borrowings[Z], .5, color=C["navy"], alpha=.9)
for i, y in enumerate(Z): lab(a, i, d.borrowings[y], f"{d.borrowings[y]:,.0f}" + (f"\n(+{m.borrowings_growth[y]*100:.0f}%)" if y > 2014 else ""), dy=4)
a2 = a.twinx(); a2.plot(XZ[1:], m.implied_interest_rate[[2015, 2016]] * 100, "-o", color=C["coral"], lw=2); a2.set_ylabel("implied interest rate (%)", color=C["coral"]); a2.grid(False); a2.set_ylim(0, 27)
for i, y in [(1, 2015), (2, 2016)]: lab(a2, i, m.implied_interest_rate[y] * 100, f"{m.implied_interest_rate[y]*100:.0f}%", dy=7, color=C["coral"])
a.set_xticks(XZ); a.set_xticklabels(["FY14", "FY15", "FY16"]); a.set_ylim(0, 6400); a.set_title("E. Debt build-up & cost of debt")
a = ax[1, 2]; a.bar(XZ - .19, d.other_assets[Z] / 1000, .38, color=C["orange"], label="Other assets (mostly receivables, WIP) ,000 cr"); a.bar(XZ + .19, d.sales[Z] / 1000, .38, color=C["blue"], label="Sales ,000 cr")
a2 = a.twinx(); a2.plot(XZ, m.other_assets_to_sales[Z], "-o", color=C["coral"], lw=2.2); a2.set_ylim(0, 2.6); a2.grid(False); a2.set_ylabel("other assets / sales (x)", color=C["coral"])
for i, y in enumerate(Z): lab(a2, i, m.other_assets_to_sales[y], f"{m.other_assets_to_sales[y]:.2f}x", dy=7, color=C["coral"])
a.set_xticks(XZ); a.set_xticklabels(["FY14", "FY15", "FY16"]); a.set_ylim(0, 8.5); a.legend(fontsize=7.5, loc="upper left"); a.set_xlabel("working capital grows while sales shrink", fontsize=8.5, color=C["slate"]); a.set_title("F. Asset bloat")
fig.suptitle("Jyoti Structures: anatomy of the cash crunch (FY2013-FY2016)", fontsize=15, weight="bold", x=.06, ha="left", y=.97)
plt.savefig("fig2_crunch_anatomy.png", dpi=150, bbox_inches="tight"); plt.close()

# ================================================================== FIG 3: balance sheet structure
B = [y for y in YRS if y >= 2014]
fig, ax = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"hspace": .15})
a = ax[0]; shade(a, labels=True, lo=2013.5)
a.bar(B, d.borrowings[B], .7, color=C["navy"], label="Borrowings"); a.bar(B, d.other_liabilities[B], .7, bottom=d.borrowings[B], color=C["sky"], label="Other liabilities (payables, provisions, interest accrued)")
a.plot(B, m.net_worth[B], "D-", color=C["coral"], lw=2, ms=5, label="Net worth (can be negative)"); a.axhline(0, color=C["ink"], lw=.8)
a.set_ylabel("INR crore"); a.set_title("Liabilities side: debt first, then unpaid obligations, while net worth collapses", pad=24); a.legend(loc="lower right", fontsize=8.5)
for y in [2014, 2016, 2021, 2026]: lab(a, y, d.borrowings[y] + d.other_liabilities[y], f"{d.borrowings[y]+d.other_liabilities[y]:,.0f}", dy=4)
b = ax[1]; shade(b, lo=2013.5)
b.bar(B, d.noncurrent_assets[B], .7, color=C["slate"], label="Fixed assets + CWIP + investments"); b.bar(B, d.other_assets[B], .7, bottom=d.noncurrent_assets[B], color=C["orange"], label="Other assets (receivables, WIP, cash)")
b2 = b.twinx(); b2.plot(B, m.other_assets_to_sales[B].clip(upper=40), "o--", color=C["purple"], lw=1.6, ms=4, label="Other assets / sales (clipped at 40x)"); b2.set_yscale("log"); b2.grid(False); b2.set_ylabel("x sales (log)", color=C["purple"])
b.set_ylabel("INR crore"); b.set_title("Asset side: the balance sheet was mostly stuck working capital, not plant"); h1, l1 = b.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels(); b.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.5)
xax(b, lo=2013.5); plt.savefig("fig3_balance_sheet.png", dpi=150, bbox_inches="tight"); plt.close()

# ================================================================== FIG 4: warning flag heatmap
fig, ax = plt.subplots(2, 1, figsize=(14, 7.5), gridspec_kw={"height_ratios": [1, 5.2], "hspace": .08}, sharex=True)
cnt = F.sum(axis=1, min_count=1)
ax[0].bar(YRS, cnt, .78, color=[C["coral"] if (v or 0) >= 5 else C["orange"] if (v or 0) >= 3 else C["amber"] if (v or 0) >= 1 else C["green"] for v in cnt.fillna(0)])
for y in YRS:
    if not np.isnan(cnt[y]): lab(ax[0], y, cnt[y], f"{cnt[y]:.0f}", dy=2, fs=8.5)
ax[0].set_ylim(0, 9.5); ax[0].set_yticks([]); ax[0].grid(False); ax[0].spines["left"].set_visible(False); ax[0].set_title("Number of warning flags tripped (out of 8)")
cm = ListedColormap(["#b7e4c7", "#e63946"]); M = np.ma.masked_invalid(F.T.values)
ax[1].set_facecolor("#e9ecef"); ax[1].imshow(M, cmap=cm, aspect="auto", extent=[2012.5, 2026.5, len(F.columns), 0], vmin=0, vmax=1)
ax[1].set_yticks(np.arange(len(F.columns)) + .5); ax[1].set_yticklabels(F.columns, fontsize=9.5); ax[1].grid(False)
ax[1].set_xticks(YRS); ax[1].set_xticklabels(XL, fontsize=8.5)
for i in range(len(F.columns) + 1): ax[1].axhline(i, color="white", lw=2)
for y in range(2013, 2028): ax[1].axvline(y - .5, color="white", lw=2)
ax[1].legend(handles=[Patch(color="#e63946", label="flag tripped"), Patch(color="#b7e4c7", label="not tripped"), Patch(color="#e9ecef", label="no data")], loc="upper center", bbox_to_anchor=(.5, -.07), ncol=3)
fig.suptitle("Rule-based early-warning flags by year (thresholds chosen illustratively, to be fixed in advance for the real test)", fontsize=13, weight="bold", x=.06, ha="left", y=.97)
plt.savefig("fig4_warning_flags.png", dpi=150, bbox_inches="tight"); plt.close()

# ================================================================== FIG 5: post-resolution recurrence
R = [y for y in YRS if y >= 2022]; XR = np.arange(len(R))
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8)); fig.subplots_adjust(wspace=.28)
a = ax[0]; a.bar(XR - .19, d.net_profit[R], .38, color=C["green"], label="Net profit"); a.bar(XR + .19, d.cfo[R], .38, color=C["coral"], label="Cash from operations"); a.axhline(0, color=C["ink"], lw=.8)
for i, y in enumerate(R): lab(a, i - .19, d.net_profit[y], f"{d.net_profit[y]:.0f}", dy=3 if d.net_profit[y] >= 0 else -3, color=C["green"], fs=8); lab(a, i + .19, d.cfo[y], f"{d.cfo[y]:.0f}", dy=3 if d.cfo[y] >= 0 else -3, color=C["coral"], fs=8)
a.set_xticks(XR); a.set_xticklabels([f"FY{str(y)[2:]}" for y in R]); a.set_title("Profit up, cash flow down again"); a.legend(fontsize=8.5)
a = ax[1]; a.plot(XR, d.debtor_days[R], "-o", color=C["purple"], lw=2.2); a.set_yscale("log"); a.set_xticks(XR); a.set_xticklabels([f"FY{str(y)[2:]}" for y in R]); a.set_title("Debtor days still above 1,000")
for i, y in enumerate(R): lab(a, i, d.debtor_days[y], f"{d.debtor_days[y]:,.0f}", dy=6, color=C["purple"], fs=8)
a = ax[2]; a.bar(XR, d.sales[R], .55, color=C["blue"], alpha=.85, label="Sales"); a2 = a.twinx(); a2.plot(XR, d.borrowings[R], "-s", color=C["navy"], lw=2, label="Borrowings"); a2.grid(False); a2.set_ylim(0, 2600)
a.set_xticks(XR); a.set_xticklabels([f"FY{str(y)[2:]}" for y in R]); a.set_title("Fast sales growth, debt creeping up")
for i, y in enumerate(R): lab(a, i, d.sales[y], f"{d.sales[y]:,.0f}", dy=3, fs=8, color=C["blue"])
fig.suptitle("After resolution: the same cash-conversion signature is back (FY2022-FY2026)", fontsize=14, weight="bold", x=.06, ha="left", y=1.03)
plt.savefig("fig5_recurrence.png", dpi=150, bbox_inches="tight"); plt.close()

# ------------------------------------------------------------------ printed tables
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print("=== key metrics, pre-failure (FY2013-16) ===")
print(m.loc[2013:2016, ["op_margin", "ebitda_minus_cfo", "gap_to_sales", "cfo_to_ebitda", "ebitda_interest_cover", "interest_to_sales", "implied_interest_rate", "net_worth", "debt_to_net_worth", "borrowings_to_assets", "borrowings_growth", "external_funding_share_of_burn", "other_assets_to_sales", "sloan_accruals", "debtor_days", "payable_days", "ccc_days"]].round(2).T.to_string())
print("\n=== first year each flag trips ===")
for c in F.columns: print(f"  {c:36s}", next((y for y in YRS if F.loc[y, c] == 1), "-"))
print("\nflags per year:", cnt.dropna().astype(int).to_dict())
print("\n=== recurrence FY2022-26 ===")
print(m.loc[2022:2026, ["sales_growth", "op_margin", "cfo_to_sales", "sloan_accruals", "borrowings_growth", "debtor_days"]].round(2).to_string())
