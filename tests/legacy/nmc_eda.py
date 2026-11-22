"""NMC Health - annual EDA (2013-2018). Run: python nmc_eda.py"""
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
OUT="plots_nmc"; os.makedirs(OUT, exist_ok=True)
EVENT=2020.2   # RNS 'Update on financial position' (verify exact date for your slide)

raw=pd.read_csv("nmc_raw.csv",parse_dates=["period_end"]).set_index("period_end")
r=pd.read_csv("nmc_ratios.csv",index_col=0,parse_dates=True)
yr=raw.index.year
r_plot=r.copy(); r_plot.index=r_plot.index.year

# ---- derived red-flag metrics ----
x=pd.DataFrame(index=raw.index)
x["rev_growth"]=raw.revenue.pct_change()
x["recv_growth"]=raw.receivables.pct_change()
x["recv_minus_rev_growth"]=x.recv_growth-x.rev_growth
x["debt_growth"]=raw.total_debt.pct_change()
x["ebitda_growth"]=raw.ebitda.pct_change()
x["ocf_growth"]=raw.ocf_ytd.pct_change()
x["ocf_to_ebitda"]=raw.ocf_ytd/raw.ebitda
x["fcf_to_debt"]=(raw.ocf_ytd-raw.capex_ytd)/raw.total_debt
x["liquid_to_st_debt"]=(raw.cash+raw.bank_deposits)/raw.st_debt
x["st_debt_share"]=raw.st_debt/raw.total_debt
x["goodwill_proxy_equity"]=raw.equity/raw.total_assets
x["net_debt_to_equity"]=(raw.total_debt-raw.cash-raw.bank_deposits)/raw.equity
x.to_csv("nmc_redflags.csv")

pd.set_option("display.width",200)
print("=== MISSING DATA ===\n",raw.isna().sum().sum(),"missing cells | n =",len(raw))
print("\n=== SUMMARY (ratios) ===\n",r.describe().T[["mean","std","min","max"]].round(3))
print("\n=== YoY / RED-FLAG METRICS ===\n",x.round(3).T)
print("\n=== Z-SCORE OF EACH RATIO vs 2013-2015 BASELINE ===")
base=r.loc[:"2015-12-31"]
z=(r-base.mean())/base.std()
print(z[["ocf_to_cl","ocf_margin","current_ratio","ccc","dso","interest_cov","netdebt_ebitda","altman_z2"]].round(2))
print("\n=== TOP CORRELATIONS WITH ocf_to_cl (n=6, indicative only) ===")
print(r.corr()["ocf_to_cl"].drop("ocf_to_cl").sort_values().round(2).to_string())

# ---- plots ----
def mark(ax):
    ax.axvline(EVENT,color="red",ls="--",lw=1)
    ax.set_xlim(2012.5,2020.5); ax.set_xticks(range(2013,2021))

fig,ax=plt.subplots(2,3,figsize=(15,7))
for a,(cols,t) in zip(ax.ravel(),[
  (["ocf_to_cl","cf_to_debt"],"Cash flow ratios"),
  (["current_ratio","quick_ratio","cash_ratio"],"Liquidity"),
  (["dso","dio","dpo","ccc"],"Working-capital days"),
  (["debt_to_equity","netdebt_ebitda","interest_cov"],"Leverage / coverage"),
  (["sloan_accruals"],"Sloan accruals (neg = cash > earnings)"),
  (["altman_z2"],"Altman Z''")]):
    for c in cols: a.plot(r_plot.index,r_plot[c],marker="o",label=c)
    if t.startswith("Altman"):
        a.axhline(2.6,color="green",ls=":"); a.axhline(1.1,color="red",ls=":")
    mark(a); a.set_title(t); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/01_ratio_panels.png",dpi=140); plt.close()

fig,ax=plt.subplots(1,3,figsize=(15,4))
ax[0].bar(yr-0.2,raw.revenue/1e3,0.4,label="Revenue"); ax[0].bar(yr+0.2,raw.ebitda/1e3,0.4,label="EBITDA")
ax[0].bar(yr+0.2,raw.ocf_ytd/1e3,0.4,color="k",alpha=.5,label="OCF"); ax[0].set_title("US$m (revenue, EBITDA, OCF)"); ax[0].legend()
ax[1].plot(yr,x.rev_growth*100,marker="o",label="Revenue"); ax[1].plot(yr,x.recv_growth*100,marker="o",label="Receivables")
ax[1].plot(yr,x.debt_growth*100,marker="o",label="Total debt"); ax[1].plot(yr,x.ebitda_growth*100,marker="o",label="EBITDA")
ax[1].set_xticks(list(yr)); ax[1].set_title("YoY growth (%)"); ax[1].legend(fontsize=8)
ax[2].plot(yr,x.liquid_to_st_debt,marker="o",label="(cash+deposits)/ST debt"); ax[2].axhline(1,color="red",ls=":")
ax[2].plot(yr,x.ocf_to_ebitda,marker="o",label="OCF/EBITDA"); ax[2].set_title("Short-term liquidity cover"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/02_growth_redflags.png",dpi=140); plt.close()

c=r.corr(); fig,a=plt.subplots(figsize=(9,8)); im=a.imshow(c,cmap="coolwarm",vmin=-1,vmax=1)
a.set_xticks(range(len(c))); a.set_xticklabels(c.columns,rotation=90); a.set_yticks(range(len(c))); a.set_yticklabels(c.columns)
plt.colorbar(im); a.set_title("NMC ratio correlations (n=6)"); plt.tight_layout(); plt.savefig(f"{OUT}/03_corr.png",dpi=140); plt.close()
print("\nSaved plots to",OUT,"and nmc_redflags.csv")
