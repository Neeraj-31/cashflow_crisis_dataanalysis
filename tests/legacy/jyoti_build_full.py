"""Jyoti Structures full-lifecycle dataset (consolidated, INR crore) from Screener.in (vendor-standardised).
FY2017 is NOT published on the Screener page -> left as a gap. FY2013 has cash-flow items only (secondary source,
cross-checked: CFO+CFI+CFF = -21 = Screener's FY2013 net cash flow)."""
import numpy as np, pandas as pd
nan = np.nan
yrs = [2014,2015,2016,2018,2019,2020,2021,2022,2023,2024,2025,2026]
d = pd.DataFrame(dict(fy=yrs,
 sales=[3636,3111,2947,324,182,16,0,4,229,451,498,751],
 expenses=[3325,3024,2926,3479,841,828,254,41,229,434,460,702],
 op_profit=[311,87,21,-3154,-658,-812,-254,-36,0,17,38,48],
 other_income=[28,55,55,3,6,0,0,1,2,12,6,22],
 interest=[319,485,671,1021,1106,1501,1495,0,0,2,0,3],
 depreciation=[40,53,54,29,27,16,10,7,5,7,8,13],
 pbt=[-20,-395,-649,-4201,-1785,-2329,-1759,-42,-4,20,35,55],
 net_profit=[-9,-396,-704,-4201,-1785,-2329,-1759,-42,-4,29,36,56],
 equity_capital=[16,22,22,22,22,22,22,127,127,146,238,239],
 reserves=[665,365,-298,-5584,-7348,-9701,-11384,25,-102,-172,325,236],
 borrowings=[1498,2942,5190,5718,6383,7007,7412,1691,1803,1916,1996,2084],
 other_liabilities=[2347,2251,1467,2634,3387,4351,5367,350,457,459,457,378],
 total_assets=[4527,5579,6381,2790,2444,1678,1417,2193,2286,2350,3015,2937],
 fixed_assets=[469,436,425,90,62,43,33,27,32,33,42,70],
 other_assets=[4051,5136,5922,2700,2382,1635,1384,2166,2253,2315,2970,2863],
 cfo=[-88,-1130,-1764,1190,-9,-4,-2,-100,-36,-19,-177,-269],
 cfi=[-40,-81,-84,-639,0,0,0,0,-11,-7,-13,-23],
 cff=[177,1252,1772,-918,0,1,0,170,-7,34,615,-74],
 debtor_days=[285,399,542,2740,4284,30382,nan,158009,3183,1642,1572,1085],
 inventory_days=[86,81,62,161,303,2629,2745,1284,207,127,174,137],
 payable_days=[285,240,93,1465,2868,25860,38714,4306,429,232,293,138],
 ccc_days=[86,239,511,1436,1719,7150,nan,154987,2961,1537,1454,1085])).set_index("fy")
d.loc[2013, ["cfo","cfi","cff"]] = [51,-80,8]      # secondary source; sums to Screener's FY2013 net cash flow (-21)
d = d.reindex(range(2013, 2027))
chk = d.dropna(subset=["total_assets"])
assert ((chk.equity_capital+chk.reserves+chk.borrowings+chk.other_liabilities-chk.total_assets).abs()<=3).all()
assert ((chk.op_profit+chk.other_income-chk.interest-chk.depreciation-chk.pbt).abs()<=3).all()
assert (chk.debtor_days+chk.inventory_days-chk.payable_days-chk.ccc_days).dropna().abs().le(3).all()
d["noncurrent_assets"] = d.total_assets - d.other_assets   # fixed assets + CWIP + investments
d.to_csv("jyoti_full.csv"); print("saved jyoti_full.csv; balance sheet, P&L and CCC identities all reconcile")
