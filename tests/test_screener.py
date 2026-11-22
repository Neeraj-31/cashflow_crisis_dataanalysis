"""The Screener parser reproduces the hand-transcribed Jyoti data exactly (saved page in tests/fixtures)."""
from pathlib import Path

import pandas as pd

from src.ingest import screener as sc

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "tests" / "fixtures" / "screener_jyoti_consolidated.html"


def test_parser_matches_transcribed_jyoti():
    raw = sc.parse_company(PAGE.read_text(encoding="utf-8"))
    raw = raw[raw.period_months == 12]
    raw.index = raw.index.year
    j = pd.read_csv(ROOT / "data" / "raw" / "jyoti_screener.csv", index_col="fy")
    yrs = [y for y in j.index if y in raw.index]
    cols = [c for c in j.columns if c in raw.columns]
    assert len(yrs) >= 10 and len(cols) >= 20
    d = (raw.loc[yrs, cols] - j.loc[yrs, cols]).abs()
    assert (d.stack().dropna() < 0.5).all()


def test_interim_column_does_not_replace_full_year():
    c = sc.to_canonical(sc.parse_company(PAGE.read_text(encoding="utf-8")))
    assert c.fy.is_unique
    assert (c.period_months == 12).all()
