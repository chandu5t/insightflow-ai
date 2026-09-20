"""Helpers to build small DataFrames exactly like the real loader does."""

import io
import textwrap
from pathlib import Path

import pandas as pd

from app.utils.dataframe_utils import clean_text_frame

BACKEND_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = BACKEND_DIR / "data" / "evaluation"
SALES_PATH = EVAL_DIR / "module3_sales.csv"  # derived revenue (quantity x unit_price)
DIRECT_PATH = EVAL_DIR / "module3_sales_direct.csv"  # direct revenue column


def frame_from_csv(text: str) -> pd.DataFrame:
    """CSV text -> DataFrame: all text, spaces trimmed, empty cells are NaN."""
    raw = pd.read_csv(io.StringIO(textwrap.dedent(text).strip()), dtype=str, keep_default_na=False)
    return clean_text_frame(raw)