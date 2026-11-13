# 1. Imports: pandas, Path from pathlib, and roc_auc_score from sklearn.metrics

from pathlib import Path
from sklearn.metrics import roc_auc_score
import pandas as pd

# 2. Build the path to data/processed/loans_36m_2013_2015.parquet
#    Hint: copy the ROOT = Path(__file__).resolve().parents[1] line from 01_prepare_data.py

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "loans_36m_2013_2015.parquet"

# 3. Read the parquet file into a DataFrame called df   (pd.read_parquet)

df = pd.read_parquet(DATA)

# 4. Keep only the test rows (split == "test") in a DataFrame called test

test = df[df["split"] == "test"]

# 5. Compute AUC three ways and print each, rounded to 4 decimals:
#      a) sub_grade  -> use test["sub_grade"].cat.codes as the score
auc_sub_grade = roc_auc_score(test["default"], test["sub_grade"].cat.codes)
#      b) grade      -> same idea
auc_grade = roc_auc_score(test["default"], test["grade"].cat.codes)
#      c) int_rate   -> already a number, use it directly
auc_int_rate = roc_auc_score(test["default"], test["int_rate"])
#    roc_auc_score takes (true labels, scores). The true labels are test["default"].
print(f"sub_grade AUC: {auc_sub_grade:.4f}")
print(f"grade AUC:     {auc_grade:.4f}")
print(f"int_rate AUC:  {auc_int_rate:.4f}")