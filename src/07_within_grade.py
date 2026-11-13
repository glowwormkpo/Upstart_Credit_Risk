# 1-2. Same setup as 06: imports, ROOT, read both files, asserts, join -> preds

from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
logreg_df = pd.read_parquet(ROOT / "data" / "processed" / "preds_logreg.parquet")
lgbm_df = pd.read_parquet(ROOT / "data" / "processed" / "preds_lgbm.parquet")

assert (logreg_df.index == lgbm_df.index).all(), "Index mismatch between files!"
assert (logreg_df["default"] == lgbm_df["default"]).all(), "Target mismatch between files!"

preds = logreg_df.join(lgbm_df[["lgbm_prob"]])

# 3. rows = []

rows = []

# 4. Loop over sub-grades:
#       for sub_grade, group in preds.groupby("sub_grade", observed=True):
#    (observed=True = only loop over sub-grades that actually appear; it's needed for category columns)
#    Append a dictionary per sub-grade with:
#       sub_grade, loans, default_rate, lgbm_auc, logreg_auc

for sub_grade, group in preds.groupby("sub_grade", observed=True):
    # Skip sub-grades with fewer than 2 outcome classes (e.g., tiny G5 buckets)
    if group["default"].nunique() < 2:
        continue

    lgbm_auc = roc_auc_score(group["default"], group["lgbm_prob"])
    logreg_auc = roc_auc_score(group["default"], group["logreg_prob"])

    rows.append({
        "sub_grade": sub_grade,
        "loans": len(group),
        "default_rate": group["default"].mean(),
        "lgbm_auc": lgbm_auc,
        "logreg_auc": logreg_auc,
    })

# 5. results = pd.DataFrame(rows), and print it rounded to 4 decimals

results = pd.DataFrame(rows)
print(results.round(4).to_string(index=False))

# 6. Print a loan-weighted average of lgbm_auc and of logreg_auc across sub-grades:
#       (results["lgbm_auc"] * results["loans"]).sum() / results["loans"].sum()
#    Weighted, because A1 has 12,372 loans and G5 has 6 - they shouldn't count equally.

weighted_lgbm = (results["lgbm_auc"] * results["loans"]).sum() / results["loans"].sum()
weighted_logreg = (results["logreg_auc"] * results["loans"]).sum() / results["loans"].sum()

print(f"\nLoan-weighted within-grade LightGBM AUC: {weighted_lgbm:.4f}")
print(f"Loan-weighted within-grade LogReg AUC:   {weighted_logreg:.4f}")
