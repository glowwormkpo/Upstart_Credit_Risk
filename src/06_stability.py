# 1. Imports: pandas, Path, roc_auc_score

from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score

# 2. ROOT, read both prediction files, the two asserts, and the join -> preds
#    (copy from 05)

ROOT = Path(__file__).resolve().parents[1]
logreg_df = pd.read_parquet(ROOT / "data" / "processed" / "preds_logreg.parquet")
lgbm_df = pd.read_parquet(ROOT / "data" / "processed" / "preds_lgbm.parquet")

assert (logreg_df.index == lgbm_df.index).all(), "Index mismatch between files!"
assert (logreg_df["default"] == lgbm_df["default"]).all(), "Target mismatch between files!"

preds = logreg_df.join(lgbm_df[["lgbm_prob"]])

# 3. Add two columns to preds:
#      "quarter"  = the quarter of issue_d             hint: preds["issue_d"].dt.quarter
#      "lc_score" = sub_grade's .cat.codes

preds["quarter"] = preds["issue_d"].dt.quarter
preds["lc_score"] = preds["sub_grade"].cat.codes

# 4. Make an empty list called rows

rows = []

# 5. Loop over the quarters:   for quarter, group in preds.groupby("quarter"):
#      group is a DataFrame holding just that quarter's loans.
#      Compute the AUC of lc_score, logreg_prob and lgbm_prob within the group,
#      then append a dictionary to rows, e.g.
#        rows.append({"quarter": quarter, "loans": len(group),
#                     "default_rate": group["default"].mean(),
#                     "lc_auc": ..., "logreg_auc": ..., "lgbm_auc": ...,
#                     "lgbm_lift": lgbm_auc - lc_auc})

for quarter, group in preds.groupby("quarter"):
    lc_auc = roc_auc_score(group["default"], group["lc_score"])
    logreg_auc = roc_auc_score(group["default"], group["logreg_prob"])
    lgbm_auc = roc_auc_score(group["default"], group["lgbm_prob"])

    rows.append({
        "quarter": quarter,
        "loans": len(group),
        "default_rate": group["default"].mean(),
        "lc_auc": lc_auc,
        "logreg_auc": logreg_auc,
        "lgbm_auc": lgbm_auc,
        "lgbm_lift": lgbm_auc - lc_auc,
    })

# 6. Turn rows into a DataFrame:  results = pd.DataFrame(rows)

results = pd.DataFrame(rows)

#    Print it rounded to 4 decimals:  print(results.round(4).to_string(index=False))

print(results.round(4).to_string(index=False))
