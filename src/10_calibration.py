# 1-2. Imports (pandas, Path), ROOT, read both files, asserts, join -> preds

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
logreg_df = pd.read_parquet(ROOT / "data" / "processed" / "preds_logreg.parquet")
lgbm_df = pd.read_parquet(ROOT / "data" / "processed" / "preds_lgbm.parquet")

assert (logreg_df.index == lgbm_df.index).all(), "Index mismatch between files!"
assert (logreg_df["default"] == lgbm_df["default"]).all(), "Target mismatch between files!"

preds = logreg_df.join(lgbm_df[["lgbm_prob"]])

# 3. Overall check - print, for lgbm_prob:
#      the mean predicted probability  vs  the actual default rate (mean of default)

mean_pred = preds["lgbm_prob"].mean()
actual_default = preds["default"].mean()
print(f"Overall mean predicted: {mean_pred:.4f}")
print(f"Overall actual default: {actual_default:.4f}\n")

# 4. Assign each loan to a decile of predicted risk:
#      preds["decile"] = pd.qcut(preds["lgbm_prob"], 10, labels=False) + 1
#    (qcut = "quantile cut": 10 groups with equal numbers of loans; +1 so they're 1-10 instead of 0-9)

preds["decile"] = pd.qcut(preds["lgbm_prob"], 10, labels=False) + 1


# 5. One summary row per decile, using named aggregation:
#      table = preds.groupby("decile").agg(
#          loans=("default", "size"),
#          avg_predicted=("lgbm_prob", "mean"),
#          actual_rate=("default", "mean"),
#      )
#    That's the pandas form of SQL's  SELECT decile, COUNT(*), AVG(...), AVG(...) ... GROUP BY decile

table = preds.groupby("decile").agg(
    loans=("default", "size"),
    avg_predicted=("lgbm_prob", "mean"),
    actual_rate=("default", "mean"),
)

# 6. Add a column "actual_vs_predicted" = actual_rate / avg_predicted, and print the table rounded to 4 decimals

table["actual_vs_predicted"] = table["actual_rate"] / table["avg_predicted"]
print(table.round(4).to_string())
