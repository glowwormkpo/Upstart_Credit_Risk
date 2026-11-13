# 1-2. Imports (add numpy as np), ROOT, read both files, asserts, join -> preds

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
logreg_df = pd.read_parquet(ROOT / "data" / "processed" / "preds_logreg.parquet")
lgbm_df = pd.read_parquet(ROOT / "data" / "processed" / "preds_lgbm.parquet")

assert (logreg_df.index == lgbm_df.index).all(), "Index mismatch between files!"
assert (logreg_df["default"] == lgbm_df["default"]).all(), "Target mismatch between files!"

preds = logreg_df.join(lgbm_df[["lgbm_prob"]])

# 3. y = preds["default"].to_numpy();  n = len(y)

y = preds["default"].to_numpy()
n = len(y)

# 4. scores = {"LC int_rate": "int_rate", "LogReg": "logreg_prob", "LightGBM": "lgbm_prob"}
#    (a dictionary: display name -> column name, so the loop below handles all three)

scores = {
    "LC int_rate": "int_rate",
    "LogReg": "logreg_prob",
    "LightGBM": "lgbm_prob",
}

# 5. rows = []
#    for approval_rate in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#        k = int(n * approval_rate)                  # how many loans get approved
#        row = {"approval_rate": approval_rate}
#        for name, col in scores.items():
#            order = np.argsort(preds[col].to_numpy())   # row positions sorted lowest -> highest risk
#            approved = order[:k]                         # the first k = the approved loans
#            row[name] = <default rate among approved>    # hint: y[approved].mean()
#        rows.append(row)

rows = []
for approval_rate in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    k = int(n * approval_rate)
    row = {"approval_rate": approval_rate}
    for name, col in scores.items():
        order = np.argsort(preds[col].to_numpy(), kind="stable")
        approved = order[:k]
        row[name] = y[approved].mean()
    rows.append(row)

# 6. results = pd.DataFrame(rows)
#    Add a column "LightGBM vs LC (%)":  (results["LightGBM"] / results["LC int_rate"] - 1) * 100
#    Print it rounded to 4 decimals.

results = pd.DataFrame(rows)
results["LightGBM vs LC (%)"] = (results["LightGBM"] / results["LC int_rate"] - 1) * 100
print(results.round(4).to_string(index=False))

# 7. Same risk, more approvals
#    a) A helper function that returns the running default rate after each approved loan:
#         def running_default_rate(col):
#             order = np.argsort(preds[col].to_numpy(), kind="stable")
#             return np.cumsum(y[order]) / np.arange(1, n + 1)
#       (cumsum / 1,2,3...n = the default rate after approving 1 loan, 2 loans, ... n loans)
#
#    b) lc_rate = running_default_rate("int_rate");  gbm_rate = running_default_rate("lgbm_prob")
#
#    c) For approval_rate in [0.6, 0.7, 0.8]:
#         k = int(n * approval_rate)
#         target = lc_rate[k - 1]                              # LC's default rate at that approval rate
#         k_gbm = np.where(gbm_rate <= target)[0].max() + 1   # most loans LightGBM can approve at <= target
#         print the approval rate, target, k_gbm / n, and the extra loans  (k_gbm - k)

def running_default_rate(col):
    order = np.argsort(preds[col].to_numpy(), kind="stable")
    return np.cumsum(y[order]) / np.arange(1, n + 1)

lc_rate = running_default_rate("int_rate")
gbm_rate = running_default_rate("lgbm_prob")

for approval_rate in [0.6, 0.7, 0.8]:
    k = int(n * approval_rate)
    target = lc_rate[k - 1]
    k_gbm = np.where(gbm_rate <= target)[0].max() + 1
    gbm_approval_rate = k_gbm / n
    extra_loans = k_gbm - k

    print(
        f"LC approval: {approval_rate:.0%} | "
        f"Target default rate: {target:.4f} | "
        f"LightGBM approval: {gbm_approval_rate:.2%} | "
        f"Extra loans: +{extra_loans:,}"
    )
