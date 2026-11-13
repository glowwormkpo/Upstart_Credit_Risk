# 1. Imports: numpy as np, pandas as pd, Path, roc_auc_score

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# 2. ROOT, then read both prediction files (preds_logreg.parquet and preds_lgbm.parquet)

ROOT = Path(__file__).resolve().parents[1]
logreg_df = pd.read_parquet(ROOT / "data" / "processed" / "preds_logreg.parquet")
lgbm_df = pd.read_parquet(ROOT / "data" / "processed" / "preds_lgbm.parquet")

# Optional sanity check before joining

assert logreg_df.index.equals(lgbm_df.index), "Files don't contain the same loans in the same order!"
assert (logreg_df["default"] == lgbm_df["default"]).all(), "Row mismatch between files!"

# 3. Join them into one DataFrame called preds.
#    Only bring lgbm_prob over from the second file (default is already in the first):
#        preds = <first>.join(<second>[["lgbm_prob"]])
#    .join() matches rows on the index - the SQL JOIN we talked about.

preds = logreg_df.join(lgbm_df[["lgbm_prob"]])

# 4. Pull out four numpy arrays:
#      y          = the default column            -> .to_numpy()
#      lc_score   = sub_grade's .cat.codes        -> .to_numpy()
#      lr_score   = logreg_prob                   -> .to_numpy()
#      gbm_score  = lgbm_prob                     -> .to_numpy()

y = preds["default"].to_numpy()
lc_score = preds["sub_grade"].cat.codes.to_numpy()
lr_score = preds["logreg_prob"].to_numpy()
gbm_score = preds["lgbm_prob"].to_numpy()

# 5. Setup:
#      rng = np.random.default_rng(42)   # seeded random generator -> same results every run
#      n = len(y)
#      n_boot = 1000
#      lr_diffs = []    # empty lists to collect results
#      gbm_diffs = []

rng = np.random.default_rng(42)
n = len(y)
n_boot = 1000
lr_diffs = []
gbm_diffs = []

# 6. Loop n_boot times:  for i in range(n_boot):
#      a) idx = rng.integers(0, n, n)     # n random row positions, repeats allowed
#      b) lc_auc = AUC of y[idx] vs lc_score[idx]
#      c) append (logreg AUC on the same idx) - lc_auc  to lr_diffs
#      d) append (lgbm AUC on the same idx) - lc_auc    to gbm_diffs

for i in range(n_boot):
    idx = rng.integers(0, n, n)
    lc_auc = roc_auc_score(y[idx], lc_score[idx])
    lr_diffs.append(roc_auc_score(y[idx], lr_score[idx]) - lc_auc)
    gbm_diffs.append(roc_auc_score(y[idx], gbm_score[idx]) - lc_auc)

# 7. For each list, print:
#      the mean difference                         -> np.mean(...)
#      the 95% CI                                  -> np.percentile(..., [2.5, 97.5])
#      the share of resamples where diff <= 0      -> np.mean(np.array(...) <= 0)
#    (That last one works like a one-sided p-value: how often did your model NOT beat LC?)

for name, diffs in [("Logistic Regression", lr_diffs), ("LightGBM", gbm_diffs)]:
    diffs_arr = np.array(diffs)
    mean_diff = np.mean(diffs_arr)
    ci_low, ci_high = np.percentile(diffs_arr, [2.5, 97.5])
    p_val = np.mean(diffs_arr <= 0)

    print(f"--- {name} vs. Lending Club ---")
    print(f"Mean AUC lift: {mean_diff:+.4f}")
    print(f"95% CI:        [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"p-value:       {p_val:.4f}\n")
