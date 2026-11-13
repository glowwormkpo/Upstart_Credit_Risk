# 1. Imports: pandas, Path, roc_auc_score, and  import lightgbm as lgb

from pathlib import Path
from sklearn.metrics import roc_auc_score
import pandas as pd
import lightgbm as lgb

# 2. Copy your setup from 03 exactly: ROOT, DATA, df, NOT_FEATURES, FEATURES,
#    and the train / valid / test split.
#    (You don't need CATEGORICAL / NUMERIC this time - LightGBM reads the category types itself.)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "loans_36m_2013_2015.parquet"

df = pd.read_parquet(DATA)

NOT_FEATURES = [
    "default",
    "split",
    "issue_d",
    "issue_year",
    "loan_status",
    "grade",
    "sub_grade",
    "int_rate",
    "installment",
]

FEATURES = [col for col in df.columns if col not in NOT_FEATURES]

train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]
test = df[df["split"] == "test"]

# 3. Create the model:
model = lgb.LGBMClassifier(
    n_estimators=3000,       # maximum number of trees (early stopping will cut this way down)
    learning_rate=0.03,      # how much each new tree is allowed to change the prediction
    num_leaves=31,           # how complex each tree can be
    min_child_samples=200,   # each leaf needs at least 200 loans -> no rules built on tiny groups
    subsample=0.8,           # each tree sees a random 80% of loans...
    subsample_freq=1,        # ...re-drawn for every tree
    colsample_bytree=0.7,    # ...and a random 70% of features
    reg_lambda=1.0,          # a penalty that keeps any single rule from getting too extreme
    random_state=42,         # makes the randomness repeatable, so you get the same result each run
    verbose=-1,              # hide LightGBM's training chatter
)

# 4. Fit on train, with valid as the early-stopping check. Fill in the blanks:

model.fit(
    train[FEATURES],
    train["default"],
    eval_X=valid[FEATURES],
    eval_y=valid["default"],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(200, verbose=False)],
)

# 5. Print model.best_iteration_  (how many trees it actually kept)

print(f"Best iteration: {model.best_iteration_}")

# 6. Score valid and test exactly like you did in 03, and print both AUCs to 4 decimals.

valid_probs = model.predict_proba(valid[FEATURES])[:, 1]
valid_auc = roc_auc_score(valid["default"], valid_probs)

test_probs = model.predict_proba(test[FEATURES])[:, 1]
test_auc = roc_auc_score(test["default"], test_probs)

print(f"Valid AUC: {valid_auc:.4f}")
print(f"Test AUC:  {test_auc:.4f}")

# 7. Save test predictions
#    Smaller this time - the first file already has issue_d, default, sub_grade and int_rate.
#    a) preds = test[["default"]].copy()     (keep default so you can sanity-check the join later)
#    b) Add a column "lgbm_prob" equal to test_probs
#    c) Save to  data/processed/preds_lgbm.parquet

preds = test[["default"]].copy()
preds["lgbm_prob"] = test_probs
preds.to_parquet(ROOT / "data" / "processed" / "preds_lgbm.parquet")
