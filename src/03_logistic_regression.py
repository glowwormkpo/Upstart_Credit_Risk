# 1. Imports: pandas, Path, and roc_auc_score (same as 02_benchmark.py for now)

from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import pandas as pd

# 2. ROOT and DATA paths, then read the parquet into df (same as 02)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "loans_36m_2013_2015.parquet"

df = pd.read_parquet(DATA)

# 3. Make a list called NOT_FEATURES with the 9 column names

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

# 4. Make a list called FEATURES: every column in df.columns that is NOT in NOT_FEATURES
#    Hint: a list comprehension  ->  [c for c in df.columns if ...]

FEATURES = [col for col in df.columns if col not in NOT_FEATURES]

# 5. Make a list called CATEGORICAL: the columns in FEATURES whose type is "category"
#    Hint: df[FEATURES].select_dtypes("category").columns.tolist()
#    Then make NUMERIC: the FEATURES that are not in CATEGORICAL (same comprehension pattern)

CATEGORICAL = df[FEATURES].select_dtypes("category").columns.tolist()
NUMERIC = [col for col in FEATURES if col not in CATEGORICAL]

# 6. Split into train / valid / test DataFrames

train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]
test = df[df["split"] == "test"]

# 7. Print len(FEATURES), len(NUMERIC), len(CATEGORICAL), and the CATEGORICAL list

print(f"Total FEATURES: {len(FEATURES)}")
print(f"NUMERIC:        {len(NUMERIC)}")
print(f"CATEGORICAL:    {len(CATEGORICAL)}")
print("Categorical columns:", CATEGORICAL)

# 8. Numeric pipeline called numeric_steps:  Pipeline([ (name, step), (name, step) ])
#      step 1 ("impute"): SimpleImputer with strategy="median", add_indicator=True
#      step 2 ("scale"):  StandardScaler()

numeric_steps = Pipeline([
    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ("scale", StandardScaler()),
])

# 9. Categorical pipeline called categorical_steps, same shape:
#      step 1 ("impute"): SimpleImputer with strategy="constant", fill_value="missing"
#      step 2 ("onehot"): OneHotEncoder with handle_unknown="ignore"
#      (handle_unknown="ignore" = if 2015 has a category train never saw, don't crash)

categorical_steps = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

# 10. ColumnTransformer called preprocess that sends NUMERIC to numeric_steps
#     and CATEGORICAL to categorical_steps:
#       ColumnTransformer([ ("num", numeric_steps, NUMERIC), ("cat", categorical_steps, CATEGORICAL) ])

preprocess = ColumnTransformer([
    ("num", numeric_steps, NUMERIC),
    ("cat", categorical_steps, CATEGORICAL),
])

# 11. Full model called model: Pipeline with two steps:
#       ("prep", preprocess) and ("logreg", LogisticRegression(max_iter=1000))

model = Pipeline([
    ("prep", preprocess),
    ("logreg", LogisticRegression(max_iter=1000)),
])

# 12. Fit on TRAIN only:  model.fit(train[FEATURES], train["default"])

model.fit(train[FEATURES], train["default"])

# 13. Score valid and test. predict_proba returns two columns (P(paid), P(default));
#     you want the second one:  model.predict_proba(valid[FEATURES])[:, 1]
#     Pass that into roc_auc_score with valid["default"], then do the same for test.
#     Print both AUCs to 4 decimals.

valid_probs = model.predict_proba(valid[FEATURES])[:, 1]
valid_auc = roc_auc_score(valid["default"], valid_probs)

test_probs = model.predict_proba(test[FEATURES])[:, 1]
test_auc = roc_auc_score(test["default"], test_probs)

print(f"Valid AUC: {valid_auc:.4f}")
print(f"Test AUC:  {test_auc:.4f}")

# 14. Save test predictions for evaluation
#     a) Make a DataFrame called preds with these test columns: issue_d, default, sub_grade, int_rate
#        Hint: test[[ ...list of column names... ]].copy()
#        (Double brackets = "give me these columns as a DataFrame"; .copy() = a separate copy you can add to)
#     b) Add a column called "logreg_prob" equal to test_probs
#     c) Save it:  preds.to_parquet(ROOT / "data" / "processed" / "preds_logreg.parquet")

preds = test[["issue_d", "default", "sub_grade", "int_rate"]].copy()
preds["logreg_prob"] = test_probs
preds.to_parquet(ROOT / "data" / "processed" / "preds_logreg.parquet")
