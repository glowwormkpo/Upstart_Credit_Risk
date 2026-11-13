"""
Step 1 - Build the modeling table for Lending Club 36-month loans.

Target: default = 1 if loan_status is 'Charged Off' or 'Default', 0 if 'Fully Paid'.
Universe: individual applications, 36-month term, issued 2013-01 through 2015-12
          (2013 is the first year the extended credit-bureau fields are populated;
           2015 is the last year whose 36-month loans had finished by the 2018Q4 data cut).
Rule: keep only information known at application time. Anything recorded after the
      loan was issued (payments, recoveries, hardship, settlement, later FICO pulls)
      leaks the outcome and is dropped.

Run from the project root:  python3 src/01_prepare_data.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv.gz"
OUT = ROOT / "data" / "processed" / "loans_36m_2013_2015.parquet"
DECISIONS = ROOT / "reports" / "column_decisions.csv"

# ---- Columns known at application time (model features) -------------------------
NUMERIC = [
    "loan_amnt", "annual_inc", "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc", "collections_12_mths_ex_med",
    "mths_since_last_major_derog", "acc_now_delinq", "tot_coll_amt", "tot_cur_bal",
    "total_rev_hi_lim", "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy", "bc_util",
    "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc", "mths_since_recent_bc",
    "mths_since_recent_bc_dlq", "mths_since_recent_inq", "mths_since_recent_revol_delinq",
    "num_accts_ever_120_pd", "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats", "num_bc_tl",
    "num_il_tl", "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats",
    "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m", "num_tl_op_past_12m",
    "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies", "tax_liens",
    "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit",
]
CATEGORICAL = ["emp_length", "home_ownership", "verification_status", "purpose"]

# ---- Lending Club's own pricing (kept only as the benchmark, never as a feature) ----
LC_PRICING = ["grade", "sub_grade", "int_rate", "installment"]

# ---- Needed to build the universe, target and derived features ---------------------
HELPER = ["term", "issue_d", "loan_status", "application_type", "earliest_cr_line"]

# ---- Excluded on purpose, with the reason (written to reports/column_decisions.csv) --
EXCLUDED = {
    "leakage: recorded after the loan was issued": [
        "funded_amnt", "funded_amnt_inv", "pymnt_plan", "out_prncp", "out_prncp_inv",
        "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
        "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d",
        "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d", "last_fico_range_high",
        "last_fico_range_low", "initial_list_status", "hardship_flag", "hardship_type",
        "hardship_reason", "hardship_status", "deferral_term", "hardship_amount",
        "hardship_start_date", "hardship_end_date", "payment_plan_start_date",
        "hardship_length", "hardship_dpd", "hardship_loan_status",
        "orig_projected_additional_accrued_interest", "hardship_payoff_balance_amount",
        "hardship_last_payment_amount", "disbursement_method", "debt_settlement_flag",
        "debt_settlement_flag_date", "settlement_status", "settlement_date",
        "settlement_amount", "settlement_percentage", "settlement_term",
    ],
    "fair lending: geography can act as a proxy for protected classes": [
        "zip_code", "addr_state",
    ],
    "not populated before late 2015 (would be missing for all training loans)": [
        "open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m", "mths_since_rcnt_il",
        "total_bal_il", "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc", "all_util",
        "inq_fi", "total_cu_tl", "inq_last_12m",
    ],
    "joint applications only (universe is individual applicants)": [
        "annual_inc_joint", "dti_joint", "verification_status_joint", "revol_bal_joint",
        "sec_app_fico_range_low", "sec_app_fico_range_high", "sec_app_earliest_cr_line",
        "sec_app_inq_last_6mths", "sec_app_mort_acc", "sec_app_open_acc",
        "sec_app_revol_util", "sec_app_open_act_il", "sec_app_num_rev_accts",
        "sec_app_chargeoff_within_12_mths", "sec_app_collections_12_mths_ex_med",
        "sec_app_mths_since_last_major_derog",
    ],
    "identifier, free text or constant": [
        "id", "member_id", "url", "desc", "emp_title", "title", "policy_code",
    ],
}


def write_decisions() -> None:
    rows = [(c, "feature", "known at application time") for c in NUMERIC + CATEGORICAL]
    rows += [(c, "benchmark only", "Lending Club's own risk grade / price") for c in LC_PRICING]
    rows += [(c, "helper", "used to build universe, target or derived features") for c in HELPER]
    rows += [(c, "excluded", why) for why, cols in EXCLUDED.items() for c in cols]
    pd.DataFrame(rows, columns=["column", "role", "reason"]).to_csv(DECISIONS, index=False)


def load() -> pd.DataFrame:
    usecols = NUMERIC + CATEGORICAL + LC_PRICING + HELPER
    parts = []
    for chunk in pd.read_csv(RAW, usecols=usecols, chunksize=200_000, low_memory=False):
        chunk = chunk[chunk["term"].str.strip().eq("36 months")
                      & chunk["application_type"].eq("Individual")]
        chunk["issue_d"] = pd.to_datetime(chunk["issue_d"], format="%b-%Y")
        chunk = chunk[(chunk["issue_d"] >= "2013-01-01") & (chunk["issue_d"] <= "2015-12-01")]
        parts.append(chunk)
    return pd.concat(parts, ignore_index=True)


def build(df: pd.DataFrame) -> pd.DataFrame:
    status_counts = df["loan_status"].value_counts()
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])].copy()
    df["default"] = df["loan_status"].isin(["Charged Off", "Default"]).astype("int8")

    # Derived features (all from application-time fields)
    ecl = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
    df["credit_history_months"] = ((df["issue_d"] - ecl).dt.days / 30.44).round()
    df["fico_mid"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)
    df["revol_bal_to_income"] = df["revol_bal"] / df["annual_inc"].replace(0, np.nan)
    df["int_rate"] = df["int_rate"].astype(float)
    df["issue_year"] = df["issue_d"].dt.year.astype("int16")

    # Out-of-time split: train 2013-01..2014-09, validation 2014-10..12, test = 2015
    df["split"] = np.select(
        [df["issue_d"] < "2014-10-01", df["issue_d"] < "2015-01-01"],
        ["train", "valid"], default="test",
    )
    for c in CATEGORICAL + ["grade", "sub_grade"]:
        df[c] = df[c].astype("category")
    df = df.drop(columns=["term", "application_type", "earliest_cr_line",
                          "fico_range_low", "fico_range_high"])
    return df, status_counts


if __name__ == "__main__":
    write_decisions()
    raw = load()
    df, status_counts = build(raw)
    df.to_parquet(OUT, index=False)
    print(f"Loans in universe before status filter: {len(raw):,}")
    print("Dropped unfinished loans:",
          {k: int(v) for k, v in status_counts.items()
           if k not in ("Fully Paid", "Charged Off", "Default")})
    summary = df.groupby("split").agg(loans=("default", "size"), default_rate=("default", "mean"),
                                      first=("issue_d", "min"), last=("issue_d", "max"))
    print(summary.loc[["train", "valid", "test"]].to_string())
    print(f"Saved {OUT.relative_to(ROOT)}  shape={df.shape}")
