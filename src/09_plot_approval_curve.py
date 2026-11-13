# 1-2. Imports: numpy, pandas, Path, and
#         import matplotlib.pyplot as plt
#      ROOT, read both files, asserts, join -> preds;  y and n as before

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
logreg_df = pd.read_parquet(ROOT / "data" / "processed" / "preds_logreg.parquet")
lgbm_df = pd.read_parquet(ROOT / "data" / "processed" / "preds_lgbm.parquet")

assert (logreg_df.index == lgbm_df.index).all(), "Index mismatch between files!"
assert (logreg_df["default"] == lgbm_df["default"]).all(), "Target mismatch between files!"

preds = logreg_df.join(lgbm_df[["lgbm_prob"]])
y = preds["default"].to_numpy()
n = len(y)

# 3. Approval rates every 1% from 40% to 100%:
#      rates = np.arange(0.40, 1.001, 0.01)
#    (1.001 instead of 1.0 so 1.00 is included - arange stops *before* its end value)

rates = np.arange(0.40, 1.001, 0.01)

# 4. Write a FUNCTION that returns one model's default-rate curve:
#      def default_curve(col):
#          order = np.argsort(preds[col].to_numpy(), kind="stable")
#          cum_defaults = np.cumsum(y[order])      # running count of defaults, safest loan first
#          return np.array([cum_defaults[int(n * a) - 1] / int(n * a) for a in rates])
#    Then:  lc_curve = default_curve("int_rate")   and   gbm_curve = default_curve("lgbm_prob")

def default_curve(col):
    order = np.argsort(preds[col].to_numpy(), kind="stable")
    cum_defaults = np.cumsum(y[order])
    return np.array([cum_defaults[int(n * a) - 1] / int(n * a) for a in rates])

lc_curve = default_curve("int_rate")
gbm_curve = default_curve("lgbm_prob")

# 5. Plot.  fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
#      ax.plot(rates * 100, lc_curve * 100,  color="#eb6834", linewidth=2)   # LC = orange
#      ax.plot(rates * 100, gbm_curve * 100, color="#2a78d6", linewidth=2)   # LightGBM = blue

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
ax.plot(rates * 100, lc_curve * 100, color="#eb6834", linewidth=2)
ax.plot(rates * 100, gbm_curve * 100, color="#2a78d6", linewidth=2)

# 6. Make it readable:
#      - Axis labels: "Share of applicants approved (%)" and "Default rate among approved loans (%)"
#      - A title that states the FINDING, not the topic, with the data source on a second line:
#          "At every approval rate, LightGBM's approved loans default less\n2015 36-month Lending Club loans (n = 282,787)"
#        Use ax.set_title(..., loc="left", fontsize=11)
#      - Label the lines directly instead of using a legend box. Put the text next to each line
#        at the 60% approval point with ax.annotate(...). Look up its xytext and textcoords
#        arguments to offset the label so it doesn't sit on top of the line.
#      - Remove the top and right borders:  ax.spines["top"].set_visible(False)  (same for "right")
#      - Light horizontal grid only, behind the lines:
#          ax.grid(axis="y", color="#e5e4df", linewidth=0.8);  ax.set_axisbelow(True)

ax.set_xlabel("Share of applicants approved (%)")
ax.set_ylabel("Default rate among approved loans (%)")
ax.set_title(
    "At every approval rate, LightGBM's approved loans default less\n"
    f"2015 36-month Lending Club loans (n = {n:,})",
    loc="left",
    fontsize=11,
)

# Direct labels anchored at the 60% approval point (index 20 in rates: 0.40 + 20*0.01 = 0.60)
idx_60 = 20
ax.annotate(
    "Lending Club (int_rate)",
    xy=(60, lc_curve[idx_60] * 100),
    xytext=(-10, 10),
    textcoords="offset points",
    ha="right",
    color="#eb6834",
    fontweight="bold",
)
ax.annotate(
    "LightGBM",
    xy=(60, gbm_curve[idx_60] * 100),
    xytext=(10, -14),
    textcoords="offset points",
    ha="left",
    color="#2a78d6",
    fontweight="bold",
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#e5e4df", linewidth=0.8)
ax.set_axisbelow(True)

# 7. fig.tight_layout(), then save to reports/figures/approval_curve.png  (fig.savefig(path))

fig.tight_layout()
out_path = ROOT / "reports" / "figures" / "approval_curve.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path)
print(f"Saved chart to {out_path}")
