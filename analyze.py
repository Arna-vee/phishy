"""
Error analysis: threshold table + examples the model gets wrong.

  python analyze.py --data data/PhiUSIIL_Phishing_URL_Dataset.csv --url-col URL --label-col label --invert-labels --n-per-class 50000
"""
import argparse

import pandas as pd

import phishy  # noqa: F401
from phishy import data
from phishy.train import metrics, train

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--url-col", default="url")
ap.add_argument("--label-col", default="label")
ap.add_argument("--invert-labels", action="store_true")
ap.add_argument("--n-per-class", type=int, default=10000)
ap.add_argument("--model", default="xgboost")
ap.add_argument("--cut", type=float, default=0.3,
                help="show phishing URLs scored below this as 'missed'")
a = ap.parse_args()

# Same data preparation and split as phishy.train, so numbers match.
df = data.balance(data.load_url_csv(a.data, a.url_col, a.label_col, a.invert_labels))
df = df.groupby("label").head(a.n_per_class).reset_index(drop=True)
bundle, _ = train(df, a.model)

_, test_df = data.group_split(df)
test_df = test_df.copy()
test_df["prob"] = bundle["model"].predict_proba(bundle["builder"].transform(test_df["url"]))[:, 1]

print("\n=== Threshold table (precision / recall / f1) ===")
for t in [0.5, 0.4, 0.3, 0.2]:
    m = metrics(test_df["label"], test_df["prob"].values, t)
    print(f"{t}: precision={m['precision']}  recall={m['recall']}  f1={m['f1']}")

pd.set_option("display.max_colwidth", 110)
pd.set_option("display.width", 200)


def show(title, subset):
    print(f"\n=== {title} ({len(subset)} total, showing up to 20) ===")
    if len(subset):
        s = subset.sample(min(20, len(subset)), random_state=1)
        print(s[["url", "prob"]].round(3).to_string(index=False))


show(f"MISSED phishing (prob < {a.cut})", test_df[(test_df.label == 1) & (test_df.prob < a.cut)])
show("FALSE ALARMS: legit flagged as phishing (prob > 0.7)",
     test_df[(test_df.label == 0) & (test_df.prob > 0.7)])