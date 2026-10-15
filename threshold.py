from phishy import data
from phishy.train import train, metrics

df = data.balance(data.load_url_csv("data/PhiUSIIL_Phishing_URL_Dataset.csv", "URL", "label", True))
df = df.groupby("label").head(50000).reset_index(drop=True)
bundle, _ = train(df)

_, test_df = data.group_split(df)
prob = bundle["model"].predict_proba(bundle["builder"].transform(test_df["url"]))[:, 1]
for t in [0.5, 0.4, 0.3, 0.2]:
    m = metrics(test_df["label"], prob, t)
    print(t, m["precision"], m["recall"], m["f1"])