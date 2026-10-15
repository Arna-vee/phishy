"""
Inference + explanation.

  python -m phishy.predict https://example.com/login
"""
import sys
from typing import List

import joblib
import pandas as pd

import phishy  # noqa: F401
from config import MODEL_PATH


class Detector:
    def __init__(self, path=MODEL_PATH):
        self.bundle = joblib.load(path)
        self.builder = self.bundle["builder"]
        self.model = self.bundle["model"]
        self.threshold = self.bundle["threshold"]

    @property
    def report(self):
        return self.bundle.get("report", {})

    def score(self, urls: List[str]) -> pd.DataFrame:
        """Batch scoring -> DataFrame(url, probability, verdict)."""
        X = self.builder.transform(urls)
        prob = self.model.predict_proba(X)[:, 1]
        return pd.DataFrame({
            "url": list(urls),
            "probability": prob.round(4),
            "verdict": ["Phishing" if p >= self.threshold else "Legitimate" for p in prob],
        })

    def explain(self, url: str, top_k: int = 6):
        """Per-feature contribution to the score (XGBoost only; else None).
        Positive = pushes towards phishing, negative = towards legitimate."""
        try:
            import xgboost as xgb
            X = self.builder.transform([url])
            booster = self.model.get_booster()
            contrib = booster.predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]
        except Exception:
            return None
        out = pd.DataFrame({"feature": X.columns, "value": X.iloc[0].values,
                            "contribution": contrib})
        out = out.reindex(out["contribution"].abs().sort_values(ascending=False).index)
        return out.head(top_k).reset_index(drop=True)

    def predict(self, url: str) -> dict:
        row = self.score([url]).iloc[0]
        return {"url": url, "verdict": row["verdict"],
                "probability": float(row["probability"]),
                "top_features": self.explain(url)}


if __name__ == "__main__":
    det = Detector()
    for u in sys.argv[1:] or ["https://example.com"]:
        r = det.predict(u)
        print(f"{r['verdict']:10s} {r['probability']:.1%}  {u}")
        if r["top_features"] is not None:
            print(r["top_features"].to_string(index=False), "\n")
