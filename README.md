# Phishy - URL phishing detector

Modular, hackathon-ready. Everything you'll want to change on exam day lives in ONE place:

| I want to...                    | Edit this                      |
|---------------------------------|--------------------------------|
| add a feature                   | `phishy/custom_features.py`    |
| add / swap a model              | `phishy/models.py`             |
| use a different dataset         | just pass `--data other.csv`   |
| change the decision threshold   | `--threshold 0.4`              |
| change the demo UI              | `app.py`                       |

## Quick start
```bash
pip install -r requirements.txt

# Option A (recommended): one dataset, both classes real URLs (columns: url, label)
python -m phishy.train --data data/urls.csv --url-col url --label-col label

# Option B: PhishTank + Tranco (legit URLs are synthetic -> treat the score with caution)
python -m phishy.train --phishtank data/PhishTank.csv --tranco data/tranco.csv

# Honest evaluation: also test on a DIFFERENT dataset
python -m phishy.train --data data/A.csv --ood-data data/B.csv

python -m phishy.predict "http://paypal-secure-login.xyz/verify"   # CLI
streamlit run app.py                                               # demo UI
pytest -q                                                          # sanity tests
```
**PhiUSIIL dataset** (UCI): its label is reversed (1 = legitimate), and the URL column is `URL`:
```bash
python -m phishy.train --data data/PhiUSIIL_Phishing_URL_Dataset.csv \
    --url-col URL --label-col label --invert-labels --n-per-class 50000
```
Only the URL column is used; its precomputed and HTML-based columns are ignored on purpose.

Labels are auto-mapped (`phishing/bad/malicious/1` -> 1, `legitimate/benign/good/0` -> 0).

## How it works
```
URL -> parse_url (strip scheme + www)        phishy/features.py
    -> @feature functions (registry)         -> FeatureBuilder (+ TLD frequency, fit on train only)
    -> model (xgboost / random_forest / logreg)   phishy/models.py
    -> Detector.predict / score / explain    phishy/predict.py
```
Design choices to mention in the viva:
- **Scheme and `www.` are stripped** so formatting can't leak the label.
- **Split by registered domain** (`GroupShuffleSplit`) so test domains are unseen.
- **One saved bundle** (builder + model + threshold + metrics) so training and prediction can't drift.
- **Explainability**: per-URL feature contributions (XGBoost `pred_contribs`).

## Hackathon cheat sheet

**Add a feature (2 minutes)** - in `phishy/custom_features.py`:
```python
from phishy.features import feature

@feature()
def has_punycode(p):            # p is a ParsedURL: p.host, p.path, p.query, p.subdomain, p.domain, p.suffix
    return 1 if "xn--" in p.host else 0
```
Then retrain. It appears in training, prediction, explanations and the UI automatically.

**Add a model**:
```python
@register_model("lightgbm")
def _lgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=300)
```
Then `--model lightgbm`.

**Add a network feature** (WHOIS / HTTP): `@feature(network=True)`, train with `--network-features`.
Slow and can fail offline - wrap in try/except and return a default.

**Idea list**, easiest first:
1. Brand impersonation: brand name (paypal, amazon, sbi...) in subdomain/path but not the registered domain.
2. Typosquatting: edit distance of the domain to a top-domain list.
3. Known URL shorteners (bit.ly, qrco.de, t.co) as a feature.
4. Threshold tuning: pick the threshold for a target recall (`--threshold`).
5. Domain age via WHOIS (network feature).
6. HTML content features: forms with password fields, external-link ratio (network feature).
7. FastAPI endpoint wrapping `Detector.predict` or a browser extension calling it.
8. Prediction history / logging in the Streamlit app.

## Known limitations (say this before they ask)
- URL structure only: no page content, domain age, or reputation.
- Abused legitimate hosts (docs.google.com, qrco.de) look legit by structure.
- The score only means something on data like what it trained on - check with `--ood-data`.
