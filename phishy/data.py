"""Data loading, cleaning and splitting.

Preferred: ONE dataset where BOTH classes are real URLs (CSV with url + label).
Fallback: PhishTank (phishing) + Tranco (legit, synthetic paths).
"""
import random

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from config import SEED
from phishy.features import parse_url

PHISH_LABELS = {"1", "phishing", "phish", "bad", "malicious", "malware", "defacement", "true"}
LEGIT_LABELS = {"0", "legitimate", "legit", "benign", "good", "safe", "false"}


def _to_binary(v):
    s = str(v).strip().lower()
    if s in PHISH_LABELS:
        return 1
    if s in LEGIT_LABELS:
        return 0
    return None  # unknown label -> dropped


def load_url_csv(path, url_col="url", label_col="label", invert_labels=False) -> pd.DataFrame:
    """Load any CSV with a URL column and a label column. Output: url, label (1 = phishing).

    invert_labels=True for datasets where 1 means LEGITIMATE (e.g. PhiUSIIL).
    Only the two needed columns are read, so wide datasets load fast.
    """
    df = pd.read_csv(path, usecols=[url_col, label_col])
    df = df.rename(columns={url_col: "url", label_col: "label"})
    df["label"] = df["label"].map(_to_binary)
    df = df.dropna()
    if invert_labels:
        df["label"] = 1 - df["label"].astype(int)
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop empty rows and duplicates (after normalisation: scheme/www ignored)."""
    df = df.dropna(subset=["url"]).copy()
    df["_key"] = df["url"].map(lambda u: parse_url(u).full)
    df = df[df["_key"] != ""].drop_duplicates("_key").drop(columns="_key")
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def balance(df: pd.DataFrame, seed=SEED) -> pd.DataFrame:
    n = df["label"].value_counts().min()
    parts = [g.sample(n=n, random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def group_split(df: pd.DataFrame, test_size=0.2, seed=SEED):
    """Split so the same registered domain never appears in both train and test."""
    groups = df["url"].map(lambda u: parse_url(u).registered_domain)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, df["label"], groups=groups))
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)


# ------------------------------------------------ PhishTank + Tranco fallback
_SUBS = ["blog", "shop", "mail", "app", "support", "m", "en", "docs",
         "help", "store", "news", "portal", "cdn", "my", "learn"]
_WORDS = ["about", "contact", "products", "services", "blog", "news", "help",
          "support", "pricing", "docs", "api", "category", "user", "profile",
          "search", "cart", "checkout", "article", "post", "download", "careers",
          "faq", "terms", "privacy", "settings", "dashboard", "payment",
          "getting-started", "reviews", "events", "electronics"]
_SUSP = ["login", "signin", "account", "update", "verify", "secure", "billing"]
_EXT = ["", "", "", ".html", ".php", ".aspx", ".htm"]
_QKEYS = ["id", "q", "page", "ref", "lang", "utm_source", "sort", "cat", "session"]


def make_legit_urls(domains, seed=SEED):
    rng = random.Random(seed)

    def word():  # ~5% of words are 'suspicious' -> no inverted shortcut
        return rng.choice(_SUSP) if rng.random() < 0.05 else rng.choice(_WORDS)

    def slug():
        s = "-".join(word() for _ in range(rng.randint(1, 3)))
        return s + (f"-{rng.randint(1, 99999)}" if rng.random() < 0.3 else "")

    def value():
        t = rng.random()
        if t < 0.4:
            return str(rng.randint(1, 999999))
        if t < 0.7:
            return rng.choice(_WORDS)
        return "".join(rng.choices("abcdef0123456789", k=rng.randint(6, 24)))

    out = []
    for d in domains:
        scheme = "https" if rng.random() < 0.85 else "http"
        r = rng.random()
        host = d if r < 0.45 else f"www.{d}" if r < 0.75 else f"{rng.choice(_SUBS)}.{d}"
        n_seg = rng.choices(range(6), [20, 22, 22, 18, 12, 6])[0]
        segs = [word() if rng.random() < 0.5 else slug() for _ in range(n_seg)]
        path = "/" + "/".join(segs) + rng.choice(_EXT) if segs else rng.choice(["", "/"])
        query = ""
        if rng.random() < 0.3:
            query = "?" + "&".join(f"{rng.choice(_QKEYS)}={value()}" for _ in range(rng.randint(1, 4)))
        out.append(f"{scheme}://{host}{path}{query}")
    return out


def build_phishtank_tranco(phishtank_csv, tranco_csv, n_per_class=10000, top_k=100_000):
    ph = pd.read_csv(phishtank_csv)["url"].dropna().drop_duplicates()
    tr = pd.read_csv(tranco_csv, header=None, names=["rank", "domain"]).dropna()
    tr = tr[tr["rank"] <= top_k].drop_duplicates("domain")   # popular domains only
    n = min(n_per_class, len(ph), len(tr))
    phishing = pd.DataFrame({"url": ph.sample(n=n, random_state=SEED).values, "label": 1})
    legit = pd.DataFrame({
        "url": make_legit_urls(tr.sample(n=n, random_state=SEED)["domain"]), "label": 0})
    return clean(pd.concat([phishing, legit], ignore_index=True))
