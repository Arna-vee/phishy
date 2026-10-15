"""Model registry. Add a model by writing a factory and decorating it."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import SEED

MODELS = {}


def register_model(name):
    def deco(factory):
        MODELS[name] = factory
        return factory
    return deco


@register_model("xgboost")
def _xgb():
    return XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                         eval_metric="logloss", random_state=SEED)


@register_model("random_forest")
def _rf():
    return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)


@register_model("logreg")
def _lr():
    # Scaler lives inside the pipeline, so prediction code needs no special case.
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
