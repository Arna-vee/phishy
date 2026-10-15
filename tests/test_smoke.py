import phishy  # noqa: F401
from phishy.features import FEATURES, FeatureBuilder, parse_url


def test_www_and_scheme_are_ignored():
    a = FeatureBuilder().fit_transform(["https://www.example.com/a"])
    b = FeatureBuilder().fit_transform(["http://example.com/a"])
    assert a.iloc[0].drop("tld_freq").equals(b.iloc[0].drop("tld_freq"))


def test_ip_detection():
    assert FEATURES["has_ip"].fn(parse_url("http://192.168.1.1/login")) == 1
    assert FEATURES["has_ip"].fn(parse_url("https://google.com")) == 0


def test_transform_matches_columns():
    fb = FeatureBuilder()
    fb.fit_transform(["https://a.com", "http://b.xyz/x?y=1"])
    X = fb.transform(["https://unseen.tld/path"])
    assert list(X.columns) == fb.columns
