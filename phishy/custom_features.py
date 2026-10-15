"""
HACKATHON PLAYGROUND: add new features here.

Example (uncomment to try):

from phishy.features import feature

@feature()
def has_punycode(p):
    return 1 if "xn--" in p.host else 0

After adding a feature, re-run:  python -m phishy.train ...
(the saved model only knows the features it was trained with)
"""
from phishy.features import feature  # noqa: F401
