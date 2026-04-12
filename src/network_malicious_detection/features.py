"""Feature engineering for classical malicious URL models."""

from __future__ import annotations

import re
from urllib.parse import urlparse

import pandas as pd

_IP_REGEX = re.compile(
    r"(([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\."
    r"([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\/)|"
    r"((0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\/)|"
    r"(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}"
)
_SUSPICIOUS_WORDS_REGEX = re.compile(
    r"paypal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr",
    re.IGNORECASE,
)
_SHORTENER_REGEX = re.compile(
    r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|"
    r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|"
    r"short\.to|budurl\.com|ping\.fm|post\.ly|just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|"
    r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|lnkd\.in|db\.tt|qr\.ae|"
    r"adf\.ly|bitly\.com|cur\.lv|tinyurl\.com|ity\.im|q\.gs|po\.st|bc\.vc|twitthis\.com|u\.to|"
    r"j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|prettylinkpro\.com|scrnch\.me|filoops\.info|"
    r"vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|link\.zip\.net",
    re.IGNORECASE,
)


def _fd_length(url: str) -> int:
    path = urlparse(url).path
    parts = path.split("/")
    return len(parts[1]) if len(parts) > 1 else 0


def _tld_length(url: str) -> int:
    host = urlparse(url).hostname or ""
    parts = host.split(".")
    return len(parts[-1]) if parts and parts[-1] else -1


def build_lexical_features(df: pd.DataFrame) -> pd.DataFrame:
    urls = df["url"].astype(str)
    features = pd.DataFrame(index=df.index)

    features["use_of_ip"] = urls.apply(lambda x: int(bool(_IP_REGEX.search(x))))
    features["abnormal_url"] = urls.apply(
        lambda x: int((urlparse(x).hostname or "") not in x)
    )
    features["count."] = urls.str.count(r"\.")
    features["count-www"] = urls.str.count("www")
    features["count@"] = urls.str.count("@")
    features["count_dir"] = urls.apply(lambda x: urlparse(x).path.count("/"))
    features["count_embed_domian"] = urls.apply(lambda x: urlparse(x).path.count("//"))
    features["sus_url"] = urls.apply(lambda x: int(bool(_SUSPICIOUS_WORDS_REGEX.search(x))))
    features["short_url"] = urls.apply(lambda x: int(bool(_SHORTENER_REGEX.search(x))))
    features["count-https"] = urls.str.count("https")
    features["count-http"] = urls.str.count("http")
    features["count%"] = urls.str.count("%")
    features["count-"] = urls.str.count("-")
    features["count="] = urls.str.count("=")
    features["url_length"] = urls.str.len()
    features["hostname_length"] = urls.apply(lambda x: len(urlparse(x).netloc))
    features["fd_length"] = urls.apply(_fd_length)
    features["tld_length"] = urls.apply(_tld_length)
    features["count-digits"] = urls.apply(lambda x: sum(ch.isdigit() for ch in x))
    features["count-letters"] = urls.apply(lambda x: sum(ch.isalpha() for ch in x))
    return features

