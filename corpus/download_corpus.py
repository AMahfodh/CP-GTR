"""Download the primary statutory corpus for LegiSafe-Bench.

Fetches authoritative, pre-extraction source text. Be a good citizen: this runs
locally (NOT in any sandbox), sends a descriptive User-Agent, and pauses between
requests. If a publisher blocks automated access, the script prints the URL so
you can save the page manually -- the corpus is only four documents.

Optional (recommended) for clean text: pip install beautifulsoup4 lxml requests
"""
import os
import re
import time
import sys

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

try:
    from bs4 import BeautifulSoup
    HAVE_BS4 = True
except ImportError:
    HAVE_BS4 = False

RAW = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(RAW, exist_ok=True)

UA = {"User-Agent": "LegiSafe-Bench/1.0 (academic research; contact: you@univ.edu)"}

# NOTE: verify each URL in a browser first; publisher endpoints change.
# eCFR date is a snapshot date -- change it to the current amendment date.
MANIFEST = {
    "coppa_16cfr312": {
        "url": "https://www.ecfr.gov/api/versioner/v1/full/2025-01-01/title-16.xml?part=312",
        "kind": "xml", "jurisdiction": "US",
        "human": "https://www.ecfr.gov/current/title-16/chapter-I/subchapter-C/part-312",
    },
    "gdpr_reg2016_679": {
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679",
        "kind": "html", "jurisdiction": "EU",
        "human": "https://eur-lex.europa.eu/eli/reg/2016/679/oj",
    },
    "ccpa_civ_1798": {
        "url": ("https://leginfo.legislature.ca.gov/faces/codes_displayText.xhtml"
                "?division=3.&part=4.&lawCode=CIV&title=1.81.5"),
        "kind": "html", "jurisdiction": "US-CA",
        "human": "https://oag.ca.gov/privacy/ccpa",
    },
    "uk_aadc": {
        "url": ("https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/"
                "childrens-information/childrens-code-guidance-and-resources/"
                "age-appropriate-design-a-code-of-practice-for-online-services/"),
        "kind": "html", "jurisdiction": "UK",
        "human": "https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/"
                 "childrens-information/",
    },
}


def strip_markup(text, kind):
    if HAVE_BS4:
        soup = BeautifulSoup(text, "lxml" if kind == "xml" else "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return re.sub(r"\n{3,}", "\n\n", soup.get_text("\n"))
    # crude fallback: drop tags
    txt = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"[ \t]{2,}", " ", txt)


def fetch(name, meta):
    raw_path = os.path.join(RAW, f"{name}.{meta['kind']}")
    txt_path = os.path.join(RAW, f"{name}.txt")
    try:
        r = requests.get(meta["url"], headers=UA, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"  [!] automated fetch failed for {name}: {e}")
        print(f"      Open manually and save into corpus/raw/{name}.txt :")
        print(f"      {meta['human']}")
        return False
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(r.text)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(strip_markup(r.text, meta["kind"]))
    print(f"  [ok] {name}: {len(r.text):,} bytes -> {txt_path}")
    return True


if __name__ == "__main__":
    print("Downloading primary statutory corpus...")
    for name, meta in MANIFEST.items():
        print(f"- {name} ({meta['jurisdiction']})")
        fetch(name, meta)
        time.sleep(2)          # be polite
    print("\nDone. Raw sources in corpus/raw/. "
          "These are the pre-extraction texts your Stage-1 extractor consumes.")
    print("Reminder: statutory currency must be confirmed with counsel; "
          "this is not legal advice.")