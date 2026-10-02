"""Fetch primary statutory texts for LegiSafe-Bench (step 1 of the corpus build).

IMPORTANT: the raw statutory texts are NOT distributed with this repository.
Replicators must obtain them from the official publishers (the URLs are listed
in MANIFEST below) and place the resulting text files in ``corpus/raw/``.
Publisher pages change over time, so the text you obtain may differ slightly
from the snapshot used in the paper, and the span counts reported by
``corpus/segment_corpus.py`` may shift accordingly. Statutory currency must be
confirmed independently; nothing here is legal advice.

What this script automates (MANIFEST, one request every 2 seconds):

    name                jurisdiction  source                         writes
    coppa_16cfr312      US            eCFR, 16 CFR Part 312 (COPPA)  .xml + .txt
    gdpr_reg2016_679    EU            EUR-Lex, Regulation 2016/679   .html + .txt
    ccpa_civ_1798       US-CA         California Civil Code 1798.x   .html + .txt
    uk_aadc             UK            ICO Age Appropriate Design     .html + .txt
                                      Code guidance page

Each fetch saves the raw response as ``corpus/raw/<name>.<kind>`` and a
tag-stripped copy as ``corpus/raw/<name>.txt``. Tag stripping uses BeautifulSoup
when installed (``pip install requests beautifulsoup4 lxml``) and a crude regex
fallback otherwise; the fallback leaves HTML entities undecoded, which the
segmenter handles. If a publisher blocks automated access, the script prints
the human-readable URL; save the page manually as ``corpus/raw/<name>.txt``.

What this script does NOT automate. ``corpus/segment_corpus.py`` expects these
file names in ``corpus/raw/``:

    coppa_16cfr312.txt      produced by this script
    gdpr_reg2016_679.txt    produced by this script
    ccpa_civ_1798.txt       produced by this script
    uk_aadc2.txt            NOT produced by this script (see below)
    OHCHR.txt               NOT produced by this script (see below)

  * UK AADC: the benchmark uses ``uk_aadc2.txt``, text extracted from the ICO
    Code of Practice PDF (page-break banners included; the segmenter removes
    them). The MANIFEST entry ``uk_aadc`` instead writes ``uk_aadc.txt`` from
    the ICO web page, which is a different document; the segmenter would also
    process it if left in ``corpus/raw/``. To reproduce the paper's corpus,
    extract the PDF text yourself, save it as ``uk_aadc2.txt``, and do not
    keep ``uk_aadc.txt`` in the folder.
  * UN CRC: not in MANIFEST. Save the text of the Convention on the Rights of
    the Child from the OHCHR website (https://www.ohchr.org) as ``OHCHR.txt``.

Run order (from the repository root):

    python corpus/download_corpus.py     # fetch + convert to .txt
    # ... add uk_aadc2.txt and OHCHR.txt manually, remove uk_aadc.txt ...
    python corpus/segment_corpus.py      # writes corpus/provisions.json

Requires: ``requests`` (hard requirement); ``beautifulsoup4`` and ``lxml``
are optional.
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