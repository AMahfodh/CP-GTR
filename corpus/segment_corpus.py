"""Segment raw statutory texts into provision spans tagged with (J, tau).

Step 2 of the corpus build. Reads every ``*.txt`` file in ``corpus/raw/`` and
writes ``corpus/provisions.json``, the input to the Stage-1 extractor
(``cpgtr/extract.py``). Each record has the fields ``id`` (``prov_NNNN``),
``source`` (raw file name), ``jurisdiction`` (J), ``tau`` (an age threshold
parsed from the span text, or ``None``) and ``text`` (the span, truncated to
STORE_CHARS characters).

The raw statutory texts are NOT distributed with this repository. Obtain them
from the official publishers as described in ``corpus/download_corpus.py``,
which automates only part of the job. The segmenter expects these files in
``corpus/raw/``:

    coppa_16cfr312.txt      US      16 CFR Part 312 (COPPA), from eCFR
    gdpr_reg2016_679.txt    EU      Regulation (EU) 2016/679, full text from
                                    EUR-Lex; only Article 8 is segmented
    ccpa_civ_1798.txt       US-CA   California Civil Code 1798.x (CCPA)
    uk_aadc2.txt            UK      ICO Age Appropriate Design Code, text
                                    extracted from the PDF
    OHCHR.txt               UN      UN Convention on the Rights of the Child

The jurisdiction is chosen from the file name (see JMAP). Any other ``*.txt``
file in ``corpus/raw/`` is also segmented, with the generic splitter and
jurisdiction "UNKNOWN", so keep that folder to the five files above
(notably, remove the ``uk_aadc.txt`` that the downloader writes).

Run order, from the repository root:

    python corpus/download_corpus.py     # fetch the sources it covers
    # add uk_aadc2.txt and OHCHR.txt by hand (not covered by the downloader)
    python corpus/segment_corpus.py      # writes corpus/provisions.json

With the snapshot of the texts used for the paper this produces 1,493 spans
(UK AADC 795, CCPA 362, UN CRC 179, COPPA 153, GDPR Art. 8: 4). Texts fetched
later may produce somewhat different counts.

Segmentation is per source rather than one generic splitter, because the
sources differ in structural marker and in what the raw file contains:

  - GDPR: the raw file is the whole regulation, but the benchmark targets
    Article 8 only. segment_gdpr() cuts the text between the "Article 8" and
    "Article 9" headings and splits that article into sub-paragraph spans.
  - COPPA: the text may contain undecoded HTML entities (for example
    ``&#xA7;`` for the section sign), which happens when the text was saved
    without BeautifulSoup. Every source is entity-decoded before segmenting.
  - CCPA: the raw text has no blank lines, so a blank-line splitter would
    return one giant span. segment_ccpa() splits on the numbered section
    headers ("1798.100.", "1798.105.", ...) instead.
  - UK AADC: PDF-extracted text, with page-break banners, lone page-number
    lines and hard-wrapped lines. segment_uk_aadc() removes the page noise and
    re-joins wrapped lines into sentence-level units before splitting.
  - UN CRC (OHCHR): the generic splitter gives a sensible result; no
    per-source function is needed.

Spans longer than MAX_SPAN_CHARS are split further on lettered or numbered
sub-section markers. validate() and the span-length check in main() raise an
error instead of writing an implausible provisions.json (wrong per-source
counts, one source dominating the corpus, or spans outside the length range).
Inspect the output and correct it by hand if it still looks wrong.
"""
import os
import re
import json
import glob
import html

RAW = os.path.join(os.path.dirname(__file__), "raw")
OUT = os.path.join(os.path.dirname(__file__), "provisions.json")

JMAP = {"coppa": "US", "gdpr": "EU", "ccpa": "US-CA", "uk_aadc": "UK", "OHCHR": "United Nations" }
AGE_RE = re.compile(r"\b(?:under|below)\s+the\s+age\s+of\s+(\d{1,2})|\b(1[0-9])\s+years",
                    re.IGNORECASE)

MAX_SPAN_CHARS = 5000     # validation ceiling, checked BEFORE the STORE_CHARS cap
MIN_SPAN_CHARS = 20       # validation floor
STORE_CHARS = 2000        # stored spans are truncated to this length


def jurisdiction_of(fname):
    for key, j in JMAP.items():
        if key in fname:
            return j
    return "UNKNOWN"


def find_tau(text):
    m = AGE_RE.search(text)
    if not m:
        return None
    return int(m.group(1) or m.group(2))


def _generic_segment(text):
    # Split on blank lines and on "Article N" / "§" markers; keep pieces >= 60 chars.
    parts = re.split(r"\n\s*\n|(?=^\s*Article\s+\d+)|(?=^\s*§)", text, flags=re.MULTILINE)
    return [p.strip() for p in parts if len(p.strip()) >= 60]


def _strip_pdf_artifacts(text):
    """Drop page-break banners ("-- Page N --") and lone page-number lines.

    PDF-derived text (the UK AADC) contains these, and they would otherwise
    fragment real paragraphs into spurious spans."""
    cleaned = []
    for line in text.split("\n"):
        s = line.strip()
        if re.match(r"^-+\s*Page\s+\d+\s*-+$", s, re.IGNORECASE):
            continue
        if re.fullmatch(r"\d{1,4}", s):   # a lone page number on its own line
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _dewrap(text):
    """Rebuild logical paragraphs from hard-wrapped PDF text.

    PDF extraction for this source puts every visual line on its own text
    line and inserts blank lines at arbitrary points, sometimes mid-sentence,
    so blank lines are not a reliable paragraph boundary. Instead, consecutive
    non-blank lines are joined until one ends in terminal punctuation
    (. ! ? :), which follows sentence and clause boundaries regardless of
    where the extractor wrapped the text.
    """
    paragraphs, buf = [], []
    for raw in text.split("\n"):
        s = raw.rstrip("\r").strip()
        if not s:
            continue   # blank lines are ignored, not treated as boundaries
        buf.append(s)
        if re.search(r"[.!?:]\s*$", s):
            paragraphs.append(" ".join(buf))
            buf = []
    if buf:
        paragraphs.append(" ".join(buf))
    return "\n\n".join(paragraphs)


def segment_gdpr(text):
    """Extract Article 8 from the full regulation and split it into spans.

    Article 8 runs from its heading to the "Article 9" heading. Raises
    RuntimeError if the "Article 8" heading is missing, rather than falling
    back to segmenting the whole regulation."""
    m = re.search(r"^\s*Article\s+8\s*$", text, flags=re.MULTILINE)
    if not m:
        raise RuntimeError(
            "gdpr_reg2016_679.txt: 'Article 8' heading not found -- raw file "
            "may have changed format; do not silently fall back to segmenting "
            "the whole regulation."
        )
    start = m.start()
    m_next = re.search(r"^\s*Article\s+9\s*$", text[start:], flags=re.MULTILINE)
    end = start + m_next.start() if m_next else len(text)
    article8 = text[start:end]
    return _generic_segment(article8)


def segment_ccpa(text):
    """Split on CCPA section headers ("1798.100.", "1798.105.", ...).

    The raw text has no blank lines, so numbered headers are the only
    structural marker."""
    parts = re.split(r"(?=\b1798\.\d+(?:\.\d+)?\.\s)", text)
    return [p.strip() for p in parts if len(p.strip()) >= 60]


def segment_uk_aadc(text):
    """Strip PDF page noise, re-join wrapped lines, then split generically."""
    return _generic_segment(_dewrap(_strip_pdf_artifacts(text)))


# Maps a substring of the raw file name to its segmenter. Files that match no
# key use _generic_segment().
SOURCE_SEGMENTERS = {
    "gdpr": segment_gdpr,
    "ccpa": segment_ccpa,
    "uk_aadc": segment_uk_aadc,
}


def _split_long_span(text, max_chars=MAX_SPAN_CHARS):
    """Split an over-long span on sub-section markers such as "(a)", "(1)".

    A span still longer than max_chars after the per-source split is usually
    one long section with nested lettered or numbered sub-sections (for
    example CCPA 1798.130). It is split on those markers, recursing on pieces
    that are still too long. If no marker is found the span is returned
    unchanged, and the length check in main() then rejects it.
    """
    if len(text) <= max_chars:
        return [text]
    parts = [p.strip() for p in re.split(r"(?=\(\w{1,3}\)\s)", text)
             if len(p.strip()) >= MIN_SPAN_CHARS]
    if len(parts) <= 1:
        return [text]
    out = []
    for p in parts:
        out.extend(_split_long_span(p, max_chars))
    return out


def _normalize_whitespace(text):
    """Remove a UTF-8 BOM and turn non-breaking spaces into plain spaces.

    html.unescape() maps "&nbsp;" to U+00A0 rather than a space; normalizing
    keeps invisible characters out of the spans passed to the extractor."""
    return text.replace("﻿", "").replace("\xa0", " ")


def segment_for(fname, text):
    # Decode HTML entities (e.g. "&#xA7;" -> section sign); harmless for clean text.
    text = _normalize_whitespace(html.unescape(text))
    for key, fn in SOURCE_SEGMENTERS.items():
        if key in fname:
            spans = fn(text)
            break
    else:
        spans = _generic_segment(text)

    out = []
    for span in spans:
        out.extend(_split_long_span(span))
    return out


# Per-source exceptions to the default validation checks (at least 5 spans per
# source, and no source above 50% of the corpus). Each is a deliberate
# exception, not a loosened threshold:
#   - gdpr_reg2016_679.txt: segment_gdpr() narrows the whole regulation to
#     Article 8, which has only three numbered paragraphs plus a heading, so
#     the minimum is 4 spans instead of 5.
#   - uk_aadc2.txt: the ICO code of practice is a much longer document (about
#     290 KB) than the other sources' scoped extracts (one GDPR article, the
#     relevant CCPA and COPPA sections, the 54 CRC articles), so it
#     legitimately makes up about 53% of the spans. Its ceiling is widened to
#     60% for this source only; any other source above 50%, or the AADC
#     above 60%, still fails validation.
MIN_SPANS_OVERRIDE = {"gdpr_reg2016_679.txt": 4}
MAX_SHARE_OVERRIDE = {"uk_aadc2.txt": 0.60}


def validate(provisions):
    """Raise RuntimeError on implausible segmentation.

    Checks for duplicate span ids, too few spans per source, and any source
    making up too large a share of the corpus (the failure mode of, for
    example, segmenting the whole GDPR, or collapsing CCPA into one span)."""
    errors = []
    by_source = {}
    seen_ids = set()
    total = len(provisions)

    for p in provisions:
        if p["id"] in seen_ids:
            errors.append(f"duplicate span id: {p['id']}")
        seen_ids.add(p["id"])
        by_source.setdefault(p["source"], []).append(p)

    for source, spans in sorted(by_source.items()):
        n = len(spans)
        min_spans = MIN_SPANS_OVERRIDE.get(source, 5)
        max_share = MAX_SHARE_OVERRIDE.get(source, 0.5)
        if n < min_spans:
            errors.append(f"{source}: only {n} spans (< {min_spans})")
        if total and n / total > max_share:
            errors.append(
                f"{source}: {n}/{total} spans ({n / total:.0%} > {max_share:.0%} of corpus)"
            )

    if errors:
        raise RuntimeError(
            "Segmentation validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )


def main():
    provisions, pid = [], 0
    span_length_errors = []

    for path in sorted(glob.glob(os.path.join(RAW, "*.txt"))):
        fname = os.path.basename(path)
        J = jurisdiction_of(fname)
        text = open(path, encoding="utf-8").read()
        for span in segment_for(fname, text):
            if len(span) < MIN_SPAN_CHARS or len(span) > MAX_SPAN_CHARS:
                span_length_errors.append(
                    f"prov_{pid:04d} ({fname}): {len(span)} chars "
                    f"(outside [{MIN_SPAN_CHARS}, {MAX_SPAN_CHARS}])"
                )
            provisions.append({
                "id": f"prov_{pid:04d}",
                "source": fname,
                "jurisdiction": J,
                "tau": find_tau(span),
                "text": span[:STORE_CHARS],       # truncated for the extractor
            })
            pid += 1

    if span_length_errors:
        raise RuntimeError(
            "Segmentation validation failed (span length):\n"
            + "\n".join(f"  - {e}" for e in span_length_errors)
        )
    validate(provisions)

    by_source = {}
    for p in provisions:
        by_source.setdefault(p["source"], 0)
        by_source[p["source"]] += 1
    print("Per-source span counts:")
    for source, n in sorted(by_source.items()):
        print(f"  {source:<28} {n:>6}  ({n / len(provisions):.1%})")
    print(f"  {'Total':<28} {len(provisions):>6}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(provisions, f, indent=2)
    print(f"\nWrote {len(provisions)} provision spans -> {OUT}")
    print("Next: feed provisions.json to your Stage-1 extractor (see cpgtr/extract.py).")

if __name__ == "__main__":
    main()
