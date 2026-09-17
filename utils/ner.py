"""Conservative spaCy-backed named-entity extraction for financial articles."""
from __future__ import annotations

import logging
from functools import lru_cache

from utils.entities import (
    BUSINESS_TYPE_SUFFIX_WORDS,
    TITLE_PREFIX_WORDS,
    is_valid_company_name,
    normalize_company_name,
)

LOG = logging.getLogger(__name__)

# spaCy entity types that, when spaCy ALSO applies one of them to the same
# (or a closely related) span elsewhere in the document, should be trusted
# over a conflicting ORG tag -- "prefer the stronger semantic type rather
# than treating every extracted span as a company." PERSON, places, dates/
# times, and other non-company categories should never become a company
# just because spaCy also happened to tag ORG somewhere.
_NON_COMPANY_ENTITY_TYPES = frozenset({
    "PERSON", "GPE", "LOC", "DATE", "TIME", "EVENT", "FAC", "PRODUCT", "NORP", "LAW",
})


@lru_cache(maxsize=1)
def _load_model():
    """Load en_core_web_sm, or fail loudly (once, via logging) rather than
    silently degrading.

    A bare `spacy.blank("en")` pipeline has NO trained NER component at
    all -- doc.ents is always empty -- which silently disables every
    person/place/date cross-check in this module (so e.g. "Trump" would
    never be filtered) and all spaCy-sourced company recall, while still
    LOOKING like NER is running (no exception, no crash). This previously
    happened silently whenever en_core_web_sm wasn't installed (it is a
    separate download from `pip install spacy` -- `python -m spacy
    download en_core_web_sm`), producing extraction results that looked
    like a mysterious accuracy regression rather than a missing model.
    Logging here makes that failure mode visible instead of silent; the
    blank-pipeline fallback is kept (better than crashing the whole
    pipeline over a missing NLP model), but every degraded run now says so.
    """
    try:
        import spacy
    except Exception as exc:
        LOG.warning("spaCy is not installed (%s); company extraction will rely on regex only.", exc)
        return None
    try:
        return spacy.load("en_core_web_sm")
    except Exception as exc:
        LOG.warning(
            "en_core_web_sm could not be loaded (%s); falling back to a blank spaCy "
            "pipeline with NO named-entity recognition. Company extraction will rely "
            "on regex only, and person/place/date cross-checks will not run. Install "
            "the model with: python -m spacy download en_core_web_sm",
            exc,
        )
        return spacy.blank("en")


def _parse(text: str):
    """Parse text once with spaCy, or return None if unavailable/empty."""
    model = _load_model()
    if model is None or not text:
        return None
    try:
        return model(text)
    except Exception:
        return None


def extract_orgs(text: str, known_tickers=None) -> list[str]:
    """Return unique, conservative ORG entities from article text.

    A thin, backward-compatible wrapper around merge_and_filter_companies
    with no regex-derived candidates -- spaCy's own ORG output only.
    """
    return merge_and_filter_companies(text, [], known_tickers=known_tickers)


def merge_and_filter_companies(text: str, regex_companies, known_tickers=None) -> list[str]:
    """Combine regex-derived company candidates with spaCy's own ORG
    entities, and filter BOTH through the same person/place/date/etc.
    cross-checks -- in a single parse of the text.

    This is what catches a regex-only candidate like "Trump" (produced by
    the possessive pattern from "Trump's oil and gas boom", which has no
    way on its own to know "Trump" is a person) -- cross-checking it
    against spaCy's PERSON tag for the same text rejects it exactly the
    way a raw spaCy ORG span would be. If spaCy isn't available, the regex
    candidates are returned unchanged (already validated by
    extract_companies() itself; nothing to cross-check against).

    Validity/normalization rules (generic-term rejection, temporal-word
    rejection, possessive stripping, government/index/media/product-term
    rejection) are shared with utils/parse.py via utils/entities.py, so the
    regex and spaCy extractors don't disagree on what counts as noise.
    `known_tickers` is forwarded to the shared validity check so a bare
    all-caps candidate (e.g. "AI", "CONDITIONS") is judged the same way a
    bare ticker candidate would be.

    Person-conflict checks, in increasing order of self-containedness --
    "prefer the stronger semantic type":
      1. Exact-text cross-check against any non-company type (PERSON, GPE,
         LOC, DATE, TIME, EVENT, FAC, PRODUCT, NORP, LAW) tagged elsewhere
         in the same document.
      2. Self-contained: the candidate itself contains a title/role word
         (TITLE_PREFIX_WORDS) immediately followed by 1-3 capitalized
         words ("CrowdStrike CEO George Kurtz") -- this doesn't need a
         separate PERSON tag to exist anywhere else.
      3. Cross-referenced: the candidate ends with a same-document PERSON
         span, AND either (a) a title word precedes it ("Fed Chairman
         Kevin Warsh") or (b) the remaining leading words already look
         like a complete company name on their own -- end in a
         business-type word like "Energy"/"Capital"/"Inc" ("Diamondback
         Energy Mehta").
      4. The candidate's own words are fully contained, as a contiguous
         run, within a longer PERSON span elsewhere in the document (e.g.
         candidate "Warsh" inside PERSON "Kevin Warsh").
    None of these apply to GPE, so "Palo Alto Networks" survives even when
    "Palo Alto" is separately tagged GPE in the same document -- only exact
    match applies there (check 1), and containment is deliberately reserved
    for PERSON only.
    """
    doc = _parse(text)
    if doc is None:
        return list(dict.fromkeys(regex_companies))
    try:
        org_texts = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        combined = list(regex_companies) + org_texts
        return _filter_conflicts(combined, doc, known_tickers=known_tickers)
    except Exception:
        return list(dict.fromkeys(regex_companies))


def _filter_conflicts(candidates, doc, known_tickers=None) -> list[str]:
    conflicting_spans = {
        ent.text.casefold() for ent in doc.ents if ent.label_ in _NON_COMPANY_ENTITY_TYPES
    }
    person_spans = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    seen: set[str] = set()
    result: list[str] = []
    for raw in candidates:
        value = normalize_company_name(raw)
        key = value.casefold()
        if key in conflicting_spans:
            continue
        if _contains_title_then_name(value):
            continue
        if _ends_with_extraneous_person(value, person_spans):
            continue
        if _contained_within_person_span(value, person_spans):
            continue
        if is_valid_company_name(value, known_tickers=known_tickers) and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _contains_title_then_name(value: str) -> bool:
    """Self-contained check: a title/role word directly followed by 1-3
    capitalized words looks like "Title FirstName LastName" regardless of
    whether spaCy separately tagged that name as PERSON anywhere else.
    """
    words = value.split()
    for i, word in enumerate(words):
        if word.casefold() not in TITLE_PREFIX_WORDS:
            continue
        remaining = words[i + 1:]
        if 1 <= len(remaining) <= 3 and all(w[:1].isupper() for w in remaining):
            return True
    return False


def _ends_with_extraneous_person(value: str, person_spans: list[str]) -> bool:
    words = value.split()
    for person in person_spans:
        person_words = person.split()
        n = len(person_words)
        if not n or n >= len(words):
            continue
        if [w.casefold() for w in words[-n:]] != [w.casefold() for w in person_words]:
            continue
        leading = words[:-n]
        if not leading:
            continue
        if any(w.casefold() in TITLE_PREFIX_WORDS for w in leading):
            return True
        if leading[-1].casefold() in BUSINESS_TYPE_SUFFIX_WORDS:
            return True
    return False


def _contained_within_person_span(value: str, person_spans: list[str]) -> bool:
    """The candidate's words are a contiguous subsequence of a longer PERSON
    span elsewhere in the document -- e.g. candidate "Warsh" is fully
    contained in PERSON "Kevin Warsh".
    """
    words = [w.casefold() for w in value.split()]
    n = len(words)
    for person in person_spans:
        person_words = [w.casefold() for w in person.split()]
        m = len(person_words)
        if n >= m:
            continue
        for start in range(m - n + 1):
            if person_words[start:start + n] == words:
                return True
    return False
