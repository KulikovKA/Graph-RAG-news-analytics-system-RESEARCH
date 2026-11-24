"""Граничные проверки на синтетическом JATS."""

from hashlib import sha256

import pytest

from prepare import extract_jats, find_duplicate, openalex_eligibility, pair_keys


TEXT = (
    "This study evaluates a reproducible measurement protocol for a reference benchmark. "
    "We collected local measurements from a common source and compared the results across "
    "independently configured instruments. The data were split into several groups by site "
    "and each location was evaluated using the same set of regression and classification "
    "metrics. The results showed lower error for two of the reference instruments, while "
    "the third instrument gave less consistent results across runs. Replication across "
    "more sites is needed to measure the impact of sampling uncertainty."
)
JATS = f"""<article article-type="research-article" xml:lang="en"
 xmlns:xlink="http://www.w3.org/1999/xlink"><front><article-meta>
 <article-id pub-id-type="pmcid">PMC1</article-id><article-id pub-id-type="doi">10.1234/one</article-id>
 <title-group><article-title>Reference benchmark</article-title></title-group>
 <contrib-group><contrib contrib-type="author"><name><surname>Test</surname><given-names>Ada</given-names>
 </name></contrib></contrib-group>
 <permissions><license xlink:href="https://creativecommons.org/licenses/by/4.0/">
 <license-p>Licensed under CC BY 4.0</license-p></license></permissions>
 <abstract><p>{TEXT}</p></abstract><abstract abstract-type="graphical"><p>Graphical caption</p></abstract>
 </article-meta></front><body><p>Body sentinel.</p></body></article>"""

CANDIDATE = {"pmcid": "PMC1", "doi": "10.1234/one"}
WORK = CANDIDATE | {
    "title": "Reference benchmark", "id": "https://openalex.org/W1", "language": "en",
    "type": "article", "is_retracted": False, "publication_year": 2023,
    "ids": {"pmcid": "https://www.ncbi.nlm.nih.gov/pmc/articles/1/"},
    "primary_location": {"source": {"type": "journal"}}, "primary_topic": {"field": {"id": "field"}},
}


def test_publisher_abstract_only():
    raw = JATS.encode()
    source = extract_jats(raw, CANDIDATE, WORK)
    assert source["text"] == TEXT
    assert source["raw_xml_sha256"] == sha256(raw).hexdigest()
    assert source["license"] == "CC-BY-4.0"
    assert source["authors"] == ["Ada Test"]


@pytest.mark.parametrize(("before", "after", "reason"), [
    ("/by/", "/by-nc/", "unconfirmed_jats_license"),
    ("PMC1", "PMC2", "jats_identity_mismatch"),
    (' abstract-type="graphical"', "", "missing_or_ambiguous_abstract"),
    ("<abstract><p>", "<abstract><permissions/><p>", "abstract_extra_material_or_terms"),
    (TEXT, "Too short.", "abstract_word_count"),
    ("The data were", "� The data were", "damaged_or_sparse_abstract"),
])
def test_reject_jats(before, after, reason):
    with pytest.raises(ValueError, match=reason):
        extract_jats(JATS.replace(before, after).encode(), CANDIDATE, WORK)


def test_openalex_pmc_id_variants():
    assert openalex_eligibility(WORK, CANDIDATE) is None
    assert openalex_eligibility(WORK | {"is_retracted": True}, CANDIDATE) == "openalex_type_language_or_retraction"
    assert openalex_eligibility(WORK, CANDIDATE | {"pmcid": "PMC2"}) == "openalex_pmcid_mismatch"


def test_deduplicate_revised_text_and_related_version():
    work = WORK | {"document_id": "W1", "text": TEXT, "source_reference": {"related_articles": []}}
    revise = work | {"document_id": "W2", "title": "Different title", "doi": "10.1234/two",
                     "pmcid": "PMC2", "text": TEXT.replace("reference benchmark.", "reference benchmark!")}
    assert find_duplicate(revise, pair_keys(revise), [pair_keys(work)])["rule"] == "text_hash"
    revise |= {"text": "Unrelated words.", "source_reference": {"related_articles": ["https://doi.org/10.1234/one"]}}
    assert find_duplicate(revise, pair_keys(revise), [pair_keys(work)])["rule"] == "related_jats_version"
