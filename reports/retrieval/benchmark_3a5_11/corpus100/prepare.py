"""Сбор и воспроизведение среза Corpus100."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from difflib import SequenceMatcher
from itertools import combinations
import json
import math
from pathlib import Path
import re
from statistics import median, mean
import sys
import time
from threading import local
import unicodedata
from xml.etree import ElementTree as ET

import requests
from tqdm import tqdm

from cgraph.common import file_sha256, read_json, sha256, write_json
from cgraph.ner_v2.io import read_documents, validate_document, validate_span, write_jsonl
from cgraph.ner_v2.openalex import OpenAlexClient


REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[3]
DATA = ROOT / "data/retrieval/benchmark_3a5_11/corpus100"
SNAPSHOT = DATA / "sources"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
PMC_CLOUD = "https://pmc-oa-opendata.s3.amazonaws.com"
VERSION = "3A.5.11.1-corpus100-v1"
LICENSES = {
    "https://creativecommons.org/licenses/by/4.0/": "CC-BY-4.0",
    "https://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
}
HISTORICAL = {
    "Corpus30": "reports/ner_v2/research_run2/input_abstracts.jsonl",
    "Corpus12": "reports/ner_v2/benchmark_run1/input_abstracts.jsonl",
}
GROUPS = {
    "readmission": ("Machine Learning", '"hospital readmission" AND "machine learning"'),
    "external_validation": ("Machine Learning", '"external validation" AND "machine learning" AND "prediction model"'),
    "chest_xray": ("Computer Vision", '("chest x-ray" OR "chest radiograph") AND "classification" AND "deep learning"'),
    "pv_imaging": ("Computer Vision", '("photovoltaic" OR "solar panels") AND ("defect detection" OR "fault detection") AND ("image" OR "imaging" OR "vision")'),
    "clinical_ner": ("NLP и LLM", '"named entity recognition" AND ("clinical notes" OR "clinical text" OR "electronic health records")'),
    "radiology_llm": ("NLP и LLM", '("large language model" OR "large language models") AND ("radiology reports" OR "radiology report")'),
    "algal_lipids": ("Биотехнологии", '("microalgae" OR "microalgal") AND "lipid production"'),
    "biomass_hydrolysis": ("Биотехнологии", '"lignocellulosic biomass" AND ("enzymatic hydrolysis" OR "enzymatic saccharification")'),
    "pv_forecast": ("Энергетика", '"photovoltaic power forecasting"'),
    "algal_biogas": ("Энергетика", '"microalgae" AND "anaerobic digestion"'),
}
TOKEN_PATTERN = r"\w+|[^\w\s]"
SELECT = "id,doi,title,publication_year,language,type,is_retracted,primary_topic,primary_location,ids,updated_date"
OPENALEX_FILTER = "doi:"


def query_extensions():
    path = REPORT / "query_extensions.json"
    return read_json(path) if path.exists() else {}


def digest(text):
    return sha256(text.encode("utf-8"))


def display(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def words(text):
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold())


def doi(value):
    return re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", (value or "").strip().lower())


def pair_keys(record):
    title = words(record["title"])
    text = words(record["text"])
    return {
        "doi": doi(record.get("doi")),
        "id": record["document_id"],
        "pmcid": record.get("pmcid"),
        "author": " ".join(words((record.get("authors") or [""])[0])),
        "title": " ".join(title),
        "title_words": set(title),
        "text_hash": digest(" ".join(text)),
        "shingles": set(zip(text, text[1:], text[2:], text[3:], text[4:])),
    }


def overlap(left, right):
    for key in ("doi", "id", "pmcid", "title", "text_hash"):
        if left[key] and left[key] == right[key]:
            return key
    titles = left["title_words"] | right["title_words"]
    if titles and len(left["title_words"] & right["title_words"]) / len(titles) >= 0.85:
        return "near_title"
    if left["author"] and left["author"] == right["author"]:
        if SequenceMatcher(None, left["title"], right["title"]).ratio() >= 0.92:
            return "author_near_title"
    shingles = left["shingles"] | right["shingles"]
    if shingles and len(left["shingles"] & right["shingles"]) / len(shingles) >= 0.75:
        return "near_abstract"
    return None


def value(node):
    return display("".join(node.itertext())) if node is not None else ""


def extract_jats(raw, candidate, work):
    article = ET.fromstring(raw)
    if article.tag.startswith("{https://jats.nlm.nih.gov/"):
        ns = article.tag.split("}")[0] + "}"
        for node in article.iter():
            node.tag = node.tag.removeprefix(ns)
    meta = article.find("./front/article-meta")
    if meta is None or article.get("article-type") not in {"research-article", "brief-report", "methods-article"}:
        raise ValueError("not_jats_research_article")
    if article.get("{http://www.w3.org/XML/1998/namespace}lang", "en") != "en":
        raise ValueError("not_english_jats")
    ids = {node.get("pub-id-type"): value(node) for node in meta.findall("./article-id")}
    pmcid = candidate["pmcid"]
    jats_pmc = ids.get("pmc") or ids.get("pmcid") or ""
    if doi(ids.get("doi")) != doi(work["doi"]) or jats_pmc.removeprefix("PMC") != pmcid.removeprefix("PMC"):
        raise ValueError("jats_identity_mismatch")
    title = value(meta.find("./title-group/article-title"))
    if words(title) != words(work["title"]):
        raise ValueError("jats_title_mismatch")
    links = meta.findall("./permissions/license")
    if len(links) != 1:
        raise ValueError("ambiguous_jats_license")
    license_element = links[0]
    license_text = value(license_element)
    xlink = "{http://www.w3.org/1999/xlink}href"
    urls = {
        item.get(xlink, "").replace("http://", "https://").rstrip("/") + "/"
        for item in license_element.iter() if item.get(xlink)
    } & LICENSES.keys()
    if len(urls) != 1 or not license_text:
        raise ValueError("unconfirmed_jats_license")
    license_url = next(iter(urls))
    abstract_nodes = [
        node for node in meta.findall("./abstract")
        if node.get("abstract-type") in {None, "", "abstract"}
        and node.get("{http://www.w3.org/XML/1998/namespace}lang", "en") == "en"
    ]
    if len(abstract_nodes) != 1:
        raise ValueError("missing_or_ambiguous_abstract")
    abstract = abstract_nodes[0]
    if any(node.tag in {"permissions", "license", "table-wrap", "fig", "media"} for node in abstract.iter()):
        raise ValueError("abstract_extra_material_or_terms")
    # Абзацы и заголовки отделены пробелами, вложенная типографика сохранена текстом.
    blocks = [value(node) for node in abstract.iter() if node.tag in {"title", "p"}]
    text = display(" ".join(blocks)) if blocks else value(abstract)
    if not 80 <= len(text.split()) <= 600:
        raise ValueError("abstract_word_count")
    if len(re.findall(r"[.!?](?:\s|$)", text)) < 3 or re.search(r"�|\[Formula see text\]|\[Image:\s", text, re.I):
        raise ValueError("damaged_or_sparse_abstract")
    authors = []
    for author in meta.findall("./contrib-group/contrib"):
        if author.get("contrib-type") == "author":
            name = author.find("name")
            authors.append(" ".join(filter(None, [value(name.find("given-names")),
                                                   value(name.find("surname"))])) if name is not None
                           else value(author.find("collab")))
    authors = list(filter(None, authors))
    if not authors:
        raise ValueError("missing_attribution")
    return {
        "title": title, "text": text, "authors": authors, "pmcid": pmcid,
        "license": LICENSES[license_url], "license_url": license_url, "license_statement": license_text,
        "copyright_notice": value(meta.find("./permissions/copyright-statement")),
        "publisher": value(article.find("./front/journal-meta/publisher/publisher-name")),
        "journal": value(article.find("./front/journal-meta/journal-title-group/journal-title")),
        "raw_xml_sha256": sha256(raw),
        "abstract_xml_sha256": sha256(ET.tostring(abstract, encoding="utf-8")),
        "related_articles": [url for node in meta.findall("./related-article") if (url := node.get(xlink))],
    }


def openalex_eligibility(work, candidate):
    if not work:
        return "missing_openalex_doi"
    if not re.fullmatch(r"https://openalex.org/W\d+", work.get("id", "")):
        return "invalid_openalex_id"
    if work.get("language") != "en" or work.get("type") != "article" or work.get("is_retracted") is not False:
        return "openalex_type_language_or_retraction"
    if work.get("publication_year") not in range(2018, 2026):
        return "openalex_year"
    if doi(candidate.get("doi")) != doi(work.get("doi")):
        return "openalex_doi_mismatch"
    pmc = (work.get("ids") or {}).get("pmcid")
    if (pmc and pmc.rstrip("/").rsplit("/", 1)[-1].upper().removeprefix("PMC")
            != candidate["pmcid"].removeprefix("PMC")):
        return "openalex_pmcid_mismatch"
    location = work.get("primary_location") or {}
    topic = work.get("primary_topic") or {}
    if (location.get("source") or {}).get("type") != "journal" or not (topic.get("field") or {}).get("id"):
        return "openalex_venue_or_topic"
    return None


def document(source, work, group):
    pmcid = source["pmcid"]
    url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    doc_doi = doi(work["doi"])
    topic = work["primary_topic"]
    text_sha = digest(source["text"])
    return validate_document({
        "contract_version": "ner-v2-source-1", "document_id": work["id"].rsplit("/", 1)[-1],
        "openalex_id": work["id"], "doi": doc_doi, "revision_id": "sha256:" + text_sha,
        "title": source["title"], "text": source["text"], "text_sha256": text_sha,
        "year": work["publication_year"], "language": "en", "discipline": GROUPS[group][0],
        "topic": GROUPS[group][0], "subgroup": group, "source_url": url,
        "primary_topic": topic, "field": topic["field"], "subfield": topic.get("subfield"),
        "authors": source["authors"], "pmcid": pmcid, "license": source["license"],
        "source_reference": {
            "provider": "PubMed Central Cloud / publisher JATS", "source_url": url,
            "section": "front/article-meta/abstract", "doi_url": "https://doi.org/" + doc_doi,
            "api_url": source["acquisition"]["xml_url"], "openalex_updated_date": work.get("updated_date"),
            "source_archive_version": source["acquisition"]["archive_version"],
            "archive_metadata_url": source["acquisition"]["metadata_url"],
            "archive_metadata_sha256": source["acquisition"]["metadata_sha256"],
            "openalex_api_url": "https://api.openalex.org/works/" + work["id"].rsplit("/", 1)[-1],
            "text_sha256": text_sha, "transformation": "JATS text extraction; NFC and whitespace collapse",
            **{key: source[key] for key in ("license", "license_url", "license_statement",
                                           "copyright_notice", "publisher", "journal",
                                           "raw_xml_sha256", "abstract_xml_sha256", "related_articles")},
        },
    })


def epmc_get(session, url, **kwargs):
    for attempt in range(3):
        try:
            response = session.get(url, timeout=(8, 20), **kwargs)
            if response.status_code < 500 and response.status_code != 429:
                return response
        except requests.RequestException:
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)
    response.raise_for_status()


def load_pool(group, query, label, session, client):
    path = DATA / f"pool_{group}_{label}.json"
    if path.exists():
        pool = read_json(path)
        if pool["expression"] != query:
            raise ValueError("Query drift for " + str(path))
        return pool
    fields = re.sub(r'"[^"]+"', lambda match: "TITLE_ABS:" + match[0], query)
    params = {
        "query": f'{fields} AND OPEN_ACCESS:Y AND FIRST_PDATE:[2018-01-01 TO 2025-12-31]',
        "pageSize": 40, "format": "json", "resultType": "lite",
    }
    response = epmc_get(session, EPMC + "/search", params=params)
    response.raise_for_status()
    body = response.json()
    rows = body["resultList"]["result"]
    dois = list(dict.fromkeys(doi(row.get("doi")) for row in rows if row.get("doi") and row.get("pmcid")))
    before = len(client.requests)
    result = client.get("works", {"filter": OPENALEX_FILTER + "|".join(dois), "per_page": 50, "select": SELECT}) if dois else None
    works = {doi(work["doi"]): work for work in result["results"]} if result else {}
    pool = {
        "group": group, "label": label, "expression": query, "params": params,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(), "status": response.status_code,
        "hit_count": body["hitCount"], "response_sha256": sha256(response.content), "rows": rows,
        "openalex_requests": client.requests[before:], "openalex_works": works,
    }
    write_json(path, pool)
    return pool


def cloud_source(session, candidate, work, provenance):
    pmcid = candidate["pmcid"]
    listing = epmc_get(session, PMC_CLOUD, params={"list-type": 2, "prefix": pmcid + ".", "delimiter": "/"})
    listing.raise_for_status()
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    root = ET.fromstring(listing.content)
    versions = [node.text.rstrip("/") for node in root.findall("./s3:CommonPrefixes/s3:Prefix", ns)]
    citation_versions = []
    for version in versions:
        url = f"{PMC_CLOUD}/{version}/{version}.json"
        metadata = epmc_get(session, url)
        metadata.raise_for_status()
        record = metadata.json()
        if record.get("is_manuscript") is False:
            citation_versions.append((version, url, metadata, record))
    if len(citation_versions) != 1:
        return {"error": "missing_or_ambiguous_pmc_published_version"}
    version, metadata_url, metadata, record = citation_versions[0]
    xml_url = f"{PMC_CLOUD}/{version}/{version}.xml"
    provenance.update({"archive_version": version, "metadata_url": metadata_url, "xml_url": xml_url,
                       "metadata_sha256": sha256(metadata.content), "archive_license_code": record.get("license_code")})
    if (record.get("pmcid") != pmcid or doi(record.get("doi")) != doi(work["doi"])
            or record.get("is_pmc_openaccess") is not True or record.get("is_retracted") is not False):
        return {"error": "pmc_archive_identity_or_status"}
    if record.get("license_code") not in {"CC BY", "CC0", "CC 0"}:
        return {"error": "pmc_archive_license"}
    response = epmc_get(session, xml_url)
    provenance.update({"http_status": response.status_code,
                       "response_sha256": sha256(response.content)})
    if not response.ok:
        return {"error": f"jats_http_{response.status_code}"}
    try:
        return extract_jats(response.content, candidate, work)
    except (ValueError, ET.ParseError) as exc:
        return {"error": str(exc) if isinstance(exc, ValueError) else "invalid_jats"}


def cached_source(candidate, work, session):
    pmcid = candidate["pmcid"]
    if not re.fullmatch(r"PMC\d+", pmcid):
        return {"error": "invalid_pmcid"}
    path = SNAPSHOT / f"{pmcid}.json"
    if path.exists():
        previous = read_json(path)
        if previous.get("acquisition", {}).get("provider") == "PMC Cloud":
            return previous
    provenance = {"retrieved_utc": datetime.now(timezone.utc).isoformat(), "provider": "PMC Cloud"}
    source = cloud_source(session, candidate, work, provenance)
    source["acquisition"] = provenance
    temporary = path.with_suffix(".partial")
    write_json(temporary, source)
    temporary.replace(path)
    return source


def pool_rows(pools, *, allow_pending=False):
    seen = set()
    for pool in pools:
        group = pool["group"]
        for rank, candidate in enumerate(pool["rows"], 1):
            paper_doi = doi(candidate.get("doi"))
            pmcid = candidate.get("pmcid")
            work = pool["openalex_works"].get(paper_doi)
            ident = (group, paper_doi, pmcid)
            row = {"group": group, "pool": pool["label"], "rank": rank,
                   "title": candidate["title"], "doi": paper_doi, "pmcid": pmcid,
                   "openalex_id": work["id"] if work else None}
            if ident in seen:
                yield row | {"reason": "repeat_in_group_pool"}, None
                continue
            seen.add(ident)
            reason = ("missing_doi_or_pmcid" if not paper_doi or not pmcid
                      else openalex_eligibility(work, candidate))
            if reason:
                yield row | {"reason": reason}, None
                continue
            snapshot = SNAPSHOT / f"{pmcid}.json"
            if allow_pending and not snapshot.exists():
                yield row | {"reason": "pending"}, None
                continue
            source = read_json(snapshot)
            row["jats"] = source.get("acquisition")
            if source.get("error"):
                yield row | {"reason": source["error"]}, None
                continue
            yield row, document(source, work, group)


def pools_on_disk():
    return sorted((read_json(path) for path in DATA.glob("pool_*.json")),
                  key=lambda pool: (list(GROUPS).index(pool["group"]), pool["label"]))


def acquire_jats(pools):
    pending = {}
    for pool in pools:
        for row in pool["rows"]:
            pmcid = row.get("pmcid")
            work = pool["openalex_works"].get(doi(row.get("doi")))
            if pmcid and not openalex_eligibility(work, row):
                pending.setdefault(pmcid, (row, work))
    if len(pending) > 400:
        raise ValueError("JATS request budget exceeded")
    thread_state = local()
    sessions = []

    def fetch(entry):
        if not hasattr(thread_state, "session"):
            thread_state.session = requests.Session()
            sessions.append(thread_state.session)
        return cached_source(*entry, thread_state.session)

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            for _ in tqdm(executor.map(fetch, pending.values()), total=len(pending), desc="JATS evidence", unit="work"):
                pass
    finally:
        for session in sessions:
            session.close()


def acquire(group):
    client = OpenAlexClient()
    with requests.Session() as session:
        selected = [group] if group else list(GROUPS)
        pools, unavailable = [], []
        for key in tqdm(selected, desc="Catalog searches", unit="pool"):
            try:
                pools.append(load_pool(key, GROUPS[key][1], "base", session, client))
            except (requests.RequestException, RuntimeError) as exc:
                unavailable.append(key)
                print(f"Catalog request incomplete for {key}: {type(exc).__name__}", flush=True)
        for label, extra in query_extensions().items():
            if extra["group"] in selected:
                pools.append(load_pool(extra["group"], extra["expression"], label, session, client))
    acquire_jats(pools)
    rows = list(pool_rows(pools))
    for key in selected:
        print(key, Counter(row.get("reason", "eligible") for row, doc in rows if row["group"] == key))
    if unavailable:
        raise RuntimeError("Missing candidate pools: " + ", ".join(unavailable))


def historical():
    documents = {}
    provenance = {}
    for label, relative in HISTORICAL.items():
        path = ROOT / relative
        docs = read_documents(path)
        if len(docs) != int(label.removeprefix("Corpus")):
            raise ValueError("Historical corpus count changed: " + label)
        provenance[label] = {"path": relative, "sha256": file_sha256(path), "count": len(docs)}
        documents[label] = [pair_keys(doc) for doc in docs]
    return documents, provenance


def related_dois(doc):
    return {
        doi(link) for link in doc["source_reference"]["related_articles"]
        if re.match(r"^(?:https?://(?:dx\.)?doi\.org/|10\.\d+/)", link)
    }


def find_duplicate(doc, keys, prior):
    links = related_dois(doc)
    for other in prior:
        reason = overlap(keys, other)
        if reason or other["doi"] in links or keys["doi"] in other.get("related_dois", ()):
            return {"rule": reason or "related_jats_version", "document_id": other["id"]}
    return None


def review(group=None):
    found = []
    historical_keys, _ = historical()
    old = [keys for corpus in historical_keys.values() for keys in corpus]
    for row, doc in pool_rows(pools_on_disk(), allow_pending=True):
        if doc and (not group or doc["subgroup"] == group) and not find_duplicate(doc, pair_keys(doc), old):
            found.append({"group": row["group"], "rank": row["rank"], "pool": row["pool"],
                          "document_id": doc["document_id"], "title": doc["title"], "text": doc["text"]})
    write_json(DATA / "review_queue.json", found)
    print(Counter(item["group"] for item in found))


def measurements(docs):
    size_words = [len(doc["text"].split()) for doc in docs]
    size_tokens = [len(re.findall(TOKEN_PATTERN, doc["text"])) for doc in docs]

    def describe(counts):
        return {"total": sum(counts), "min": min(counts), "median": median(counts),
                "mean": round(mean(counts), 2), "p95": sorted(counts)[math.ceil(0.95 * len(counts)) - 1],
                "max": max(counts)}

    by_topic = {}
    for topic in dict.fromkeys(doc["topic"] for doc in docs):
        indices = [i for i, doc in enumerate(docs) if doc["topic"] == topic]
        by_topic[topic] = {
            "count": len(indices), "words": describe([size_words[i] for i in indices]),
            "tokens": describe([size_tokens[i] for i in indices]),
        }
    return {
        "document_count": len(docs), "words": describe(size_words), "tokens": describe(size_tokens),
        "tokenizer": {"name": "Unicode Regex v1", "pattern": TOKEN_PATTERN, "implementation": "Python 3.11 re",
                      "special_tokens": False},
        "words_histogram": {f"{low}-{high}": sum(low <= count <= high for count in size_words)
                            for low, high in [(80, 149), (150, 199), (200, 249), (250, 299), (300, 399), (400, 600)]},
        "per_year": dict(sorted(Counter(str(doc["year"]) for doc in docs).items())),
        "per_topic": by_topic, "per_subgroup": dict(Counter(doc["subgroup"] for doc in docs)),
        "licenses": dict(Counter(doc["license"] for doc in docs)),
    }


def verify_documents(docs):
    expected = Counter({group: 10 for group in GROUPS})
    if Counter(doc["subgroup"] for doc in docs) != expected:
        raise ValueError("Subgroup quotas failed")
    historical_keys, provenance = historical()
    if provenance["Corpus30"]["sha256"] != read_json(
        REPORT.parents[1] / "benchmark_3a5_9/PROFILE.json"
    )["corpus_sha256"]:
        raise ValueError("Corpus30 does not match frozen profile C")
    for doc in docs:
        validate_document(doc)
        ref = doc["source_reference"]
        if not (doc["revision_id"] == "sha256:" + digest(doc["text"])
                and doc["document_id"] == doc["openalex_id"].rsplit("/", 1)[-1]
                and doc["language"] == "en" and doc["topic"] == GROUPS[doc["subgroup"]][0]
                and doc["discipline"] == doc["topic"] and doc["year"] in range(2018, 2026)
                and ref["section"] == "front/article-meta/abstract"
                and doc["license"] == LICENSES.get(ref["license_url"])
                and 80 <= len(doc["text"].split()) <= 600):
            raise ValueError("Document contract failed: " + doc["document_id"])
        text = doc["text"]
        for match in re.finditer(TOKEN_PATTERN, text):
            validate_span(text, match.start(), match.end(), match[0])
    keys = [pair_keys(doc) | {"related_dois": related_dois(doc)} for doc in docs]
    for left, right in combinations(zip(docs, keys, strict=True), 2):
        if duplicate := find_duplicate(left[0], left[1], [right[1]]):
            raise ValueError(f"Corpus overlap: {left[0]['document_id']} {duplicate}")
    for doc, identity in zip(docs, keys, strict=True):
        for label, old in historical_keys.items():
            if duplicate := find_duplicate(doc, identity, old):
                raise ValueError(f"Historical overlap: {label} {doc['document_id']} {duplicate}")
    return provenance


def evidence(docs, pairs):
    indexed = {doc["document_id"]: doc for doc in docs}
    links = []
    for pair in pairs:
        spans = []
        for span in pair["spans"]:
            doc_id = span["document_id"]
            doc = indexed[doc_id]
            text = doc["text"]
            start, end = span["start"], span["end"]
            surface = text[start:end]
            validate_span(text, start, end, surface)
            if digest(surface) != span["surface_sha256"]:
                raise ValueError("Evidence surface changed: " + doc_id)
            spans.append({"document_id": doc_id, "topic": doc["topic"], "subgroup": doc["subgroup"],
                          "source_url": doc["source_url"], "revision_id": doc["revision_id"],
                          "start": start, "end": end, "surface_sha256": span["surface_sha256"]})
        if len(spans) != 2 or spans[0]["document_id"] == spans[1]["document_id"]:
            raise ValueError("Expected two distinct works per comparison")
        links.append({"axis": pair["axis"], "comparison": pair["comparison"], "evidence": spans})
    within = [a["subgroup"] for a, b in (link["evidence"] for link in links) if a["subgroup"] == b["subgroup"]]
    across = [[a["topic"], b["topic"]] for a, b in (link["evidence"] for link in links) if a["topic"] != b["topic"]]
    if set(within) != GROUPS.keys() or len(across) < 4 or {topic for pair in across for topic in pair} != {doc["topic"] for doc in docs}:
        raise ValueError("Insufficient thematic comparisons")
    return {"within_subgroups": sorted(set(within)), "between_topics": across, "examples": links}


def lock_row(doc):
    return {key: value for key, value in doc.items() if key != "text"} | {
        "word_count": len(doc["text"].split()),
        "token_count": len(re.findall(TOKEN_PATTERN, doc["text"])),
        "normalized_text_sha256": digest(" ".join(words(doc["text"]))),
    }


def make_outputs(target):
    decisions = read_json(REPORT / "review_decisions.json")
    pools = pools_on_disk()
    if {pool["group"] for pool in pools} != GROUPS.keys():
        raise ValueError("Missing registered candidate pools")
    accepted = {group: set(ids) for group, ids in decisions["accepted"].items()}
    manual = decisions["excluded"]
    docs, audit, selected_keys = [], [], []
    historical_keys, _ = historical()
    old = [keys for corpus in historical_keys.values() for keys in corpus]
    for row, doc in pool_rows(pools):
        if doc:
            group, doc_id = doc["subgroup"], doc["document_id"]
            identity = pair_keys(doc) | {"related_dois": related_dois(doc)}
            duplicate = find_duplicate(doc, identity, old)
            if duplicate:
                row |= {"reason": "historical_overlap", "duplicate_of": duplicate}
            elif doc_id in manual.get(group, {}):
                row |= {"reason": "content_review_excluded", "review_note": manual[group][doc_id]}
            elif doc_id in accepted[group]:
                docs.append(doc)
                selected_keys.append(identity)
                accepted[group].remove(doc_id)
                row["reason"] = "selected"
            else:
                row["reason"] = "eligible_not_selected"
                row["duplicate_of"] = find_duplicate(doc, identity, selected_keys)
                if row["duplicate_of"]:
                    row["reason"] = "overlap_with_earlier_selected"
                else:
                    del row["duplicate_of"]
        audit.append(row)
    if any(accepted.values()):
        raise ValueError(f"Manual selection missing from eligible pools: {accepted}")
    provenance = verify_documents(docs)
    thematic = evidence(docs, decisions["pairs"])
    corpus_path = target / "abstracts.jsonl"
    write_jsonl(corpus_path, docs)
    write_json(target / "document_lock.json", [lock_row(doc) for doc in docs])
    write_json(target / "corpus_statistics.json", measurements(docs))
    counts = Counter(row["reason"] for row in audit)
    identities = {row["doi"] or row["pmcid"] or row["title"] for row in audit}
    selected_identities = {doc["doi"] for doc in docs}
    write_json(target / "selection_audit.json", {
        "study": VERSION, "candidate_occurrences": len(audit),
        "unique_works_or_candidates": len(identities),
        "unique_unselected_candidates": len(identities - selected_identities),
        "selected_occurrences": counts["selected"], "excluded_occurrences": len(audit) - counts["selected"],
        "reason_counts": dict(sorted(counts.items())), "historical": provenance,
        "review_criteria": decisions["criteria"], "thematic_overlap": thematic, "candidates": audit,
    })
    write_json(target / "acquisition_provenance.json", {
        "study": VERSION, "seed": None, "providers": ["Europe PMC (discovery)", "OpenAlex (identity)",
                                                       "PubMed Central Cloud (publisher JATS)"],
        "acquisition_policy": "Unpaid API, one process, up to four JATS HTTP requests concurrently, fixed catalog snapshots",
        "query_extensions": query_extensions(),
        "transport_adjustments": [
            {"provider": "Europe PMC", "operation": "fullTextXML",
             "outcome": "read timeouts on probe", "replacement": "PMC Cloud versioned JATS"},
            {"provider": "Europe PMC", "operation": "pv_forecast catalog search",
             "original_expression": '("photovoltaic power" OR "solar power") AND ("power forecasting" OR "power prediction" OR "photovoltaic forecasting")',
             "outcome": "read timeouts", "replacement": GROUPS["pv_forecast"][1]},
        ],
        "pools": [{key: value for key, value in pool.items() if key not in {"rows", "openalex_works"}}
                  | {"returned": len(pool["rows"]), "local_snapshot_sha256": file_sha256(
                      DATA / f"pool_{pool['group']}_{pool['label']}.json")}
                  for pool in pools],
        "versions": {"python": ".".join(map(str, sys.version_info[:3])), "requests": requests.__version__,
                     "openalex_adapter_sha256": file_sha256(ROOT / "src/cgraph/ner_v2/openalex.py"),
                     "spec_sha256": file_sha256(REPORT / "SPEC.md"),
                     "source_contract_sha256": file_sha256(ROOT / "contracts/ner_v2/source.schema.json")},
    })
    write_json(target / "corpus_manifest.json", {
        "stage": VERSION, "state": "FROZEN_BEFORE_INFERENCE", "text_layer": "published abstracts only",
        "document_count": len(docs), "source_contract": "ner-v2-source-1", "seed": None,
        "documents": [{"document_id": doc["document_id"], "revision_id": doc["revision_id"]} for doc in docs],
        "text_file": (DATA / "abstracts.jsonl").relative_to(ROOT).as_posix(),
        "text_file_sha256": file_sha256(corpus_path),
        "report_sha256": {name: file_sha256(target / name) for name in REPORT_NAMES if name != "corpus_manifest.json"},
        "review_decisions_sha256": file_sha256(REPORT / "review_decisions.json"),
        "prepare_sha256": file_sha256(Path(__file__)),
    })


REPORT_NAMES = ("document_lock.json", "acquisition_provenance.json", "selection_audit.json",
                "corpus_statistics.json", "corpus_manifest.json")


def build():
    import tempfile

    with tempfile.TemporaryDirectory(dir=DATA) as temp:
        target = Path(temp)
        make_outputs(target)
        for name in ("abstracts.jsonl", *REPORT_NAMES):
            destination = (DATA if name == "abstracts.jsonl" else REPORT) / name
            if destination.exists() and destination.read_bytes() != (target / name).read_bytes():
                raise ValueError(f"Refusing to overwrite frozen file: {destination}")
        for name in ("abstracts.jsonl", *REPORT_NAMES):
            destination = (DATA if name == "abstracts.jsonl" else REPORT) / name
            if not destination.exists():
                (target / name).replace(destination)
    print("PASS: frozen outputs built from local candidate snapshots and reviewed selection")


def verify():
    manifest = read_json(REPORT / "corpus_manifest.json")
    if file_sha256(Path(__file__)) != manifest["prepare_sha256"] or file_sha256(
        REPORT / "review_decisions.json"
    ) != manifest["review_decisions_sha256"]:
        raise ValueError("Preparation program or review decision checksum mismatch")
    for name, expected in manifest["report_sha256"].items():
        if file_sha256(REPORT / name) != expected:
            raise ValueError("Report checksum mismatch: " + name)
    local_corpus = ROOT / manifest["text_file"]
    if file_sha256(local_corpus) != manifest["text_file_sha256"]:
        raise ValueError("Text checksum mismatch")
    docs = read_documents(local_corpus)
    if manifest["documents"] != [{"document_id": doc["document_id"], "revision_id": doc["revision_id"]}
                                 for doc in docs]:
        raise ValueError("Manifest document order or revision mismatch")
    if [lock_row(doc) for doc in docs] != read_json(REPORT / "document_lock.json"):
        raise ValueError("Document rows differ from lock")
    if measurements(docs) != read_json(REPORT / "corpus_statistics.json"):
        raise ValueError("Statistics differ from text")
    verify_documents(docs)
    decisions = read_json(REPORT / "review_decisions.json")
    if evidence(docs, decisions["pairs"]) != read_json(REPORT / "selection_audit.json")["thematic_overlap"]:
        raise ValueError("Evidence differs from audited offsets")
    print("PASS: 100 documents, hashes, distributions, deduplication, history and evidence offsets")


def restore():
    lock = read_json(REPORT / "document_lock.json")
    manifest = read_json(REPORT / "corpus_manifest.json")
    dest = ROOT / manifest["text_file"]
    if dest.exists():
        verify()
        return
    docs = []
    with requests.Session() as session:
        for row in tqdm(lock, desc="Restore licensed abstracts", unit="work"):
            response = epmc_get(session, row["source_reference"]["api_url"])
            response.raise_for_status()
            source = extract_jats(response.content, row, row)
            if (digest(source["text"]) != row["text_sha256"] or source["license"] != row["license"]
                    or source["authors"] != row["authors"]
                    or source["license_statement"] != row["source_reference"]["license_statement"]
                    or source["copyright_notice"] != row["source_reference"]["copyright_notice"]):
                raise ValueError("Live source differs from locked abstract or attribution: " + row["document_id"])
            doc = {key: value for key, value in row.items()
                   if key not in {"word_count", "token_count", "normalized_text_sha256"}}
            docs.append(doc | {"text": source["text"]})
    verify_documents(docs)
    import tempfile

    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as temp:
        temporary = Path(temp) / "abstracts.jsonl"
        write_jsonl(temporary, docs)
        if file_sha256(temporary) != manifest["text_file_sha256"]:
            raise ValueError("Restored file differs from lock")
        temporary.replace(dest)
    verify()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["acquire", "review", "build", "verify", "restore"])
    parser.add_argument("--group", choices=GROUPS)
    args = parser.parse_args()
    if args.command == "acquire":
        acquire(args.group)
    elif args.command == "review":
        review(args.group)
    else:
        globals()[args.command]()


if __name__ == "__main__":
    main()
