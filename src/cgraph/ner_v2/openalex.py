"""Small, seeded OpenAlex corpus acquisition; no PDFs or downstream inference."""
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from cgraph.common import file_sha256, fingerprint, sha256, write_json, read_json
from .io import validate_document, write_jsonl
from .progress import progress

API_BASE = "https://api.openalex.org"
DISCIPLINES = (
    ("Computer Science / Machine Learning", "Computer Science", "Computer Science"),
    ("Medicine", "Medicine", "Medicine"),
    ("Physics", "Physics", "Physics and Astronomy"),
    ("Biology", "Biology", "Biochemistry, Genetics and Molecular Biology"),
    ("Economics", "Economics", "Economics, Econometrics and Finance"),
    ("History", "History", "History"),
)
SELECT = "id,doi,title,publication_year,language,type,primary_topic,primary_location,abstract_inverted_index,updated_date"


class OpenAlexClient:
    def __init__(self, session=None):
        self.session = session or requests.Session()
        self.api_key = os.environ.get("OPENALEX_API_KEY")
        self.requests = []

    def get(self, endpoint, params=None):
        public = dict(params or {})
        private = dict(public)
        if self.api_key:
            private["api_key"] = self.api_key
        for attempt in range(3):
            time.sleep(0.25 if attempt == 0 else 2 ** attempt)
            try:
                response = self.session.get(f"{API_BASE}/{endpoint}", params=private, timeout=(10, 30))
            except requests.RequestException:
                if attempt < 2:
                    continue
                raise RuntimeError(f"OpenAlex unavailable: {endpoint} (network failure)") from None
            self.requests.append({"endpoint": endpoint, "params": public, "status": response.status_code})
            if response.status_code == 429 or response.status_code >= 500:
                wait = response.headers.get("Retry-After", "0")
                try:
                    seconds = max(0, float(wait))
                except ValueError:
                    seconds = 0
                if seconds > 30 or attempt == 2:
                    raise RuntimeError(f"OpenAlex unavailable: {endpoint} (HTTP {response.status_code})")
                time.sleep(seconds)
                continue
            if response.status_code != 200:
                # Never include response URLs, bodies or exception text: they may echo a key.
                raise RuntimeError(f"OpenAlex unavailable: {endpoint} (HTTP {response.status_code})")
            try:
                return response.json()
            except ValueError:
                raise RuntimeError(f"OpenAlex invalid JSON: {endpoint}") from None
        raise RuntimeError("OpenAlex retry budget exhausted")


def reconstruct_abstract(index):
    """Strict positional reconstruction, following the source repository's safeguards."""
    if not isinstance(index, dict) or not index:
        raise ValueError("Missing abstract inverted index")
    positions = {}
    for word, offsets in index.items():
        if not isinstance(word, str) or not word.strip() or not isinstance(offsets, list) or not offsets:
            raise ValueError("Invalid abstract token")
        for offset in offsets:
            if type(offset) is not int or not 0 <= offset < 20000 or offset in positions:
                raise ValueError("Invalid or duplicate abstract position")
            positions[offset] = word
    if not positions or set(positions) != set(range(len(positions))):
        raise ValueError("Abstract position gap")
    text = " ".join(positions[i] for i in range(len(positions)))
    if not text.strip():
        raise ValueError("Empty abstract")
    return text


def discover_fields(client):
    fields = []
    for discipline, query, expected in progress(DISCIPLINES, desc="Discover OpenAlex fields", unit="field"):
        result = client.get("fields", {"search": query, "per_page": 25})
        matches = [x for x in result["results"] if x["display_name"] == expected]
        if discipline == "History" and not matches:
            # History is an OpenAlex subfield; discover its parent field through the API.
            subs = client.get("subfields", {"search": "History", "per_page": 25})
            candidates = [x for x in subs["results"] if x["display_name"] == "History"]
            if len(candidates) != 1:
                raise ValueError("Cannot uniquely discover OpenAlex History subfield")
            sub = candidates[0]
            field = client.get("fields/" + sub["field"]["id"].rsplit("/", 1)[-1])
            fields.append({"discipline": discipline, "field": {"id": field["id"], "display_name": field["display_name"]},
                           "subfield": {"id": sub["id"], "display_name": sub["display_name"]},
                           "filter": "primary_topic.subfield.id:" + sub["id"].rsplit("/", 1)[-1]})
        else:
            if len(matches) != 1:
                raise ValueError(f"Cannot uniquely discover field: {discipline}")
            field = matches[0]
            fields.append({"discipline": discipline, "field": {"id": field["id"], "display_name": field["display_name"]},
                           "filter": "primary_topic.field.id:" + field["id"].rsplit("/", 1)[-1]})
    return fields


def select_works(candidates, used, seed):
    pool = []
    for work in sorted(candidates, key=lambda w: w["id"]):
        try:
            text = reconstruct_abstract(work.get("abstract_inverted_index"))
        except ValueError:
            continue
        topic = work.get("primary_topic") or {}
        source = (work.get("primary_location") or {}).get("source") or {}
        if (work["id"] not in used and work.get("language") == "en" and
                work.get("type") == "article" and work.get("title") and topic.get("id") and
                source.get("type") in {"journal", "conference"} and 80 <= len(text.split()) <= 600):
            pool.append(work)
    random.Random(seed).shuffle(pool)
    chosen, topics = [], set()
    for work in pool:
        topic = work["primary_topic"]["id"]
        if topic in topics:
            continue
        topics.add(topic)
        chosen.append(work)
        if len(chosen) == 2:
            return chosen
    raise ValueError("Fewer than two valid articles with different primary topics")


def source_record(work, field):
    text = reconstruct_abstract(work["abstract_inverted_index"])
    wid = work["id"].rsplit("/", 1)[-1]
    if not re.fullmatch(r"W[0-9]+", wid):
        raise ValueError("Invalid OpenAlex work ID")
    digest = sha256(text.encode("utf-8"))
    record = {"contract_version": "ner-v2-source-1", "document_id": wid, "openalex_id": work["id"],
              "revision_id": "sha256:" + digest, "text_sha256": digest,
              "text": text, "title": work["title"], "year": work["publication_year"],
              "doi": work.get("doi"), "language": work["language"], "discipline": field["discipline"],
              "field": field["field"], "subfield": (work["primary_topic"].get("subfield")),
              "primary_topic": work["primary_topic"], "source_url": work["id"],
              "source_reference": {"provider": "openalex", "source_url": work["id"],
                                   "api_url": f"{API_BASE}/works/{wid}", "doi": work.get("doi"),
                                   "landing_page_url": (work.get("primary_location") or {}).get("landing_page_url"),
                                   "section": "abstract", "source_updated_date": work.get("updated_date"),
                                   "text_sha256": digest}}
    return validate_document(record)


def prepare_corpus(output="data/ner_v2/openalex_12.jsonl", *, seed=42,
                   snapshot="contracts/ner_v2/openalex_snapshot.json", refresh=False):
    snapshot = Path(snapshot)
    if snapshot.exists() and not refresh:
        saved = read_json(snapshot)
        if saved["seed"] != seed:
            raise ValueError("Seed differs from frozen snapshot; request refresh explicitly")
    else:
        client = OpenAlexClient()
        fields = discover_fields(client)
        used, selected, pools = set(), [], []
        for field in progress(fields, desc="Fetch OpenAlex corpus", unit="field"):
            params = {"filter": field["filter"] + ",has_abstract:true,language:en,type:article",
                      "sample": 30, "seed": seed, "per_page": 30, "select": SELECT}
            payload = client.get("works", params)
            raw = payload["results"]
            # Retain full candidate responses locally for audit; never record authenticated URLs.
            write_json(f"artifacts/ner_v2/openalex/{field['discipline'].split('/')[0].strip()}.json", payload)
            choices = select_works(raw, used, seed)
            for work in choices:
                if work["primary_topic"]["field"]["id"] != field["field"]["id"]:
                    raise ValueError("Selected work has wrong field")
                if field.get("subfield") and work["primary_topic"]["subfield"]["id"] != field["subfield"]["id"]:
                    raise ValueError("Selected work has wrong subfield")
                used.add(work["id"])
                selected.append({"field": field, "work": work})
            pools.append({"discipline": field["discipline"], "params": params,
                          "candidate_ids": [w["id"] for w in raw], "response_fingerprint": fingerprint(payload)})
        saved = {"seed": seed, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                 "policy": "sample30-en-article-journal-conference-80to600words-distinct-primary-topics-sort-id-shuffle-v2",
                 "fields": fields, "pools": pools, "selected": selected, "requests": client.requests}
        write_json(snapshot, saved)
    records = [source_record(item["work"], item["field"])
               for item in progress(saved["selected"], desc="Prepare abstracts", unit="doc")]
    if len(records) != 12 or len({r["document_id"] for r in records}) != 12:
        raise ValueError("Corpus must contain twelve unique OpenAlex works")
    for discipline, _, _ in DISCIPLINES:
        subset = [r for r in records if r["discipline"] == discipline]
        if len(subset) != 2 or len({r["primary_topic"]["id"] for r in subset}) != 2:
            raise ValueError("Corpus discipline/topic balance mismatch")
    write_jsonl(output, records)
    write_json("reports/ner_v2/corpus_manifest.json", {
        "status": "prepared_no_inference", "seed": seed, "snapshot_sha256": file_sha256(snapshot),
        "jsonl_sha256": file_sha256(output), "jsonl": str(output), "fields": saved["fields"],
        "selection_policy": saved["policy"], "retrieved_at": saved["retrieved_at"],
        "documents": [{k: r[k] for k in ("document_id", "discipline", "title", "year", "doi", "field", "primary_topic", "text_sha256")}
                      for r in records]})
    return records
