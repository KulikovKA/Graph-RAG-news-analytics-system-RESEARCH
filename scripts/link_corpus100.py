"""Поиск seeds в опубликованном поколении Corpus100 без подключения к Neo4j."""
from argparse import ArgumentParser
import json
from pathlib import Path
import re

from tokenizers import Tokenizer

from benchmark_contextual_shared_concepts import MODEL as DEFAULT_MODEL
from contextual_shared_concepts import DocumentWindows, fit_window
from corpus_graph_v2 import OUTPUT as GRAPH_ROOT, stored_inputs
from shared_concepts_v2_pipeline import RECIPE, RESEARCH_GRAPH_NAMESPACE, REVISION, WINDOW, load_checked
from cgraph.entity_linking_v2 import NAMESPACE, EntityLinkerV2, LinkingPolicy
from cgraph.ner_v2.io import validate_span
from cgraph.query_analysis import QueryAnalyzer
from cgraph.shared_concepts_v2 import MODEL, digest


class QueryContexts:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, query):
        text = query["text"]
        windows = DocumentWindows(text)
        result = []
        for mention in query["mentions"]:
            name, start, end = (mention[key] for key in ("surface_text", "start", "end"))
            validate_span(text, start, end, name)
            left, right = windows.window(WINDOW, start, end)
            lo, hi = fit_window(self.tokenizer, text, name, left, right, start, end)
            result.append(f"{name}. {text[lo:hi]}")
        return result


def load_generation(graph_root: Path):
    current_path = graph_root / "CURRENT.json"
    current = load_checked(current_path)
    request = current["request"]
    if not re.fullmatch(r"[0-9a-f]{64}", request):
        raise ValueError("Invalid graph generation request ID")
    folder = graph_root / "generations" / request
    inputs = stored_inputs(folder, request)
    catalog = load_checked(folder / "catalog" / "catalog.json")
    verification = load_checked(folder / "verification.json")
    recipe = inputs.metadata["recipe"]
    fingerprints = {entity.entity_id: entity.fingerprint for entity in inputs.entities}
    documents = inputs.source["documents"]
    if (NAMESPACE != RESEARCH_GRAPH_NAMESPACE or inputs.metadata["research_graph"] != NAMESPACE
            or catalog["generation"] != current["generation"] or catalog["status"] != "complete"
            or verification["status"] != "PASS" or fingerprints != catalog["fingerprints"]
            or len(fingerprints) != len(inputs.entities)
            or fingerprints.keys() != inputs.source["resolution"]["entities"].keys()
            or any(entity.document_id not in documents for entity in inputs.entities)):
        raise ValueError("Entity bundle differs from the published Corpus100 generation")
    recipe_hash = digest(recipe)
    if (recipe["model"] != MODEL or recipe["revision"] != REVISION or recipe["tokenizer_revision"] != REVISION
            or recipe["window"] != WINDOW or recipe["embedding"] != RECIPE
            or not {"tokenizer.json", "model.safetensors", "config.json"} <= recipe["model_files_sha256"].keys()
            or any(entity.recipe_hash != recipe_hash for entity in inputs.entities)):
        raise ValueError("Corpus100 vectors do not match the supported E5/S1 recipe")
    if load_checked(current_path) != current:
        raise ValueError("Corpus100 CURRENT changed during loading, retry with the new generation")
    snapshot = dict(namespace=NAMESPACE, generation=current["generation"], request=request,
                    payload_sha256=current["payload_sha256"], recipe_sha256=recipe_hash,
                    entity_index_sha256=digest(fingerprints), entity_count=len(inputs.entities),
                    document_count=len(documents))
    return inputs, snapshot


def build_linker(inputs, model_path: Path):
    tokenizer = model_path / "tokenizer.json"
    model = dict(root=str(model_path), tokenizer=str(tokenizer), files=[
        dict(path=str(model_path / name), sha256=digest)
        for name, digest in inputs.metadata["recipe"]["model_files_sha256"].items()
    ])
    raw_tokenizer = Tokenizer.from_file(str(tokenizer))
    raw_tokenizer.no_padding()
    raw_tokenizer.no_truncation()
    return EntityLinkerV2(inputs.entities, model, QueryContexts(raw_tokenizer))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--graph-root", type=Path, default=GRAPH_ROOT)
    parser.add_argument("--e5-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--threshold-name", type=float, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--threshold-score", type=float)
    args = parser.parse_args()
    policy = LinkingPolicy(args.threshold_name, args.alpha, args.top_k, args.threshold_score)
    inputs, snapshot = load_generation(args.graph_root)
    linker = build_linker(inputs, args.e5_model)
    result = linker.link(QueryAnalyzer().analyze(args.question), policy)
    print(json.dumps(result | {"graph": snapshot}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
