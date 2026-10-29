"""Fresh-process, offline pretrained-only DEV extraction with process-wide access guard."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import time

START, CPU = time.perf_counter(), time.process_time()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                  TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="", PYTHONUTF8="1")


def install_guard(gold_mode):
    reads = set()

    def audit(event, args):
        if event == "socket.connect":
            raise PermissionError("Inference network forbidden")
        if event == "open" and isinstance(args[0], (str, bytes)):
            path = str(args[0]).replace("\\", "/").lower()
            if (any(p in path for p in ("/labels/", "/gold/", "/holdout/", "/calibration/", "/re_model/"))
                    or path.endswith("/ontology.v1.0.json")
                    or path.endswith(("cgraph/labels.py", "cgraph/relations.py"))
                    or ("diagnostic_entities" in path and not gold_mode)):
                raise PermissionError("Gold/closed-split/fitted-model access forbidden")
            if "/data/inputs/" in path or "diagnostic_entities" in path:
                reads.add(path)
        if event == "import" and args[0] in {"cgraph.labels", "cgraph.relations", "cgraph.evaluation", "cgraph.zeroshot_evaluation"}:
            raise PermissionError("Annotation/training/evaluation import forbidden during extraction")
    sys.addaudithook(audit)
    return reads


class FixedDiagnosticEntities:
    calls = encoder_calls = max_sequence_tokens = load_ms = 0

    def __init__(self, cfg):
        from cgraph.common import file_sha256, fingerprint, read_json
        path = Path("artifacts/c_diag_002/diagnostic_entities.json")
        self.data = read_json(path)
        if self.data["config_sha256"] != fingerprint(cfg) or self.data["manifest_sha256"] != file_sha256(cfg["manifest"]):
            raise ValueError("Gold projection belongs to another protocol")
        self.identity = dict(purpose=self.data["purpose"], entity_export_sha256=file_sha256(path))

    def predict(self, chunk):
        return [dict(m) for m in self.data["chunks"][chunk["chunk_id"]]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["control", "chem", "battery", "union", "gold"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reads = install_guard(args.variant == "gold")
    from cgraph.common import file_sha256, fingerprint, read_json, write_json
    from cgraph.inputs import load_inputs, load_ontology
    from cgraph.ner import ScientificNER
    from cgraph.diagnostic_ner import PretrainedScientificNER, UnionNER
    from cgraph.zeroshot_nli import EntailmentRE
    from cgraph.zeroshot_pipeline import extract_document
    from cgraph.validation import validate_prediction
    import psutil
    import torch

    cfg = read_json("config/c_diag_002.json")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new/empty")
    output.mkdir(parents=True, exist_ok=True)
    docs, ontology = load_inputs(cfg), load_ontology(cfg)
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    if args.variant == "gold":
        ner = FixedDiagnosticEntities(cfg)
    else:
        models = []
        if args.variant in {"control", "union"}:
            models.append(ScientificNER(cfg))
        for alias in ("chem", "battery"):
            if args.variant in {alias, "union"}:
                models.append(PretrainedScientificNER(cfg["scientific_ner"][alias], cfg["diagnostic_protocol"]["model_files_contract"]))
        ner = models[0] if len(models) == 1 else UnionNER(models)
    re_model = EntailmentRE(cfg, ontology)
    print(f"{args.variant}: pretrained checkpoints loaded on CPU", flush=True)
    for doc in docs:
        pred = extract_document(doc, ner, re_model, ontology, cfg)
        pred.update(architecture_id="C-DIAG-002", variant=args.variant,
                    mode="gold_entities_re_diagnostic" if args.variant == "gold" else "end_to_end")
        validate_prediction(pred)
        write_json(output / f"{doc['document_code']}.json", pred)
        print(f"{args.variant}/{doc['document_code']}: " + str({key: sum(len(c[key]) for c in pred["chunks"])
              for key in ("entities", "candidate_pairs", "raw_proposals", "assertions")}), flush=True)
        if psutil.Process().memory_info().peak_wset >= 32 * 1024**3:
            raise MemoryError("32 GiB process memory budget exceeded")
    mem = psutil.Process().memory_info()
    files = sorted(set(p.as_posix() for p in Path("src/cgraph").glob("*.py")
                       if p.name not in {"labels.py", "relations.py"}) |
                   {"scripts/run_c_diag_002.py", "config/c_diag_002.json", "reports/C_DIAG_002_PROTOCOL.md"})
    run = dict(architecture_id="C-DIAG-002", variant=args.variant,
               mode="gold_entities_re_diagnostic" if args.variant == "gold" else "end_to_end",
               documents=[d["document_code"] for d in docs],
               wall_ms=1000 * (time.perf_counter() - START), cpu_ms=1000 * (time.process_time() - CPU),
               peak_ram_bytes=mem.peak_wset, peak_ram_method="Windows PeakWorkingSetSize; full fresh process",
               ner_model=ner.identity, relation_model=re_model.identity,
               ner_load_ms=ner.load_ms, re_load_ms=re_model.load_ms,
               ner_calls=ner.calls, ner_encoder_calls=ner.encoder_calls, re_calls=re_model.calls,
               re_encoder_calls=re_model.encoder_calls, re_score_summary=re_model.score_summary,
               ner_max_sequence_tokens=ner.max_sequence_tokens, re_max_sequence_tokens=re_model.max_sequence_tokens,
               scientific_output_counts={k: dict(source_label_counts=m.source_label_counts,
                                                 unsupported_window_spans=m.unsupported_spans)
                                         for k, m in enumerate(getattr(ner, "models", [ner]))
                                         if hasattr(m, "source_label_counts")},
               config_sha256=fingerprint(cfg), config_file_sha256=file_sha256("config/c_diag_002.json"),
               protocol_sha256=file_sha256("reports/C_DIAG_002_PROTOCOL.md"),
               source_manifest_sha256=file_sha256(cfg["manifest"]), dependency_lock_sha256=file_sha256("uv.lock"),
               source_code_files_sha256={p: file_sha256(p) for p in files},
               prediction_files_sha256={d["document_code"]: file_sha256(output / f"{d['document_code']}.json") for d in docs},
               code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               code_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
               runtime={p: importlib.metadata.version(p) for p in ("torch", "transformers", "gliner", "glirel", "flair", "tokenizers", "psutil")},
               torch_threads=torch.get_num_threads(), training_performed=False, llm_calls=0,
               gold_in_inference=args.variant == "gold", gold_relations_in_inference=False,
               regime="Fresh process CPU float32; includes imports/loading/hashing/serialization; excludes downloads/probes/evaluation")
    write_json(output / "run.json", run)
    write_json(output / "access_audit.json", dict(completed=True, run_sha256=file_sha256(output / "run.json"),
               guard_script_sha256=file_sha256(__file__), source_and_diagnostic_reads=sorted(reads),
               gold_entities_permitted=args.variant == "gold", gold_relation_access=False, network_access=False,
               annotation_training_evaluation_imported=bool({"cgraph.labels", "cgraph.relations", "cgraph.evaluation", "cgraph.zeroshot_evaluation"} & sys.modules.keys())))
    print({k: run[k] for k in ("variant", "wall_ms", "cpu_ms", "peak_ram_bytes")}, flush=True)


if __name__ == "__main__":
    main()
