"""Pinned downloads and offline runtime preparation. Never constructs a NER model."""
import importlib.metadata
import os
import shutil
import time
from pathlib import Path

import requests

from cgraph.common import file_sha256, read_json, write_json
from .progress import progress

MODEL_ROOT = Path("artifacts/ner_v2/models")
LOCK_PATH = Path("contracts/ner_v2/model_lock.json")


def _matches(path, spec):
    if not path.is_file() or path.stat().st_size != spec["size"]:
        return False
    if spec.get("sha256"):
        return file_sha256(path) == spec["sha256"]
    import hashlib
    raw = path.read_bytes()
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == spec["git_blob_sha"]


def _download(repo, name, spec, destination):
    if _matches(destination, spec):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Reuse historical files only after checking the pinned upstream digest.
    for candidate in Path("artifacts/models").glob(f"*/{name}"):
        if _matches(candidate, spec):
            shutil.copyfile(candidate, destination)
            return
    url = f"https://huggingface.co/{repo['repo_id']}/resolve/{repo['revision']}/{name}"
    temporary = destination.with_name(destination.name + ".partial")
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(20, 60)) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream, progress(total=spec["size"], desc=name, unit="B",
                        unit_scale=True, unit_divisor=1024, position=1, leave=False) as downloaded:
                    for block in response.iter_content(4 * 1024 * 1024):
                        stream.write(block)
                        downloaded.update(len(block))
            if not _matches(temporary, spec):
                raise ValueError(f"Downloaded file digest mismatch: {repo['repo_id']}/{name}")
            temporary.replace(destination)
            return
        except requests.RequestException:
            if attempt == 2:
                raise RuntimeError(f"Download failed: {repo['repo_id']}/{name}") from None
            time.sleep(2 ** attempt)


def _link_or_copy(source, destination):
    if destination.exists():
        if file_sha256(source) != file_sha256(destination):
            raise ValueError(f"Runtime file mismatch: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def prepare_models(lock_path=LOCK_PATH, root=MODEL_ROOT):
    lock, root = read_json(lock_path), Path(root)
    required = sum(f["size"] for r in lock["repositories"].values() for f in r["files"].values())
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    # Reserve enough for downloads, partial files and copy fallback for runtime weights.
    if free < required * 3 + 1024 ** 3:
        raise OSError("Insufficient free disk space for NER v2 models")
    print(f"Disk free: {free} bytes; pinned downloads: {required} bytes", flush=True)
    model_files = [(key, repo, name, spec) for key, repo in lock["repositories"].items()
                   for name, spec in repo["files"].items()]
    with progress(model_files, desc="Model files", unit="file") as files:
        for key, repo, name, spec in files:
            files.set_postfix_str(f"{key}/{name}")
            _download(repo, name, spec, root / key / name)
    for size, spec in progress(lock["models"].items(), desc="Offline runtimes", unit="model"):
        checkpoint = root / spec["checkpoint_key"]
        backbone = root / spec["backbone_key"]
        runtime = root / f"runtime_{size}"
        runtime.mkdir(parents=True, exist_ok=True)
        for name in lock["repositories"][spec["backbone_key"]]["files"]:
            if name != "README.md":
                _link_or_copy(backbone / name, runtime / name)
        config = read_json(checkpoint / "gliner_config.json")
        config["encoder_config"] = read_json(backbone / "config.json")
        write_json(runtime / "gliner_config.json", config)
        _link_or_copy(checkpoint / spec["weights"], runtime / spec["weights"])
    inventory = verify_models(lock_path, root)
    write_json("reports/ner_v2/model_inventory.json", inventory)
    return inventory


def verify_models(lock_path=LOCK_PATH, root=MODEL_ROOT):
    """Hash every checkpoint, tokenizer and runtime file; no torch deserialization."""
    lock, root = read_json(lock_path), Path(root)
    files = {}
    model_files = [(key, name, spec) for key, repo in lock["repositories"].items()
                   for name, spec in repo["files"].items()]
    with progress(model_files, desc="Verify model files", unit="file") as checked:
        for key, name, spec in checked:
            checked.set_postfix_str(f"{key}/{name}")
            path = root / key / name
            if not _matches(path, spec):
                raise ValueError(f"Missing or corrupt pinned file: {path}")
            files[f"{key}/{name}"] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    for size, spec in progress(lock["models"].items(), desc="Verify offline runtimes", unit="model"):
        runtime = root / f"runtime_{size}"
        expected = read_json(root / spec["checkpoint_key"] / "gliner_config.json")
        expected["encoder_config"] = read_json(root / spec["backbone_key"] / "config.json")
        if read_json(runtime / "gliner_config.json") != expected:
            raise ValueError("Offline encoder config differs from locked config")
        for name, file_spec in lock["repositories"][spec["backbone_key"]]["files"].items():
            if name != "README.md" and not _matches(runtime / name, file_spec):
                raise ValueError(f"Offline tokenizer/config mismatch: {size}/{name}")
        weight_spec = lock["repositories"][spec["checkpoint_key"]]["files"][spec["weights"]]
        if not _matches(runtime / spec["weights"], weight_spec):
            raise ValueError(f"Offline weights mismatch: {size}")
        files[f"runtime_{size}/gliner_config.json"] = {
            "bytes": (runtime / "gliner_config.json").stat().st_size,
            "sha256": file_sha256(runtime / "gliner_config.json"),
        }
    versions = {name: importlib.metadata.version(name) for name in (
        "gliner", "torch", "transformers", "huggingface-hub", "tokenizers", "sentencepiece",
        "safetensors", "spacy", "en-core-web-sm", "numpy", "requests", "pytest", "tqdm",
    )}
    return {"status": "files_verified_no_model_loaded", "files": files, "libraries": versions}


def load_detector(size, root=MODEL_ROOT):
    """Future benchmark only. Callers must explicitly authorize inference in the CLI."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from gliner import GLiNER
    path = Path(root) / f"runtime_{size}"
    return GLiNER.from_pretrained(str(path), map_location="cpu", local_files_only=True, strict=True)


def verify_tokenizers(settings):
    """Offline dependency/tokenizer smoke check on synthetic text; no GLiNER instance."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from gliner.config import GLiNERConfig
    from gliner.data_processing.processor import UniEncoderSpanProcessor
    from transformers import AutoTokenizer
    from .detection import label_descriptions
    descriptions = label_descriptions(read_json(settings["ontology"]))
    result = {}
    for size in progress(("large", "medium", "small"), desc="Verify tokenizers", unit="model"):
        path = Path(settings["model_root"]) / f"runtime_{size}"
        config = GLiNERConfig(**read_json(path / "gliner_config.json"))
        # Keep the pinned DeBERTa Unigram/Metaspace tokenizer unchanged. In 5.1.0,
        # the broad large-vocabulary heuristic may emit an unrelated Mistral warning.
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        tokenizer.add_tokens(["[FLERT]", config.ent_token, config.sep_token], special_tokens=True)
        processor = UniEncoderSpanProcessor(config, tokenizer, None)
        text = ("Scientific models analyze cells and data using careful experimental methods. " * 6).strip()
        words = [item[0] for item in processor.words_splitter(text)]
        inputs, _ = processor.prepare_inputs([words], list(descriptions))
        ids = tokenizer(inputs, is_split_into_words=True, truncation=False)["input_ids"][0]
        if len(ids) > min(settings["max_input_tokens"], config.encoder_config.max_position_embeddings):
            raise ValueError("Synthetic tokenizer check exceeds token budget")
        result[size] = {"tokenizer_class": type(tokenizer).__name__, "backend": type(tokenizer.backend_tokenizer.model).__name__,
                        "tokens_in_synthetic_60word_window": len(ids),
                        "encoder_max_position_embeddings": config.encoder_config.max_position_embeddings,
                        "max_len": config.max_len, "weights_loaded": False}
    write_json("reports/ner_v2/offline_tokenizer_check.json", result)
    return result
