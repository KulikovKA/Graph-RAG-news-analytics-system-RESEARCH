"""Локальные адаптеры и учёт токенов для Stage 3A.5.8."""

from importlib.metadata import distributions
import math
import os
from pathlib import Path
import platform
from time import perf_counter

from frozen_pool import HERE, file_sha, read_json


REGISTRY = read_json(HERE / "MODEL_REGISTRY.json")
DEFAULT_MODEL_ROOT = Path.home() / "Desktop/embedding-models/rerankers"
MINIMUM = {"torch": (2, 11), "transformers": (5, 7), "sentence-transformers": (5, 4, 1)}
QWEN_PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QWEN_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
MXBAI_PREFIX = (
    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\nquery: "
)
MXBAI_SUFFIX = (
    "\nYou are a search relevance expert who evaluates how well documents match search queries. For each "
    "query-document pair, carefully analyze the semantic relationship between them, then provide your "
    "binary relevance judgment (0 for not relevant, 1 for relevant).\nRelevance:<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def environment():
    packages = ("torch", "transformers", "sentence-transformers", "tokenizers", "huggingface-hub",
                "safetensors", "sentencepiece", "numpy", "psutil", "tqdm", "jinja2",
                "scikit-learn", "scipy", "regex", "packaging")
    versions = {dist.metadata["Name"].lower().replace("_", "-"): dist.version
                for dist in distributions() if dist.metadata["Name"]}
    problems = [f"{name} is missing" for name in packages if name not in versions]
    for name, minimum in MINIMUM.items():
        if name in versions:
            from packaging.version import Version

            if Version(versions[name]).release[:len(minimum)] < minimum:
                problems.append(f"{name} {versions[name]} is incompatible; requires >= {'.'.join(map(str, minimum))}")
    if platform.python_version_tuple()[:2] != ("3", "11"):
        problems.append("The reproducible runtime uses Python 3.11")
    return {"python": platform.python_version(), "platform": platform.platform(),
            "cpu": platform.processor(), "packages": versions}, problems


def cache_path(spec):
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    hub = Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub"))
    return hub / ("models--" + spec["repository"].replace("/", "--")) / "snapshots" / spec["revision"]


def model_path(model, root, overrides):
    spec = REGISTRY[model]
    if model in overrides:
        return Path(overrides[model]).resolve()
    local = root / spec["local_folder"]
    snapshot = cache_path(spec)
    return (snapshot if not local.exists() and snapshot.is_dir() else local).resolve()


def inspect_model(model, path):
    spec = REGISTRY[model]
    files = {}
    try:
        required = ["config.json", "tokenizer_config.json", "tokenizer.json", *spec["weights_sha256"]]
        if spec["adapter"] == "ettin":
            required += ["modules.json", "config_sentence_transformers.json", "sentence_bert_config.json",
                         "1_Pooling/config.json", "2_Dense/config.json", "3_LayerNorm/config.json",
                         "4_Dense/config.json"]
        for name in required:
            if not (path / name).is_file():
                raise ValueError(f"Missing local file {path / name}; restore pinned snapshot or use --model-path")
        if spec["architecture"] not in read_json(path / "config.json").get("architectures", []):
            raise ValueError(f"Expected {spec['architecture']} in {path / 'config.json'}")
        if spec["adapter"] == "ettin":
            modules = read_json(path / "modules.json")
            expected = [
                ("", "sentence_transformers.base.modules.transformer.Transformer"),
                ("1_Pooling", "sentence_transformers.sentence_transformer.modules.pooling.Pooling"),
                ("2_Dense", "sentence_transformers.base.modules.dense.Dense"),
                ("3_LayerNorm", "sentence_transformers.sentence_transformer.modules.layer_norm.LayerNorm"),
                ("4_Dense", "sentence_transformers.base.modules.dense.Dense"),
            ]
            if [(row["path"], row["type"]) for row in modules] != expected:
                raise ValueError("Unexpected Ettin scoring module chain")
        extras = {file.relative_to(path).as_posix() for file in path.iterdir() if file.is_file()
                  and (file.suffix in {".json", ".jinja", ".model", ".txt"} or file.name == "vocab")}
        for name in sorted(set(required) | extras):
            files[name] = file_sha(path / name)
            expected_hash = spec["weights_sha256"].get(name)
            if expected_hash and files[name] != expected_hash:
                raise ValueError(f"Model checksum mismatch for {path / name} (pinned revision {spec['revision']})")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"path": str(path), "revision": spec["revision"], "status": "UNAVAILABLE", "error": str(exc)}
    return {"path": str(path), "revision": spec["revision"], "status": "READY", "files_sha256": files}


def causal_prompt(adapter, question):
    if adapter == "qwen":
        return (f"{QWEN_PREFIX}<Instruct>: {QWEN_INSTRUCTION}\n<Query>: {question}\n<Document>: ",
                QWEN_SUFFIX, ("yes", "no"))
    return f"{MXBAI_PREFIX}{question}\ndocument: ", MXBAI_SUFFIX, ("1", "0")


def tokenize_pairs(tokenizer, question, passages, limit, adapter):
    if adapter in ("qwen", "mixedbread"):
        prefix, suffix, _ = causal_prompt(adapter, question)
        texts = [prefix + text + suffix for text in passages]
        tokens = tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        original = [len(ids) for ids in tokens["input_ids"]]
        shortened = []
        for text, ids, offsets in zip(passages, tokens["input_ids"], tokens["offset_mapping"], strict=True):
            to_drop = max(0, len(ids) - limit)
            if to_drop:
                document_end = len(prefix) + len(text)
                boundaries = [max(0, left - len(prefix)) for left, right in offsets
                              if left < document_end and right > len(prefix)]
                keep = len(boundaries) - to_drop
                # Повторная токенизация сохраняет шаблон даже на стыке текстовых сегментов.
                while keep >= 0:
                    cut = boundaries[keep]
                    ids = tokenizer([prefix + text[:cut] + suffix], add_special_tokens=False,
                                    verbose=False)["input_ids"][0]
                    if len(ids) <= limit:
                        break
                    keep -= 1
                if keep < 0:
                    raise ValueError("Sequence limit cannot preserve the complete question and reranker template")
            shortened.append(ids)
        encoded = {"input_ids": shortened, "attention_mask": [[1] * len(s) for s in shortened]}
    else:
        queries = [question] * len(passages)
        encoded = tokenizer(queries, passages, verbose=False)
        original = [len(ids) for ids in encoded["input_ids"]]
        if max(original) > limit:
            if len(tokenizer(question, add_special_tokens=False)["input_ids"]) + tokenizer.num_special_tokens_to_add(
                pair=True
            ) >= limit:
                raise ValueError("Sequence limit cannot preserve the complete question")
            encoded = tokenizer(queries, passages, truncation="only_second", max_length=limit)
    counts = [len(ids) for ids in encoded["input_ids"]]
    if max(counts) > limit:
        raise ValueError("Tokenizer could not enforce the sequence limit")
    lengths = [{"original_tokens": before, "input_tokens": after, "truncated_tokens": before - after,
                "truncation_fraction": (before - after) / before}
               for before, after in zip(original, counts, strict=True)]
    return tokenizer.pad(encoded, padding=True, return_tensors="pt"), lengths


class Adapter:
    def __init__(self, model_id, path, device, limit):
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

        spec = REGISTRY[model_id]
        self.kind, self.device, self.torch, self.limit = spec["adapter"], device, torch, limit
        options = {"local_files_only": True, "trust_remote_code": False}
        model_options = {"dtype": torch.float32, "use_safetensors": True, "attn_implementation": "sdpa"}
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, **options)
        if self.kind == "ettin":
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(str(path), device=device, max_length=limit, activation_fn=torch.nn.Identity(),
                                      model_kwargs=model_options, **options)
        else:
            causal = self.kind in ("qwen", "mixedbread")
            factory = AutoModelForCausalLM if causal else AutoModelForSequenceClassification
            self.model = factory.from_pretrained(path, **options, **model_options).to(device)
            if causal:
                self.tokenizer.padding_side = "left"
                _, _, labels = causal_prompt(self.kind, "")
                ids = [self.tokenizer(label, add_special_tokens=False)["input_ids"] for label in labels]
                expected_ids = [[9693], [2152]] if self.kind == "qwen" else [[16], [15]]
                if ids != expected_ids:
                    raise ValueError("Invalid trained response token IDs in local tokenizer")
                self.label_ids = [i[0] for i in ids]
        self.model.eval()

    def score(self, question, passages):
        started = perf_counter()
        inputs, lengths = tokenize_pairs(self.tokenizer, question, passages, self.limit, self.kind)
        tokenize_seconds = perf_counter() - started
        started = perf_counter()
        inputs = inputs.to(self.device)
        with self.torch.inference_mode():
            if self.kind == "ettin":
                scores = self.model(dict(inputs))["scores"]
            elif self.kind == "classifier":
                scores = self.model(**inputs).logits
                if scores.shape != (len(passages), 1):
                    raise ValueError("Expected exactly one sequence-classification logit per pair")
            else:
                logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
                positive, negative = self.label_ids
                scores = logits[:, positive] - logits[:, negative]
        scores = scores.float().flatten().cpu().tolist()
        inference_seconds = perf_counter() - started
        if len(scores) != len(passages) or not all(map(math.isfinite, scores)):
            raise ValueError("Model returned invalid pair scores")
        return scores, lengths, tokenize_seconds, inference_seconds
