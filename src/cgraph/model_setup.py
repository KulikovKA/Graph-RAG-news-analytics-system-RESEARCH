"""Pinned file downloader. Plain HTTPS fallback avoids hanging Xet transports."""
import os
import shutil
from pathlib import Path
from urllib.request import urlopen

from .common import file_sha256, read_json, write_json


def download_models(cfg):
    inventory = {}
    for repo, rev, directory, files in [
        (cfg["ner"]["checkpoint"], cfg["ner"]["revision"], cfg["ner"]["local_dir"], ["gliner_config.json", "pytorch_model.bin", "README.md"]),
        (cfg["ner"]["tokenizer_checkpoint"], cfg["ner"]["tokenizer_revision"], cfg["ner"]["tokenizer_dir"], ["config.json", "tokenizer_config.json", "spm.model", "README.md"])
    ]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name in files:
            dest = directory / name
            if not dest.exists():
                url = f"https://huggingface.co/{repo}/resolve/{rev}/{name}"
                temp = dest.with_suffix(dest.suffix + ".partial")
                with urlopen(url, timeout=60) as response, temp.open("wb") as stream:
                    shutil.copyfileobj(response, stream, length=4 * 1024 * 1024)
                temp.replace(dest)
            inventory[f"{repo}@{rev}/{name}"] = file_sha256(dest)
    lock = Path("contracts/model_files.json")
    if lock.exists() and read_json(lock) != inventory:
        raise ValueError("Downloaded model/tokenizer differs from pinned file hashes")
    if not lock.exists():
        write_json(lock, inventory)
    runtime_model(cfg)
    return inventory


def runtime_model(cfg):
    """Embed pinned backbone config; GLiNER must not fetch mutable 'main'."""
    original, tokenizer = Path(cfg["ner"]["local_dir"]), Path(cfg["ner"]["tokenizer_dir"])
    lock = read_json("contracts/model_files.json")
    for repo, rev, directory in [(cfg["ner"]["checkpoint"], cfg["ner"]["revision"], original),
                                 (cfg["ner"]["tokenizer_checkpoint"], cfg["ner"]["tokenizer_revision"], tokenizer)]:
        for key, expected in lock.items():
            prefix = f"{repo}@{rev}/"
            if key.startswith(prefix) and file_sha256(directory / key[len(prefix):]) != expected:
                raise ValueError("Pinned model/tokenizer SHA-256 mismatch")
    path = original.parent / "runtime_gliner"
    path.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer_config.json", "spm.model"):
        shutil.copyfile(tokenizer / name, path / name)
    effective = read_json(original / "gliner_config.json")
    effective["encoder_config"] = read_json(tokenizer / "config.json")
    write_json(path / "gliner_config.json", effective)
    weights = path / "pytorch_model.bin"
    if not weights.exists():
        try:
            os.link(original / "pytorch_model.bin", weights)
        except OSError:
            shutil.copyfile(original / "pytorch_model.bin", weights)
    if file_sha256(weights) != file_sha256(original / "pytorch_model.bin"):
        raise ValueError("Runtime weights differ from checkpoint")
    return path
