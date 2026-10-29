"""Pinned pretrained assets; no fitting or checkpoint conversion."""
from pathlib import Path
import shutil
from urllib.request import urlopen

from .common import file_sha256, read_json, write_json
from .model_setup import download_models


def inventory(cfg):
    result = {}
    specs = [cfg["zero_shot_re"]]
    if "backbone" in cfg["zero_shot_re"]:
        specs.append(cfg["zero_shot_re"]["backbone"])
    for spec in specs:
        for name in spec["files"]:
            result[f"{spec['checkpoint']}@{spec['revision']}/{name}"] = file_sha256(Path(spec["local_dir"]) / name)
    return result


def verify(cfg):
    actual = inventory(cfg)
    if actual != read_json(cfg["zero_shot_re"]["file_contract"]):
        raise ValueError("Pretrained RE asset SHA-256 mismatch")
    return actual


def download(cfg):
    download_models(cfg)
    specs = [cfg["zero_shot_re"]]
    if "backbone" in cfg["zero_shot_re"]:
        specs.append(cfg["zero_shot_re"]["backbone"])
    for spec in specs:
        directory = Path(spec["local_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        for name in spec["files"]:
            dest = directory / name
            if not dest.exists():
                url = f"https://huggingface.co/{spec['checkpoint']}/resolve/{spec['revision']}/{name}"
                temp = dest.with_suffix(dest.suffix + ".partial")
                with urlopen(url, timeout=60) as source, temp.open("wb") as target:
                    shutil.copyfileobj(source, target, 4 * 1024 * 1024)
                temp.replace(dest)
    # A published contract is mandatory; never silently bless a new download.
    return verify(cfg)
