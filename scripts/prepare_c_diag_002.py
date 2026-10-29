"""Download only explicitly pinned, complete pretrained token classifiers (no data)."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sys
import time

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgraph.common import file_sha256, read_json, write_json


def fetch(spec):
    directory = Path(spec["local_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    for name in spec["files"]:
        dest = directory / name
        if not dest.exists():
            url = f"https://huggingface.co/{spec['checkpoint']}/resolve/{spec['revision']}/{name}"
            for attempt in range(3):
                try:
                    with requests.get(url, stream=True, timeout=(30, 90)) as response:
                        response.raise_for_status()
                        with dest.with_suffix(dest.suffix + ".partial").open("wb") as out:
                            for block in response.iter_content(4 * 1024 * 1024):
                                out.write(block)
                    dest.with_suffix(dest.suffix + ".partial").replace(dest)
                    break
                except requests.RequestException:
                    if attempt == 2:
                        raise
                    time.sleep(2)
        print(f"downloaded {spec['checkpoint']}/{name}", flush=True)
    return {f"{spec['checkpoint']}@{spec['revision']}/{name}": file_sha256(directory / name)
            for name in spec["files"]}


if __name__ == "__main__":
    cfg = read_json("config/c_diag_002.json")
    with ThreadPoolExecutor(max_workers=2) as pool:
        contracts = list(pool.map(fetch, cfg["scientific_ner"].values()))
    inventory = {k: v for contract in contracts for k, v in contract.items()}
    path = Path("contracts/c_diag_002_model_files.json")
    if path.exists() and read_json(path) != inventory:
        raise ValueError("Pinned scientific checkpoint hashes changed")
    write_json(path, inventory)
