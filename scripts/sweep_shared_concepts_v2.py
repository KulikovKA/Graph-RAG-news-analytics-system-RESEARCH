"""Последовательно проверяет явно заданные конфигурации EC, CC и delta на Corpus100."""

from argparse import ArgumentParser
from pathlib import Path

from filelock import FileLock
from tqdm import tqdm

from cgraph.shared_concepts_v2 import Config, digest
from run_shared_concepts_v2 import OUTPUT, limits, read_configuration, resource_options, slug
from shared_concepts_v2_pipeline import PREVIOUS_RUN, frozen_corpus, load_checked, run_catalog, write_checked


def sweep(entities, metadata, configurations, output, run_id, **options):
    output.mkdir(parents=True, exist_ok=True)
    rows = [(label, config.to_dict()) for label, config in configurations]
    identity = dict(configurations=rows, source=metadata, run_id=run_id)
    with FileLock(output / ".sweep.lock", timeout=0):
        lock = output / "configurations.json"
        if lock.exists():
            if digest(load_checked(lock)) != digest(identity):
                raise ValueError("Experiment ID already belongs to another input or configuration list")
        else:
            write_checked(lock, identity)
        results = []
        for label, config in tqdm(configurations, desc="V2 configurations", unit="config"):
            result = run_catalog(entities, config, f"research:corpus100:{run_id}:{label}",
                                 output / label, metadata=metadata, regression=True,
                                 ee_cache=output / "ee_cache", **options)
            results.append(dict(label=label, config=config.to_dict(), generation=result["generation"],
                                ee=result["ee"], statistics=result["statistics"],
                                wall_seconds=result["wall_seconds"], peak_rss_bytes=result["peak_rss_bytes"]))
            write_checked(output / "results.json", dict(identity=digest(identity), results=results,
                                                        complete=len(results) == len(rows)))
    return results


def configurations(path):
    rows = read_configuration(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Provide an explicit nonempty list of trial configurations")
    trials = [(slug(row["label"]), Config.from_dict({k: v for k, v in row.items() if k != "label"}))
              for row in rows]
    if len({label.casefold() for label, _ in trials}) != len(trials) or any(
            label.casefold() == "ee_cache" for label, _ in trials):
        raise ValueError("Duplicate or reserved configuration name")
    return trials


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--configurations", type=Path, required=True)
    parser.add_argument("--run-id", type=slug, required=True)
    parser.add_argument("--frozen-run", type=Path, default=PREVIOUS_RUN)
    resource_options(parser)
    args = parser.parse_args()
    options = limits(args)
    trials = configurations(args.configurations)
    print("ПРЕДУПРЕЖДЕНИЕ: ЭКСПЕРИМЕНТ SHARED CONCEPTS V2 МОЖЕТ ЗАНЯТЬ НЕСКОЛЬКО МИНУТ.", flush=True)
    entities, metadata = frozen_corpus(args.frozen_run)
    output = OUTPUT / "sweeps" / args.run_id
    sweep(entities, metadata, trials, output, args.run_id, **options)
    print(output / "results.json")


if __name__ == "__main__":
    main()
