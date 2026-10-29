"""Execute the real zero-shot CLI with process-wide Gold/network access denied."""
import runpy
import sys


def audit(event, args):
    if event == "socket.connect":
        raise PermissionError("Inference network access forbidden")
    if event == "open" and isinstance(args[0], (str, bytes)):
        path = str(args[0]).replace("\\", "/").lower()
        if (any(p in path for p in ("/labels/", "/gold/", "/holdout/", "/calibration/", "/re_model/"))
                or path.endswith("/ontology.v1.0.json")
                or path.endswith(("cgraph/labels.py", "cgraph/relations.py"))):
            raise PermissionError("Gold, closed splits, or historical fitted model accessed by inference")
    if event == "import" and args[0] in {"cgraph.labels", "cgraph.relations"}:
        raise PermissionError("Training/annotation module import forbidden")


sys.addaudithook(audit)
sys.argv = ["cgraph-zero", "extract", *sys.argv[1:]]
runpy.run_module("cgraph.zeroshot_cli", run_name="__main__")
assert not {"cgraph.labels", "cgraph.relations"} & sys.modules.keys()
from pathlib import Path
from cgraph.common import file_sha256, write_json
output = Path(sys.argv[sys.argv.index("--output") + 1])
write_json(output / "access_audit.json", dict(guard="process-wide Python audit hook", completed=True,
    forbidden_access=["Gold labels", "CALIBRATION/HOLDOUT paths", "historical fitted RE", "network connections"],
    annotation_or_training_module_imported=False, run_sha256=file_sha256(output / "run.json"),
    guard_script_sha256=file_sha256(__file__), graph_semantics_certified=False))
print("ACCESS GUARD PASS: actual inference completed; this does not certify graph semantics", file=sys.stderr)
