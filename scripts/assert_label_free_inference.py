"""Run the actual extract CLI with a process-wide file-access audit guard."""
import runpy
import sys


def audit(event, args):
    if event == "socket.connect":
        raise PermissionError("Extraction must run offline; network connections are forbidden")
    if event == "open" and isinstance(args[0], (str, bytes)):
        path = str(args[0]).replace("\\", "/").lower()
        if "/labels/" in path or "/gold/" in path or path.endswith("/ontology.v1.0.json") or path.endswith("cgraph/labels.py"):
            raise PermissionError("Inference attempted to access labels or original example-bearing ontology")


sys.addaudithook(audit)
sys.argv = ["cgraph", "extract", *sys.argv[1:]]
runpy.run_module("cgraph.cli", run_name="__main__")
if "cgraph.labels" in sys.modules:
    raise RuntimeError("Inference imported the annotation module")
print("PASS: real extraction completed with label-file access and network connections denied", file=sys.stderr)
