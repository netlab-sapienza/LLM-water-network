"""Freeze a measured protocol including model failures, without changing inference code."""
import argparse
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from waterbench.methods import TASKS,BASELINE_TASKS
from waterbench.runner import endpoint_config
from waterbench.storage import code_hash,digest,file_hash,read,verify_dataset,write


def freeze(dataset, baseline_path, endpoint_paths, evaluation, notes_path, declaration_path,
           selection_path, baseline_seeds, run_root=None):
    dataset,evaluation = Path(dataset),Path(evaluation)
    m = verify_dataset(dataset); pilot = read(evaluation/"evaluation.json")
    if m["config"]["stage"]!="core-candidate" or (dataset/"freeze.json").exists():
        raise ValueError("Freeze only a new core-candidate dataset")
    if pilot["stage"]!="pilot" or not pilot["coverage"] or any(c["missing"] or c["observed"]!=c["expected"] for c in pilot["coverage"]):
        raise ValueError("Complete measured pilot coverage required; missing attempts are not model outcomes")
    scientific = lambda c: {k:v for k,v in c.items() if k not in ("seed","cluster_seed","counts","stage")}
    if scientific(m["config"])!=scientific(pilot["dataset_config"]) or pilot["inference_code_hash"]!=code_hash(exclude_scoring=True):
        raise ValueError("Pilot protocol/inference code differs from core")
    # Verify original evaluated records, not merely the report's declared failure count.
    root = Path(run_root) if run_root else evaluation.parent.parent/"runs"
    if not pilot.get("source_hashes"): raise ValueError("Pilot source checksums required")
    for relative,sha in pilot["source_hashes"].items():
        p = (root/relative).resolve()
        if root.resolve() not in p.parents or file_hash(p)!=sha: raise ValueError("Pilot source changed or escaped run root")
    statuses = Counter()
    for coverage in pilot["coverage"]:
        directory = root/coverage["run"][:20]
        rm = read(directory/"manifest.json")
        if digest(rm["identity"])!=coverage["run"] or rm["id"]!=coverage["run"]: raise ValueError("Corrupt pilot run identity")
        records = list((directory/"records").glob("*.json"))
        if len(records)!=coverage["expected"]: raise ValueError("Incomplete pilot records")
        failed = 0
        for path in records:
            if path.relative_to(root).as_posix() not in pilot["source_hashes"]: raise ValueError("Unevaluated pilot record")
            record = read(path); status = record["status"]
            if status not in ("ok","invalid"): raise ValueError("Resolve pilot infrastructure/identity/interruption failures before freeze")
            if status=="invalid" and record.get("method")!="llm": raise ValueError("Resolve invalid baseline output before freeze")
            statuses[status]+=1; failed+=status!="ok"
        if failed!=coverage["failed"]: raise ValueError("Pilot failure count differs from records")
    declaration = read(declaration_path)
    if not declaration.get("operator_declaration") or not isinstance(declaration.get("unavailable_fields"),dict):
        raise ValueError("Explicit operator declaration and per-field unavailability reasons required")
    notes = Path(notes_path).read_text(encoding="utf-8").strip()
    if not notes: raise ValueError("Measured budget and model-failure rationale required")
    endpoints = [endpoint_config(p) for p in endpoint_paths]
    required = {(t,n) for t in TASKS for n in m["config"]["networks"]}
    unavailable = {}
    for e in endpoints:
        label = e["label"]+":"+e["mode"]
        if e!=pilot["method_configurations"].get(label): raise ValueError("Endpoint changed from measured pilot; sidecar does not override generation or identity")
        observed = {(r["task"],r["network"]) for r in pilot["aggregates"] if r["method"]==label and r["n"]>0}
        if not required.issubset(observed): raise ValueError("Measured pilot required for all tasks/networks; validity need not equal one")
        unavailable[label] = []
        for field in ("checkpoint","revision","quantization","backend_version","chat_template"):
            if e.get(field) in (None,"","UNKNOWN","SET_ON_SERVER"):
                if not declaration["unavailable_fields"].get(field): raise ValueError("Explain unavailable field: "+field)
                unavailable[label].append(field)
    cfg = read(baseline_path)
    descriptor = {k:v for k,v in cfg.items() if k!="training_seed"}
    if any(v!=descriptor for k,v in pilot["method_configurations"].items() if k in BASELINE_TASKS):
        raise ValueError("Baseline configuration differs from evaluated pilot")
    selection = read(selection_path)
    if not selection or not baseline_seeds or any(type(s) is not int for s in baseline_seeds): raise ValueError("Frozen selection and explicit baseline seeds required")
    write(dataset/"freeze.json", {"dataset_id":m["id"],"code_hash":code_hash(exclude_scoring=True),
        "baseline_config":cfg,"baseline_seeds":baseline_seeds,"generation_seeds":[None],"endpoints":endpoints,
        "pilot_evaluation":pilot["id"],"pilot_status_counts":dict(statuses),"notes":notes,
        "operator_metadata_declaration":declaration,"unavailable_metadata":unavailable,
        "specialist_selection":selection,"selection_sha256":file_hash(selection_path),
        "primary_metrics":{"leak":"event F1 + all-event localization + validity","forecast":"MAE + validity","cluster":"ARI + validity"},
        "policy":"Complete audited attempts; invalid model outputs retained. Unknown provenance explicit; exact pilot endpoints unchanged.",
        "freeze_launcher_sha256":file_hash(__file__)})
    return dataset/"freeze.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",required=True); p.add_argument("--endpoints",nargs="+",required=True)
    p.add_argument("--baseline-config",default="configs/baselines.json")
    p.add_argument("--pilot-evaluation",required=True); p.add_argument("--pilot-run-root")
    p.add_argument("--notes",required=True); p.add_argument("--metadata-declaration",required=True)
    p.add_argument("--selection",required=True); p.add_argument("--baseline-seeds",nargs="+",type=int,required=True)
    a = p.parse_args()
    try: print(freeze(a.dataset,a.baseline_config,a.endpoints,a.pilot_evaluation,a.notes,a.metadata_declaration,a.selection,a.baseline_seeds,a.pilot_run_root))
    except (ValueError,OSError,KeyError) as exc: p.exit(1,"ERROR: "+str(exc)+"\n")


if __name__=="__main__": main()
