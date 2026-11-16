"""Portable commands. Core test requires a development freeze."""
import argparse
import sys
from pathlib import Path

from .data import acquire, prepare
from .runner import endpoint_config, run
from .scoring import evaluate, report
from .storage import code_hash, environment, read, verify_dataset, write
from .backend import discover
from .methods import BASELINE_TASKS, TASKS
from .clustering import augment
from .exchange import export_cases, import_responses
from .catalog import catalog

BASELINES = list(BASELINE_TASKS)


def main():
    parser = argparse.ArgumentParser(prog="waterbench")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("doctor"); p.add_argument("--output", default="results/environment.json")
    p = sub.add_parser("acquire"); p.add_argument("--networks-dir", default="data/networks")
    p = sub.add_parser("prepare")
    p.add_argument("--config", required=True); p.add_argument("--networks-dir", default="data/networks")
    p.add_argument("--output", required=True)
    p = sub.add_parser("augment-cluster", help="Extend immutable hydraulic dataset with synthetic consumption clustering")
    p.add_argument("--dataset", required=True); p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20261005)
    p = sub.add_parser("discover", help="Read loaded inventory; save immutable endpoint and snapshot, no inference")
    p.add_argument("--template", default="configs/unsloth.example.json")
    p.add_argument("--output", required=True); p.add_argument("--label", required=True)
    p.add_argument("--model"); p.add_argument("--metadata", help="JSON operator declarations; no secrets")
    p.add_argument("--mode", choices=["direct", "tools"])
    p.add_argument("--response-model-id", action="append", help="Explicit documented response model alias; repeatable")
    p = sub.add_parser("run")
    p.add_argument("--dataset", required=True); p.add_argument("--output", default="results/runs")
    p.add_argument("--methods", nargs="+", choices=BASELINES + ["llm"], default=BASELINES)
    p.add_argument("--split", nargs="+", choices=["train", "validation", "test"], default=["validation"])
    p.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    p.add_argument("--networks", nargs="+", choices=["grid", "ctown", "ltown"], default=["grid", "ctown", "ltown"])
    p.add_argument("--baseline-config", default="configs/baselines.json")
    p.add_argument("--endpoint"); p.add_argument("--repetition", type=int, default=0)
    p.add_argument("--generation-seed", type=int, help="Only for backend with empirically verified seed_supported:true")
    p.add_argument("--max-cases", type=int); p.add_argument("--stop-after", type=int)
    p.add_argument("--session-id"); p.add_argument("--session-name")
    p = sub.add_parser("evaluate")
    p.add_argument("--dataset", required=True); p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--output", default="results/evaluations"); p.add_argument("--bootstrap", type=int, default=1000)
    p = sub.add_parser("report"); p.add_argument("--evaluation", required=True)
    p = sub.add_parser("catalog", help="Register actual experiment setups and exact inputs/prompts; offline")
    p.add_argument("--dataset", required=True); p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("audit"); p.add_argument("--dataset", required=True)
    p = sub.add_parser("export-cases", help="Exact development prompts for subscription clients; no API calls")
    p.add_argument("--dataset", required=True); p.add_argument("--output", required=True)
    p.add_argument("--baseline-config", default="configs/baselines.json")
    p.add_argument("--split", nargs="+", choices=["train", "validation", "test"], default=["validation"])
    p.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    p.add_argument("--networks", nargs="+", choices=["grid", "ctown", "ltown"], default=["grid", "ctown", "ltown"])
    p.add_argument("--max-cases", type=int)
    p = sub.add_parser("import-responses", help="Validate first raw subscription answers offline")
    p.add_argument("--dataset", required=True); p.add_argument("--packet", required=True)
    p.add_argument("--responses", required=True); p.add_argument("--metadata", required=True)
    p.add_argument("--output", default="results/runs"); p.add_argument("--repetition", type=int, default=0)
    p = sub.add_parser("freeze")
    p.add_argument("--dataset", required=True); p.add_argument("--baseline-config", default="configs/baselines.json")
    p.add_argument("--endpoints", nargs="+", required=True)
    p.add_argument("--pilot-evaluation", required=True)
    p.add_argument("--notes", required=True, help="Path to measured pilot/budget/selection rationale")
    p.add_argument("--generation-seeds", nargs="+", type=int, help="Explicit permitted inference seeds in final test")
    p.add_argument("--baseline-seeds", nargs="+", type=int, help="Explicit permitted baseline training seeds in final test")
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            result = environment(); write(args.output, result); print(result)
        elif args.command == "acquire":
            print(acquire(args.networks_dir))
        elif args.command == "prepare":
            print(prepare(args.config, args.networks_dir, args.output)["id"])
        elif args.command == "augment-cluster":
            print(augment(args.dataset, args.output, args.seed))
        elif args.command == "discover":
            template = endpoint_config(args.template)
            if args.mode: template["mode"] = args.mode
            if args.response_model_id: template["response_model_ids"] = args.response_model_id
            print(discover(template, args.output, args.label, args.model, args.metadata))
        elif args.command == "run":
            print(run(args.dataset, args.output, args.methods, args.split, args.tasks, args.networks,
                      args.baseline_config, args.endpoint, args.repetition, args.max_cases, args.stop_after,
                      args.session_id, args.session_name, args.generation_seed))
        elif args.command == "evaluate":
            if args.bootstrap < 100:
                raise ValueError("Use >=100 bootstrap resamples")
            print(evaluate(args.dataset, args.runs, args.output, args.bootstrap))
        elif args.command == "report":
            print(report(args.evaluation))
        elif args.command == "catalog":
            print(catalog(args.dataset, args.runs, args.output))
        elif args.command == "export-cases":
            print(export_cases(args.dataset, args.output, args.baseline_config, args.split, args.tasks, args.networks, args.max_cases))
        elif args.command == "import-responses":
            print(import_responses(args.dataset, args.packet, args.responses, args.metadata, args.output, args.repetition))
        elif args.command == "audit":
            m = verify_dataset(args.dataset)
            cases = read(Path(args.dataset) / "cases.json")
            if len({c["id"] for c in cases}) != len(cases):
                raise ValueError("Duplicate case ID")
            units = {}
            for c in cases:
                if c["unit"] in units and units[c["unit"]] != c["split"]:
                    raise ValueError("Unit shared between splits")
                units[c["unit"]] = c["split"]
            print(f"Verified {m['id']}: {len(cases)} cases, {len(units)} independent realizations")
        elif args.command == "freeze":
            dataset = Path(args.dataset); m = verify_dataset(dataset)
            if m["config"]["stage"] != "core-candidate" or (dataset / "freeze.json").exists():
                raise ValueError("Freeze only a new core-candidate dataset")
            pilot = read(Path(args.pilot_evaluation) / "evaluation.json")
            if pilot["stage"] != "pilot" or any(c["missing"] or c["failed"] for c in pilot["coverage"]):
                raise ValueError("Complete successful pilot required")
            notes = Path(args.notes).read_text(encoding="utf-8")
            endpoints = [read(p) if read(p).get("provider_scope")=="subscription" else endpoint_config(p) for p in args.endpoints]
            scientific = lambda cfg: {k:v for k,v in cfg.items() if k not in ("seed", "cluster_seed", "counts", "stage")}
            if scientific(m["config"]) != scientific(pilot.get("dataset_config", {})) or pilot.get("inference_code_hash") != code_hash(exclude_scoring=True):
                raise ValueError("Pilot protocol/code differs from core; re-evaluate or repeat development with matching protocol")
            for e in endpoints:
                fields = ("model", "label", "platform", "reasoning_mode", "operator_declaration") if e.get("provider_scope")=="subscription" else ("checkpoint", "revision", "quantization", "backend_version", "chat_template")
                if any(e.get(k) in (None, "", "UNKNOWN", "SET_ON_SERVER") for k in fields):
                    raise ValueError("Complete measured model metadata before freeze")
                if e != pilot.get("method_configurations", {}).get(e["label"]+":"+e["mode"]):
                    raise ValueError("Endpoint metadata/generation differs from the evaluated pilot arm")
                if e.get("provider_scope")=="subscription" and (e.get("mode")!="direct" or e.get("comparison_track")!="supplementary-product-level"):
                    raise ValueError("Subscription final test requires declared supplementary product track")
                required = {(t, n) for t in TASKS for n in m["config"]["networks"]}
                observed = {(r["task"], r["network"]) for r in pilot["aggregates"]
                            if r["method"] == e["label"] + ":" + e["mode"] and r["validity"] == 1}
                if not required.issubset(observed):
                    raise ValueError("Every selected endpoint/mode needs successful real pilot on three tasks/all networks")
            write(dataset / "freeze.json", {"dataset_id": m["id"], "code_hash": code_hash(exclude_scoring=True),
                  "baseline_config": read(args.baseline_config), "endpoints": endpoints,
                  "pilot_evaluation": pilot["id"], "notes": notes, "generation_seeds": args.generation_seeds or [None],
                  "baseline_seeds": args.baseline_seeds or [read(args.baseline_config)["training_seed"]],
                  "primary_metrics": {"leak": "event F1 (valid episodes) + overall detection/localization@1", "forecast": "MAE + validity", "cluster": "ARI + validity"}})
            print(dataset / "freeze.json")
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
