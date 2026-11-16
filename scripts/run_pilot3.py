"""Sequential development campaign, with a durable list of immutable runs. Own endpoint only."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from waterbench.methods import BASELINE_TASKS, TASKS
from waterbench.runner import run, endpoint_config
from waterbench.scoring import evaluate, report
from waterbench.catalog import catalog
from waterbench.storage import read, write, verify_dataset, digest, code_hash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True); p.add_argument("--endpoint", required=True)
    p.add_argument("--output", default="results/campaigns")
    p.add_argument("--baseline-config", default="configs/baselines.json")
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--bootstrap", type=int, help="Default: 1000 validation, 10000 test")
    p.add_argument("--generation-seeds", nargs="+", type=int)
    p.add_argument("--execute", action="store_true", help="Without this flag only print size; no model calls")
    args = p.parse_args()
    manifest = verify_dataset(args.dataset); ep = endpoint_config(args.endpoint)
    cases = [c for c in read(Path(args.dataset)/"cases.json") if c["split"]==args.split]
    bootstrap = args.bootstrap if args.bootstrap is not None else (10000 if args.split=="test" else 1000)
    if bootstrap < 100: p.error("Need at least 100 bootstrap resamples")
    if set(c["task"] for c in cases)!=set(TASKS) or args.repetitions<1:
        p.error("Need all three tasks and at least one repetition")
    if args.generation_seeds and (len(args.generation_seeds)!=args.repetitions or ep.get("seed_supported") is not True):
        p.error("One seed per repetition; empirically verified seed_supported:true required")
    calls = sum(len(c.get("decision_hours", [None])) for c in cases)*args.repetitions
    cfg = read(args.baseline_config)
    identity = {"dataset_id":manifest["id"], "endpoint":ep, "baseline":cfg, "repetitions":args.repetitions,
                "generation_seeds":args.generation_seeds, "split":args.split}
    identity["code_hash"] = code_hash(exclude_scoring=True)
    root = Path(args.output)/digest(identity)[:20]
    print(f"Plan: {len(cases)} {args.split} cases, {args.repetitions} repetitions, {calls} direct LLM calls; zero retries. Output {root}")
    if not args.execute: return
    if args.split=="test" and not (Path(args.dataset)/"freeze.json").exists():
        p.error("Freeze required before test execution; use scripts/freeze_final.py")
    if ep["mode"]!="direct": p.error("First campaign requires direct mode")
    run_paths = []
    write(root/"campaign.json", {"plan":identity, "runs":run_paths, "llm_decisions_planned":calls})
    # Fit every baseline first, then one synchronous LLM run at a time.
    for method_group in (list(BASELINE_TASKS), ["llm"]):
        for rep in range(args.repetitions):
            baseline = root/f"baseline-rep{rep}.json"
            write(baseline, {**cfg, "training_seed":cfg["training_seed"]+rep})
            seed = args.generation_seeds[rep] if method_group==["llm"] and args.generation_seeds else None
            path = run(args.dataset, root/"runs", method_group, [args.split], TASKS,
                       manifest["config"]["networks"], baseline, args.endpoint if method_group==["llm"] else None,
                       repetition=rep, generation_seed=seed)
            run_paths.append(str(path))
            write(root/"campaign.json", {"plan":identity, "runs":run_paths, "llm_decisions_planned":calls})
    evaluation = evaluate(args.dataset, run_paths, root/"evaluations", bootstrap_count=bootstrap)
    report(evaluation)
    catalog(args.dataset, run_paths, root/"catalog")
    print(evaluation)


if __name__=="__main__": main()
