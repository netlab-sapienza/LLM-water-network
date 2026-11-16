"""Run real baseline smoke, interrupt/resume safely, regenerate offline figures."""
import argparse
import subprocess
import sys
from pathlib import Path

from waterbench.data import acquire, prepare
from waterbench.runner import run
from waterbench.scoring import evaluate, report
from waterbench.storage import digest, read, verify_dataset, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/datasets/smoke-all")
    args = p.parse_args()
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], check=True)
    subprocess.run([sys.executable, "-m", "scripts.check_hydraulics"], check=True)
    acquire("data/networks")
    dataset = Path(args.dataset)
    if not dataset.exists():
        prepare("configs/smoke-all.json", "data/networks", dataset)
    verify_dataset(dataset)
    params = (dataset, "results/verified/runs", ["physics-residual", "extra-trees", "seasonal-naive", "hist-gradient-boosting"],
              ["validation", "test"], ["leak", "forecast"], ["grid", "ctown", "ltown"], "configs/baselines.json")
    root = run(*params, stop_after=7)
    before = {p.name: p.read_bytes() for p in (root/"records").glob("*.json")}
    assert run(*params) == root
    assert all((root/"records"/name).read_bytes() == content for name, content in before.items())
    evaldir = evaluate(dataset, [root], "results/verified/evaluations", 1000)
    assert all(r["missing"] == r["failed"] == 0 for r in read(evaldir/"evaluation.json")["coverage"])
    report(evaldir)
    before_report = {p.name:p.read_bytes() for p in evaldir.glob("*.csv")}
    assert evaluate(dataset, [root], "results/verified/evaluations", 1000) == evaldir
    assert all((evaldir/name).read_bytes() == content for name, content in before_report.items())
    summary = {"dataset": dataset.as_posix(), "run": root.as_posix(), "evaluation": evaldir.as_posix(),
               "records": len(list((root/"records").glob("*.json"))),
               "checks": ["software fixtures", "real hydraulics", "baseline pipeline", "planned interruption/resume",
                          "offline CSV reproducibility", "figures from archived real baseline outputs"],
               "llm_verified": False, "linux_verified": False, "scientific_status": "functional smoke only"}
    write("results/verified/verification.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
