"""Functional CPU-only check of selected methods and accounting on existing hydraulic data."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from waterbench.methods import BASELINE_TASKS,TASKS
from waterbench.runner import run
from waterbench.scoring import evaluate,report
from waterbench.catalog import catalog
from waterbench.storage import read,write,file_hash,verify_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",required=True)
    p.add_argument("--output",default="results/baselines-v040")
    args = p.parse_args()
    root = Path(args.output); manifest = verify_dataset(args.dataset)
    original = file_hash("waterbench/data.py")
    cfg = read("configs/baselines.json"); paths = []
    for rep in range(3):
        config = root/f"baseline-{rep}.json"
        write(config,{**cfg,"training_seed":cfg["training_seed"]+rep})
        paths.append(run(args.dataset,root/"runs",list(BASELINE_TASKS),["validation","test"],TASKS,
                         manifest["config"]["networks"],config,repetition=rep))
    evaluation = evaluate(args.dataset,paths,root/"evaluations",bootstrap_count=100)
    report(evaluation); catalog(args.dataset,paths,root/"catalog")
    result = read(evaluation/"evaluation.json")
    assert all(c["missing"]==c["failed"]==0 for c in result["coverage"]),result["coverage"]
    assert set(r["method"] for r in result["aggregates"])==set(BASELINE_TASKS)
    assert original==file_hash("waterbench/data.py")
    write(root/"verification.json",{"purpose":"functional check, not confirmatory scientific results",
          "dataset_id":manifest["id"],"evaluation":str(evaluation),"runs":[str(x) for x in paths],
          "methods":BASELINE_TASKS,"coverage":result["coverage"],"hydraulic_generator_sha256":original,
          "repetitions":3,"bootstrap_resamples":100,"real_llm_calls":0})
    print(evaluation)


if __name__=="__main__": main()
