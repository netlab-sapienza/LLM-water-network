"""Supplementary specialists on existing frozen inputs; original code/freeze stay intact."""
import argparse
import datetime as dt
import importlib.metadata as metadata
import os
import platform
import sys
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "2"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from waterbench import methods, scoring
from waterbench.storage import read, write, digest, file_hash, verify_dataset, code_hash

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/supplementary-baselines-v1"
DATA = ROOT / "data/datasets"
TASKS = {"lightgbm-residual": "leak", "chronos2-median": "forecast", "shape-kasba": "cluster"}
CFG = {
    "training_seed": 20261002,
    "classification": {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 7,
        "max_depth": -1, "min_child_samples": 5, "reg_lambda": 1.0,
        "n_jobs": 2, "verbosity": -1, "deterministic": True, "force_col_wise": True},
    "alarm_threshold": 0.5,
    "clustering": {"distance": "msm", "distance_params": {"c": 1.0},
        "max_iter": 300, "tol": 1e-6, "ba_subset_size": 0.5,
        "initial_step_size": 0.05, "decay_rate": 0.1},
    "forecast": {"repository": "amazon/chronos-2", "device": "cpu", "dtype": "float32",
        "context_length": 168, "prediction_length": 24, "batch_size": 1,
        "point_quantile": 0.5, "nonnegative_clip": True, "cross_learning": False,
        "fine_tuning": False, "future_covariates": False},
    "design": "supplementary extension chosen after inspecting original test; fixed parameters, no test tuning",
}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def prepare():
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(CFG["forecast"]["repository"], token=False)
    target = snapshot_download(CFG["forecast"]["repository"], revision=info.sha,
        token=False, local_dir=OUT / "model/chronos-2", allow_patterns=["*.json", "*.safetensors"])
    files = {p.relative_to(target).as_posix(): file_hash(p) for p in Path(target).rglob("*")
             if p.is_file() and ".cache" not in p.parts}
    write(OUT / "model/provenance.json", {"repository": info.id, "revision": info.sha,
          "files": files, "downloaded_utc": now(), "path": str(Path(target).resolve())})


def load_forecaster():
    import torch
    from chronos import Chronos2Pipeline
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    p = read(OUT / "model/provenance.json")
    for name, expected in p["files"].items():
        assert file_hash(Path(p["path"]) / name) == expected, name
    return Chronos2Pipeline.from_pretrained(p["path"], device_map="cpu", torch_dtype=torch.float32)


def fit_leak(dataset, network, seed):
    from lightgbm import LGBMClassifier
    x, y, zx, zy = [], [], [], []
    for case in read(dataset / "cases.json"):
        if case["split"] != "train" or case["task"] != "leak" or case["network"] != network:
            continue
        obs = read(dataset / "observations" / (case["id"] + ".json"))
        truth = read(dataset / "evaluator" / (case["id"] + ".json"))
        for hour in case["decision_hours"]:
            f = methods.features(methods.visible(obs, hour))
            active = bool(truth["has_leak"] and truth["start_hour"] <= hour < truth["end_hour"])
            x.append(f); y.append(active)
            if active:
                zx.append(f); zy.append(truth["zone"])
    assert len(set(y)) == 2 and len(set(zy)) > 1
    params = {**CFG["classification"], "random_state": seed}
    return (LGBMClassifier(**params, class_weight="balanced").fit(np.asarray(x), y),
            LGBMClassifier(**params).fit(np.asarray(zx), zy))


def prediction(name, state, obs, seed):
    if name == "lightgbm-residual":
        detector, local = state
        x = methods.features(obs).reshape(1, -1)
        score = float(detector.predict_proba(x)[0][list(detector.classes_).index(True)])
        order = np.argsort(-local.predict_proba(x)[0], kind="stable")[:3]
        alarm = score >= CFG["alarm_threshold"]
        return {"alarm": alarm, "score": score,
                "ranking": [str(local.classes_[i]) for i in order] if alarm else []}, {}
    if name == "shape-kasba":
        from aeon.clustering import KASBA
        ids = obs["consumer_ids"]
        estimator = KASBA(n_clusters=obs["k"], random_state=seed, **CFG["clustering"])
        labels = estimator.fit_predict(np.asarray([obs["profiles"][i] for i in ids])[:, None, :])
        return {"assignments": dict(zip(ids, map(int, labels)))}, {"estimator_parameters": estimator.get_params()}
    import torch
    torch.manual_seed(seed)
    quantiles, _ = state.predict_quantiles(
        torch.tensor(obs["history_L_s"], dtype=torch.float32).reshape(1, 1, 168),
        prediction_length=24, quantile_levels=[0.5], batch_size=1, context_length=168)
    values = quantiles[0].detach().cpu().numpy().reshape(-1)
    assert values.shape == (24,), values.shape
    return {"forecast_L_s": np.maximum(values, 0).tolist()}, {"unclipped_median_L_s": values.tolist()}


def run(dataset, name, split, rep, cfg, forecaster, env):
    manifest = verify_dataset(dataset)
    cases = [c for c in read(dataset / "cases.json") if c["split"] == split and c["task"] == TASKS[name]]
    ident = {"schema": 1, "dataset_id": manifest["id"], "code_hash": code_hash(exclude_scoring=True),
        "adapter_sha256": file_hash(__file__), "environment": env, "methods": [name], "cases": cases,
        "baseline_config": cfg, "endpoint": None, "repetition": rep,
        "prompt_version": digest({"system": methods.SYSTEM, "instructions": methods.INSTRUCTIONS})}
    rid = digest(ident)
    root = OUT / str(manifest["config"]["seed"]) / split / "runs" / rid[:20]
    parent = root.parent
    for prior in parent.glob("*/manifest.json"):
        previous = read(prior)
        compare = {k: v for k, v in previous["identity"].items() if k != "adapter_sha256"}
        if compare == {k: v for k, v in ident.items() if k != "adapter_sha256"}:
            expected = sum(len(c.get("decision_hours", [None])) for c in cases)
            records = list((prior.parent / "records").glob("*.json"))
            if len(records) == expected and all(read(p)["status"] != "started" for p in records):
                print("Reusing complete existing run", prior.parent, flush=True)
                return prior.parent
            raise RuntimeError("Existing partial run needs explicit audit; no implicit repetition")
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "writer.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        write(root / "manifest.json", {"id": rid, "identity": ident, "dataset_path_hint": str(dataset),
            "purpose": "supplementary", "timing_note": "CPU only; at most two threads; contention possible with ongoing local runs"})
        states = {}
        for case in cases:
            obs_all = read(dataset / "observations" / (case["id"] + ".json"))
            if case["network"] not in states:
                t, cpu = time.perf_counter(), time.process_time()
                state = fit_leak(dataset, case["network"], cfg["training_seed"]) if name == "lightgbm-residual" else forecaster
                states[case["network"]] = state
                write(root / "fits" / (case["network"] + ".json"), {"method": name,
                    "task": TASKS[name], "network": case["network"], "fit_s": time.perf_counter()-t,
                    "fit_cpu_s": time.process_time()-cpu, "training_seed": cfg["training_seed"],
                    "training_splits": ["train"] if name == "lightgbm-residual" else [],
                    "parameters": cfg, "fit_note": "shared Chronos loading cost saved separately; KASBA per-case fit included in prediction timing"})
                if name == "lightgbm-residual":
                    import joblib
                    joblib.dump(state, root / "fits" / (case["network"] + ".joblib"))
                    write(root / "models" / (case["network"] + "-estimators.json"),
                          {"detector": state[0].get_params(), "localizer": state[1].get_params()})
            for hour in case.get("decision_hours", [None]):
                recid = digest([rid, case["id"], name, hour])[:24]
                path = root / "records" / (recid + ".json")
                if path.exists():
                    assert read(path)["status"] != "started", "Interrupted attempt requires review; no automatic retry"
                    continue
                obs = methods.visible(obs_all, hour)
                record = {"id": recid, "run_id": rid, "case": case, "method": name,
                    "method_label": name, "hour": hour, "status": "started", "prediction": None,
                    "prompt_hash": digest({"system": methods.SYSTEM, "user": methods.prompt(obs)}),
                    "observation_hash": digest(obs), "started_utc": now(), "retries": 0}
                write(path, record)
                t, cpu = time.perf_counter(), time.process_time()
                try:
                    pred, extra = prediction(name, states[case["network"]], obs, cfg["training_seed"])
                    record.update(status="ok", prediction=methods.validate(pred, obs), diagnostic=extra)
                except (ValueError, AssertionError) as exc:
                    record.update(status="invalid", error=str(exc))
                record.update(elapsed_s=time.perf_counter()-t, client_cpu_s=time.process_time()-cpu, ended_utc=now())
                write(path, record)
            print(f"{split} rep={rep} {name} {case['network']} {case['id']} done", flush=True)
    finally:
        lock.unlink()
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-model", action="store_true")
    parser.add_argument("--check", action="store_true", help="One contract check per task on validation, no experiment records")
    args = parser.parse_args()
    if args.prepare_model:
        prepare(); return
    methods.BASELINE_TASKS.update(TASKS)  # This process only; frozen package files are unchanged.
    packages = {d.metadata["Name"]: d.version for d in metadata.distributions()}
    env = {"python": sys.version, "platform": platform.platform(), "packages": packages,
           "cpu_count": os.cpu_count(), "threads": 2, "device": "cpu"}
    cfg = {**CFG, "model_provenance": read(OUT / "model/provenance.json")}
    write(OUT / "settings.json", {"configuration": cfg, "environment": env, "adapter_sha256": file_hash(__file__),
        "started_utc": now(), "seeds": [20261031, 20261101, 20261102], "repetitions": 3,
        "metrics_protocol": "docs/protocol.md", "supplementary": True})
    t = time.perf_counter()
    forecaster = load_forecaster()
    write(OUT / "model/load.json", {"load_s": time.perf_counter()-t, "device": "cpu", "dtype": "float32"})
    if args.check:
        dataset = DATA / "core-20261031"
        verify_dataset(dataset)
        checks = []
        for name, task in TASKS.items():
            case = next(c for c in read(dataset / "cases.json")
                        if c["task"] == task and c["split"] == "validation" and c["network"] == "grid")
            hour = case.get("decision_hours", [None])[0]
            obs = methods.visible(read(dataset / "observations" / (case["id"] + ".json")), hour)
            if task == "leak":
                assert len(obs["hours"]) == hour + 1
            state = fit_leak(dataset, "grid", CFG["training_seed"]) if task == "leak" else forecaster
            pred, _ = prediction(name, state, obs, CFG["training_seed"])
            methods.validate(pred, obs)
            checks.append({"method": name, "case_id": case["id"], "schema": "ok"})
        write(OUT / "contract-check.json", checks)
        print(checks, flush=True)
        return
    source_hashes = {str(p): file_hash(p) for seed in (20261031, 20261101, 20261102)
                     for p in (DATA / f"core-{seed}" / "manifest.json", DATA / f"core-{seed}" / "freeze.json")}
    write(OUT / "original-assets.json", source_hashes)
    campaigns = []
    # Fixed settings saved before scoring test labels. Validation is reported separately.
    for seed in (20261031, 20261101, 20261102):
        dataset = DATA / f"core-{seed}"
        for split in ("validation", "test"):
            paths = []
            for rep in range(3):
                config = {**cfg, "training_seed": CFG["training_seed"]+rep}
                for name in TASKS:
                    paths.append(run(dataset, name, split, rep, config, forecaster, env))
            evaluation = scoring.evaluate(dataset, paths, OUT / str(seed) / split / "evaluations", bootstrap_count=10000)
            scoring.report(evaluation)
            campaigns.append({"seed": seed, "split": split, "evaluation": str(evaluation), "runs": list(map(str, paths))})
            write(OUT / "campaigns.json", campaigns)
    assert all(file_hash(p) == expected for p, expected in source_hashes.items()), "Original assets changed"
    write(OUT / "DONE.json", {"completed_utc": now(), "campaigns": campaigns, "original_assets_preserved": True})


if __name__ == "__main__":
    main()
