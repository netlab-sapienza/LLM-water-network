"""Auditable experiment register: actual inputs, messages, configurations and access regimes."""
from pathlib import Path
from .methods import SYSTEM, INSTRUCTIONS, prompt, visible
from .storage import canonical, digest, read, verify_dataset, write

MODEL_FIELDS = ("model", "checkpoint", "revision", "architecture", "parameter_count", "quantization",
                "backend", "backend_version", "chat_template", "tokenizer_revision", "context_limit_tokens",
                "reasoning_mode", "provider_scope", "platform", "generation", "seed_supported", "model_metadata")
ACCESS = {
    "llm": "Supplied observation only; zero-shot; tools only if explicitly enabled in endpoint",
    "physics-residual": "Observation plus threshold calibrated on nominal validation units",
    "extra-trees": "Observation plus labeled train and fixed 0.5 detection threshold",
    "seasonal-naive": "Observation only for prediction; train differences used only for evaluator MASE scale",
    "hist-gradient-boosting": "Observation plus separate train demand series",
    "shape-kmeans": "Observation only; no latent labels or train fitting",
    "shape-dtw-average": "Observation only; no latent labels or train fitting"}
ACCESS.update({"residual-svm": "Observation plus labeled train; fixed RBF SVC; margin score uncalibrated",
               "lasso-direct": "Observation plus separate train series; train-only scaling; fixed regularization",
               "forecast-ensemble": "Observation plus train-fitted members; fixed equal weights; no test tuning",
               "shape-msm-medoids": "Observation only; MSM cost fixed; exact medoid selection; no latent labels"})


def catalog(dataset, run_dirs, output):
    dataset, output = Path(dataset), Path(output)
    dm = verify_dataset(dataset)
    manifests = [read(Path(p)/"manifest.json") for p in run_dirs]
    cases = {c["id"]:c for c in read(dataset/"cases.json")}
    packets, experiments = {}, []
    for root, manifest in zip(map(Path,run_dirs), manifests):
        ident = manifest["identity"]
        if digest(ident)!=manifest["id"] or ident["dataset_id"]!=dm["id"]:
            raise ValueError("Foreign/corrupt run in catalog")
        endpoint = ident.get("endpoint")
        experiments.append({"run_id":manifest["id"], "case_count":len(ident["cases"]), "methods":ident["methods"],
                            "tasks":sorted({c["task"] for c in ident["cases"]}),
                            "networks":sorted({c["network"] for c in ident["cases"]}),
                            "splits":sorted({c["split"] for c in ident["cases"]}),
                            "repetition":ident["repetition"], "generation_seed":ident.get("generation_seed"),
                            "baseline_config":ident["baseline_config"], "environment":ident["environment"],
                            "model_setup":{k:endpoint.get(k) for k in MODEL_FIELDS} if endpoint else None,
                            "endpoint_config":endpoint, "inference_code_hash":ident["code_hash"],
                            "comparison_track":manifest.get("comparison_track", "local API" if endpoint else "algorithm baseline"),
                            "technical_fixture":bool(endpoint and endpoint.get("technical_fixture")),
                            "raw_results_path":str(root)})
        for recordpath in (root/"records").glob("*.json"):
            r = read(recordpath); c = r["case"]
            if c!=cases.get(c["id"]): raise ValueError("Foreign case")
            obs = visible(read(dataset/"observations"/(c["id"]+".json")), r["hour"])
            messages = [{"role":"system", "content":SYSTEM}, {"role":"user", "content":prompt(obs)}]
            if r["observation_hash"]!=digest(obs) or r["prompt_hash"]!=digest({"system":SYSTEM,"user":prompt(obs)}):
                raise ValueError("Input/prompt differs from contract")
            tracepath = root/"traces"/(r["id"]+".json")
            if r["method"]=="llm" and tracepath.exists() and read(tracepath)["messages"][:2]!=messages:
                raise ValueError("Actual request messages differ from declared prompt")
            key = digest([dm["id"], c["id"], r["hour"], r["prompt_hash"]])[:24]
            packets[key] = {"request_id":key, "case_id":c["id"], "task":c["task"], "network":c["network"],
                            "split":c["split"], "condition":c["condition"], "hour":r["hour"],
                            "observation_hash":r["observation_hash"], "prompt_hash":r["prompt_hash"],
                            "observation":obs, "messages":messages}
    result = {"dataset_id":dm["id"], "dataset_config":dm["config"], "parent_dataset_id":dm.get("parent_dataset_id"),
              "network_sources":dm.get("networks"), "demand_scales":dm.get("demand_scales"),
              "system_prompt":SYSTEM, "task_instructions":INSTRUCTIONS, "information_access":ACCESS,
              "experiments":experiments, "unique_decisions":len(packets),
              "equality_note":"Same numeric test observations; training/calibration access differs and is disclosed. Product system-role delivery may differ.",
              "unknown_metadata_policy":"Null/UNKNOWN means unavailable, never inferred from product name"}
    result["id"] = digest(result)
    write(output/"experiment_register.json",result)
    (output/"inputs_and_prompts.jsonl").write_text("".join(canonical(packets[k])+"\n" for k in sorted(packets)),encoding="utf-8")
    return output
