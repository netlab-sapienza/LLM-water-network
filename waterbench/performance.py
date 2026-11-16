"""Offline resource accounting includes invalid responses; never infer unmeasured resources."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from .storage import read


def number(value):
    return type(value) in (int, float) and np.isfinite(value) and value >= 0


def decision_cost(root, record):
    case = record["case"]
    trace_path = root/"traces"/(record["id"]+".json")
    trace = read(trace_path) if trace_path.exists() else {}
    responses = trace.get("responses", [])
    usages = trace.get("usage", record.get("usage", [])) or []
    timings = trace.get("http_request_elapsed_s", record.get("http_request_elapsed_s", [])) or []
    row = {"run": record["run_id"], "record_id": record["id"], "case_id": case["id"],
           "task": case["task"], "network": case["network"], "condition": case["condition"],
           "split": case["split"], "method": record["method_label"], "hour": record["hour"],
           "status": record["status"], "elapsed_s": record.get("elapsed_s"),
           "client_cpu_s": record.get("client_cpu_s"), "api_calls": len(responses),
           "http_s": sum(timings) if timings and all(number(t) for t in timings) else None,
           "truncated_calls": sum(c.get("finish_reason")=="length" for r in responses for c in r.get("choices", [])),
           "inflight_unaccounted": int(bool(trace.get("inflight"))),
           "reasoning_text_observed": int(any((c.get("message") or {}).get("reasoning_content") for r in responses for c in r.get("choices", [])))}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [u.get(field) if isinstance(u,dict) else None for u in usages]
        row[field] = sum(values) if values and all(number(v) for v in values) and len(values)==len(responses) else None
    reasons = [(u.get("completion_tokens_details") or {}).get("reasoning_tokens") if isinstance(u,dict) else None for u in usages]
    row["reported_reasoning_tokens"] = sum(reasons) if reasons and all(number(v) for v in reasons) else None
    row["reasoning_accounting_anomaly"] = int(row["reasoning_text_observed"] and row["reported_reasoning_tokens"]==0)
    # Cloud UI import: latency/tokens may be genuinely unavailable. Never replace them with zero.
    row.update(peak_vram_bytes=None, energy_joules=None, actual_cost_usd=None)
    return row


def export_performance(dataset, run_dirs, output, table):
    costs, fits, decisions = [], [], []
    for path in run_dirs:
        root = Path(path)
        manifest = read(root/"manifest.json")
        for p in sorted((root/"fits").glob("*.json")):
            f = read(p)
            fits.append({"run":manifest["id"], "method":f["method"], "task":f["task"], "network":f["network"],
                         "fit_s":f["fit_s"], "fit_cpu_s":f.get("fit_cpu_s"), "training_seed":f["training_seed"]})
        for p in sorted((root/"records").glob("*.json")):
            r = read(p); costs.append(decision_cost(root,r))
            if r["case"]["task"] == "leak" and r["status"]=="ok":
                truth = read(Path(dataset)/"evaluator"/(r["case"]["id"]+".json"))
                active = bool(truth["has_leak"] and truth["start_hour"]<=r["hour"]<truth["end_hour"])
                decisions.append({"run":manifest["id"], "unit":r["case"]["unit"], "case_id":r["case"]["id"],
                    **{k:r["case"][k] for k in ("task","network","condition","split")},
                    "method":r["method_label"], "hour":r["hour"], "actual":int(active),
                    "predicted":int(r["prediction"]["alarm"]), "score":r["prediction"]["score"]})
    keys = ("task","network","condition","split","method")
    groups = defaultdict(list)
    for r in costs: groups[tuple(r[k] for k in keys)].append(r)
    aggregates = []
    for key, rows in sorted(groups.items()):
        row = dict(zip(keys,key)); row.update(observed_decisions=len(rows), valid_decisions=sum(r["status"]=="ok" for r in rows),
            failed_decisions=sum(r["status"]!="ok" for r in rows), api_calls=sum(r["api_calls"] for r in rows),
            truncated_calls=sum(r["truncated_calls"] for r in rows),
            reasoning_accounting_anomalies=sum(r["reasoning_accounting_anomaly"] for r in rows),
            inflight_unaccounted=sum(r["inflight_unaccounted"] for r in rows))
        for name in ("elapsed_s","http_s","prompt_tokens","completion_tokens","total_tokens","client_cpu_s"):
            values = [r[name] for r in rows if number(r[name])]
            row[name+"_known_decisions"] = len(values)
            row[name+"_observed_sum"] = float(sum(values)) if values else None
            row[name+"_total"] = float(sum(values)) if len(values)==len(rows) else None
            if name in ("elapsed_s","http_s"):
                for label,q in (("median",.5),("p90",.9),("p95",.95)):
                    row[name+"_"+label] = float(np.quantile(values,q)) if values else None
        valid = row["valid_decisions"]
        row["elapsed_s_per_valid_decision_including_failures"] = row["elapsed_s_total"]/valid if valid and row["elapsed_s_total"] is not None else None
        row["completion_tokens_per_http_s"] = row["completion_tokens_total"]/row["http_s_total"] if row["completion_tokens_total"] is not None and row["http_s_total"] else None
        row.update(peak_vram_bytes=None, energy_joules=None, actual_cost_usd=None)
        aggregates.append(row)
    dgroups = defaultdict(list)
    for r in decisions: dgroups[tuple(r[k] for k in keys)].append(r)
    decision_metrics = []
    for key, rows in sorted(dgroups.items()):
        y = np.array([r["actual"] for r in rows]); p = np.array([r["predicted"] for r in rows]); s = np.array([r["score"] for r in rows])
        tp,tn,fp,fn = (int(np.sum(mask)) for mask in ((y==1)&(p==1),(y==0)&(p==0),(y==0)&(p==1),(y==1)&(p==0)))
        sensitivity = tp/(tp+fn) if tp+fn else None
        specificity = tn/(tn+fp) if tn+fp else None
        both = len(set(y))==2
        decision_metrics.append({**dict(zip(keys,key)), "valid_decisions":len(rows), "independent_units":len({r["unit"] for r in rows}),
            "tp":tp,"tn":tn,"fp":fp,"fn":fn, "accuracy":float(np.mean(y==p)),
            "sensitivity":sensitivity,"specificity":specificity,
            "balanced_accuracy":(sensitivity+specificity)/2 if both else None,
            "roc_auc":float(roc_auc_score(y,s)) if both else None,
            "average_precision":float(average_precision_score(y,s)) if both else None,
            "brier_score_diagnostic":float(np.mean((s-y)**2))})
    table(output/"efficiency_decisions.csv",costs); table(output/"efficiency.csv",aggregates)
    table(output/"fit_costs.csv",fits); table(output/"leak_decisions.csv",decisions)
    table(output/"decision_metrics.csv",decision_metrics)
    for task in ("leak","forecast","cluster"):
        folder = output/"tasks"/task; folder.mkdir(parents=True,exist_ok=True)
        for name,rows in (("efficiency_decisions",costs),("fit_costs",fits),("decisions",decisions)):
            table(folder/(name+".csv"),[r for r in rows if r["task"]==task])
    return aggregates, decision_metrics
