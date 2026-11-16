"""Offline event-level metrics and paired bootstrap on independent episodes."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .methods import SYSTEM, PRIMARY, TASKS, fit, prompt, validate, visible, supports
from .clustering import scores as cluster_scores
from .storage import code_hash, digest, read, verify_dataset, write

SCORER_VERSION = "1.2"


def case_score(case, records, truth, scale=None, geometry=None):
    expected = case.get("decision_hours", [None])
    by_hour = {r["hour"]: r for r in records}
    complete = all(h in by_hour and by_hour[h]["status"] == "ok" for h in expected)
    base = {"case_id": case["id"], "unit": case["unit"], "network": case["network"], "task": case["task"],
            "split": case["split"], "condition": case["condition"], "complete": int(complete),
            "expected_decisions": len(expected), "observed_decisions": len(by_hour),
            "failed_decisions": sum(r["status"] != "ok" for r in records),
            "latency_s": sum(r.get("elapsed_s", 0) or 0 for r in records) if records and all(r.get("elapsed_s") is not None for r in records) else None, "success": 0}
    if case["task"] == "cluster":
        base.update(ari=None, nmi=None)
        if complete:
            values = cluster_scores(by_hour[None]["prediction"], truth, sorted(truth["assignments"]))
            base.update(**values, success=int(values["ari"] >= 1 - 1e-12))
        return base
    if case["task"] == "forecast":
        base.update(mae=None, rmse=None, mase=None, smape_pct=None, wape_pct=None, horizon_abs_errors=None)
        if complete:
            predicted = np.asarray(by_hour[None]["prediction"]["forecast_L_s"])
            errors = predicted - np.asarray(truth["future_L_s"])
            mae = float(np.abs(errors).mean())
            mase = mae / scale if scale and scale > 1e-12 else None
            actual = np.asarray(truth["future_L_s"])
            denominator = np.abs(actual)+np.abs(predicted)
            smape = float(200*np.divide(np.abs(errors), denominator, out=np.zeros_like(errors), where=denominator>0).mean())
            wape = float(100*np.abs(errors).sum()/np.abs(actual).sum()) if np.abs(actual).sum()>0 else None
            base.update(mae=mae, rmse=float(np.sqrt(np.mean(errors ** 2))), mase=mase,
                        smape_pct=smape, wape_pct=wape,
                        horizon_abs_errors=np.abs(errors).tolist(), success=int(mase is not None and mase <= 1))
        return base
    base.update(event=int(truth["has_leak"]), tp=0, fp=0, fn=int(truth["has_leak"]),
                detected=0, localized1=0, localized3=0, delay_h=None, distance_m=None,
                normal_days=((truth["start_hour"] if truth["has_leak"] else 24) - expected[0]) / 24)
    base["normal_days"] = max(0, base["normal_days"])
    if not complete:
        return base
    prev = False
    first = None
    for hour in expected:
        output = by_hour[hour]["prediction"]
        alarm = output["alarm"]
        active = truth["has_leak"] and truth["start_hour"] <= hour < truth["end_hour"]
        if alarm and not active and not prev:
            base["fp"] += 1
        if alarm and active and first is None:
            first = (hour, output)
        prev = alarm
    if first:
        hour, pred = first
        ranks = pred["ranking"]
        hit1, hit3 = int(ranks[0] == truth["zone"]), int(truth["zone"] in ranks[:3])
        base.update(tp=1, fn=0, detected=1, localized1=hit1, localized3=hit3,
                    delay_h=hour - truth["start_hour"], success=hit1,
                    distance_m=geometry["distances_m"][truth["zone"]][ranks[0]] if geometry else None)
    elif not truth["has_leak"]:
        base["success"] = int(base["fp"] == 0)
    return base


def average(values):
    values = [x for x in values if x is not None]
    return float(np.mean(values)) if values else None


def metrics(rows):
    valid = [r for r in rows if r["complete"]]
    result = {"n": len(rows), "n_valid": len(valid), "validity": len(valid) / len(rows) if rows else None,
              "latency_s": average([r["latency_s"] for r in rows]), "success": average([r["success"] for r in rows])}
    if rows[0]["task"] in ("forecast", "cluster"):
        for metric in (("mae", "rmse", "mase", "smape_pct", "wape_pct") if rows[0]["task"] == "forecast" else ("ari", "nmi")):
            result[metric] = average([r[metric] for r in valid])
        return result
    tp, fp, fn = (sum(r[x] for r in valid) for x in ("tp", "fp", "fn"))
    result.update(precision=tp / (tp+fp) if tp+fp else None, recall=tp / (tp+fn) if tp+fn else None,
                  f1=2*tp / (2*tp+fp+fn) if 2*tp+fp+fn else None,
                  false_alarms_per_day=sum(r["fp"] for r in valid) / sum(r["normal_days"] for r in valid)
                  if sum(r["normal_days"] for r in valid) > 0 else None,
                  delay_h=average([r["delay_h"] for r in valid]),
                  localized_top1_conditional=sum(r["localized1"] for r in valid) / tp if tp else None,
                  localized_top3_conditional=sum(r["localized3"] for r in valid) / tp if tp else None,
                  overall_detection_localization_top1=sum(r["localized1"] for r in rows) / sum(r["event"] for r in rows)
                  if sum(r["event"] for r in rows) else None,
                  overall_detection_localization_top3=sum(r["localized3"] for r in rows) / sum(r["event"] for r in rows)
                  if sum(r["event"] for r in rows) else None,
                  localization_distance_m=average([r["distance_m"] for r in valid]),
                  events_all=sum(r["event"] for r in rows), events_valid=tp+fn, detected_events=tp)
    return result


def bootstrap(rows, metric, count, rng):
    # Independent dataset realizations; repeated inference/training seeds stay in the same unit block.
    groups = defaultdict(list)
    for row in rows:
        groups[row["unit"]].append(row)
    units = list(groups)
    values = []
    for _ in range(count):
        sample = [r for i in rng.integers(0, len(units), len(units)) for r in groups[units[i]]]
        value = metrics(sample).get(metric)
        if value is not None:
            values.append(value)
    return [float(x) for x in np.quantile(values, [0.025, 0.975])] if values else [None, None]


def table(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader(); writer.writerows(rows)


def evaluate(dataset, run_dirs, output, bootstrap_count=1000):
    dataset, output = Path(dataset), Path(output)
    manifest = verify_dataset(dataset)
    manifests = [read(Path(p) / "manifest.json") for p in run_dirs]
    if len({m["id"] for m in manifests}) != len(manifests):
        raise ValueError("Duplicate run directories")
    arms = {}
    replicate_keys = set()
    for m in manifests:
        ident = m["identity"]
        for method in ident["methods"]:
            label = ident["endpoint"]["label"] + ":" + ident["endpoint"]["mode"] if method == "llm" else method
            descriptor = ident["endpoint"] if method == "llm" else {k:v for k,v in ident["baseline_config"].items() if k != "training_seed"}
            if label in arms and arms[label] != descriptor:
                raise ValueError("Different configurations share a method label; evaluate separately or rename the model label")
            arms[label] = descriptor
            for case in ident["cases"]:
                if not supports(method, case["task"]): continue
                key = (label, case["id"], ident["repetition"], ident.get("generation_seed"))
                if key in replicate_keys:
                    raise ValueError("Overlapping run selections reuse the same method/case/repetition/seed")
                replicate_keys.add(key)
    for m in manifests:
        if digest(m["identity"]) != m["id"] or m["identity"]["dataset_id"] != manifest["id"]:
            raise ValueError("Run/dataset identity mismatch")
    from .storage import file_hash
    source_hashes = {f"{m['id'][:20]}/{p.relative_to(Path(root)).as_posix()}": file_hash(p)
                     for root, m in zip(run_dirs, manifests) for p in sorted(Path(root).rglob("*.json"))}
    evaluation_id = digest({"runs": sorted(m["id"] for m in manifests), "data": manifest["id"],
                            "scorer": SCORER_VERSION, "code_hash": code_hash(), "bootstrap": bootstrap_count,
                            "result_hashes": source_hashes})
    output = output / evaluation_id[:20]
    output.mkdir(parents=True, exist_ok=True)
    scored, failures, coverage = [], [], []
    for rundir, rm in zip(run_dirs, manifests):
        root = Path(rundir)
        records = [read(p) for p in sorted((root / "records").glob("*.json"))]
        grouped = defaultdict(list)
        allowed = {c["id"]: c for c in rm["identity"]["cases"]}
        seen = set()
        for r in records:
            expected_id = digest([rm["id"], r["case"]["id"], r["method"], r["hour"]])[:24]
            if r["id"] != expected_id or r["id"] in seen or r["run_id"] != rm["id"] or r["case"] != allowed.get(r["case"]["id"]):
                raise ValueError("Corrupt/duplicate/foreign record")
            label = rm["identity"]["endpoint"]["label"]+":"+rm["identity"]["endpoint"]["mode"] if r["method"]=="llm" else r["method"]
            if r["method"] not in rm["identity"]["methods"] or not supports(r["method"], r["case"]["task"]) or r["hour"] not in r["case"].get("decision_hours", [None]) or r["method_label"]!=label:
                raise ValueError("Foreign method/hour/label")
            seen.add(r["id"])
            if r["status"] == "ok":
                obs = visible(read(dataset / "observations" / f"{r['case']['id']}.json"), r["hour"])
                validate(r["prediction"], obs)
                if r["observation_hash"] != digest(obs):
                    raise ValueError("Observation digest changed")
                if r["prompt_hash"] != digest({"system": SYSTEM, "user": prompt(obs)}):
                    raise ValueError("Prompt digest changed; use original scorer/code for old prompts")
            else:
                failures.append({"run": rm["id"], "case": r["case"]["id"], "method": r["method_label"], "hour": r["hour"], "status": r["status"]})
            grouped[(r["case"]["id"], r["method"])].append(r)
        for case in rm["identity"]["cases"]:
            for method in rm["identity"]["methods"]:
                if not supports(method, case["task"]):
                    continue
                label = rm["identity"]["endpoint"]["label"] + ":" + rm["identity"]["endpoint"]["mode"] if method == "llm" else method
                rs = grouped[(case["id"], method)]
                truth = read(dataset / "evaluator" / f"{case['id']}.json")
                scale, geometry = None, None
                if case["task"] == "forecast":
                    scale = fit(dataset, "forecast", case["network"], "seasonal-naive", rm["identity"]["baseline_config"])["scale"]
                elif case["task"] == "leak":
                    geometry = read(dataset / "evaluator" / f"{case['network']}_geometry.json")
                row = case_score(case, rs, truth, scale, geometry)
                row.update(method=label, run=rm["id"], repetition=rm["identity"]["repetition"],
                           generation_seed=rm["identity"].get("generation_seed"))
                scored.append(row)
        expected = sum(len(c.get("decision_hours", [None])) for c in rm["identity"]["cases"]
                       for method in rm["identity"]["methods"] if supports(method, c["task"]))
        coverage.append({"run": rm["id"], "expected": expected, "observed": len(records), "missing": expected-len(records),
                         "failed": sum(r["status"] != "ok" for r in records)})
    groups = defaultdict(list)
    for row in scored:
        groups[(row["task"], row["network"], row["condition"], row["split"], row["method"])].append(row)
    aggregates = []
    for key, rows in sorted(groups.items()):
        metadata = dict(zip(("task", "network", "condition", "split", "method"), key))
        aggregate = metrics(rows)
        primary = PRIMARY[key[0]]
        ci = bootstrap(rows, primary, bootstrap_count, np.random.default_rng(20261002))
        aggregate.update(primary_metric=primary, ci_low=ci[0], ci_high=ci[1], independent_units=len({r["unit"] for r in rows}))
        aggregates.append({**metadata, **aggregate})
    paired = []
    strata = defaultdict(dict)
    for key, rows in groups.items():
        strata[key[:4]][key[4]] = rows
    for key, methods in strata.items():
        names = sorted(methods)
        for i, a in enumerate(names):
            for b in names[i+1:]:
                aa = defaultdict(list); bb = defaultdict(list)
                for r in methods[a]: aa[r["unit"]].append(r)
                for r in methods[b]: bb[r["unit"]].append(r)
                all_units = sorted(set(aa) & set(bb))
                units = [u for u in all_units if all(r["complete"] for r in aa[u] + bb[u])]
                if not units:
                    continue
                metric = PRIMARY[key[0]]
                ra = [r for u in units for r in aa[u]]; rb = [r for u in units for r in bb[u]]
                ma, mb = metrics(ra)[metric], metrics(rb)[metric]
                rng = np.random.default_rng(20261002)
                diffs = []
                for _ in range(bootstrap_count):
                    selected = [units[j] for j in rng.integers(0, len(units), len(units))]
                    va = metrics([r for u in selected for r in aa[u]])[metric]
                    vb = metrics([r for u in selected for r in bb[u]])[metric]
                    if va is not None and vb is not None:
                        diffs.append(va-vb)
                ci = np.quantile(diffs, [0.025, 0.975]).tolist() if diffs else [None, None]
                paired.append({**dict(zip(("task", "network", "condition", "split"), key)), "method_a": a, "method_b": b,
                               "metric": metric, "difference_a_minus_b": ma-mb if ma is not None and mb is not None else None,
                               "ci_low": ci[0], "ci_high": ci[1], "paired_units": len(units), "shared_units_all": len(all_units),
                               "valid_a": sum(r["complete"] for r in ra), "valid_b": sum(r["complete"] for r in rb)})
    horizon = []
    for key, rows in groups.items():
        if key[0] == "forecast":
            errors = [r["horizon_abs_errors"] for r in rows if r["complete"]]
            if errors:
                horizon.extend({"network": key[1], "split": key[3], "method": key[4], "hour": h+1, "mae_L_s": float(v)}
                               for h, v in enumerate(np.mean(errors, axis=0)))
    table(output / "cases.csv", [{k: v for k, v in r.items() if k != "horizon_abs_errors"} for r in scored])
    table(output / "metrics.csv", aggregates); table(output / "paired.csv", paired)
    table(output / "failures.csv", failures); table(output / "coverage.csv", coverage); table(output / "horizons.csv", horizon)
    repeat_groups = defaultdict(list)
    for row in scored:
        repeat_groups[(row["method"], row["case_id"])].append(row)
    variability = []
    for (method, case_id), rs in sorted(repeat_groups.items()):
        metric = PRIMARY[rs[0]["task"]]
        values = [metrics([r]).get(metric) for r in rs if r["complete"]]
        values = [v for v in values if v is not None]
        variability.append({"method": method, "case_id": case_id, "unit": rs[0]["unit"], "task": rs[0]["task"],
                            "replicates": len(rs), "valid_replicates": sum(r["complete"] for r in rs), "metric": metric,
                            "mean": average(values), "std": float(np.std(values, ddof=1)) if len(values)>1 else None})
    table(output / "replicates.csv", variability)
    from .performance import export_performance
    efficiency, decision_metrics = export_performance(dataset, run_dirs, output, table)
    task_tables = {"metrics": aggregates, "paired": paired,
                   "cases": [{k:v for k,v in r.items() if k != "horizon_abs_errors"} for r in scored],
                   "replicates": variability, "efficiency": efficiency, "decision_metrics": decision_metrics}
    for task in TASKS:
        folder = output/"tasks"/task
        folder.mkdir(parents=True, exist_ok=True)
        for name, rows in task_tables.items(): table(folder/(name+".csv"), [r for r in rows if r["task"]==task])
    write(output / "evaluation.json", {"id": evaluation_id, "dataset_id": manifest["id"], "scorer_version": SCORER_VERSION,
          "dataset_config": manifest["config"], "method_configurations": arms,
          "inference_code_hash": manifests[0]["identity"]["code_hash"] if len({m["identity"]["code_hash"] for m in manifests})==1 else None,
          "run_inference_code_hashes": sorted({m["identity"]["code_hash"] for m in manifests}),
          "run_ids": [m["id"] for m in manifests], "source_hashes": source_hashes, "bootstrap_replicates": bootstrap_count,
          "aggregates": aggregates, "paired": paired, "coverage": coverage, "stage": manifest["config"]["stage"]})
    return output


def report(evaluation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    evaluation = Path(evaluation)
    result = read(evaluation / "evaluation.json")
    rows = result["aggregates"]
    captions = []
    for task in TASKS:
        taskrows = [r for r in rows if r["task"] == task]
        if not taskrows:
            continue
        fig, ax = plt.subplots(figsize=(9, max(3, 0.40 * len(taskrows) + 1)))
        labels = []
        for i, row in enumerate(taskrows):
            value = row[row["primary_metric"]]
            labels.append(f"{row['network']} / {row['condition']} / {row['split']} / {row['method']} (n={row['independent_units']})")
            if value is not None:
                low, high = row["ci_low"], row["ci_high"]
                ax.plot([low, high], [i, i], color="#26747c")
                ax.scatter(value, i, color="#26747c", zorder=3)
            else:
                ax.text(0, i, "no valid predictions", va="center")
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_xlabel({"leak": "Event F1, conditional on complete valid episode", "forecast": "MAE (L/s), valid forecasts", "cluster": "Adjusted Rand index, valid synthetic partitions"}[task])
        ax.set_title(f"{result['stage']} — {task} — simulation-based benchmark")
        ax.grid(axis="x", alpha=0.2); fig.tight_layout()
        for ext in ("png", "svg", "pdf"):
            fig.savefig(evaluation / f"{task}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)
        captions.append(f"{task}: source metrics.csv, evaluation {result['id']}; 95% percentile bootstrap by independent generator realization, {result['bootstrap_replicates']} resamples. n labels count independent units. Small smoke/pilot samples are functional/exploratory, not confirmatory. Failures and missing cases are in coverage.csv; conditional metrics must be accompanied by validity and overall success.")
    # A fragment for inclusion in the author's document, not a standalone paper.
    def escape(text):
        for a, b in (("\\", "\\textbackslash{}"), ("_", "\\_"), ("%", "\\%"), ("&", "\\&"), ("#", "\\#")):
            text = text.replace(a, b)
        return text
    lines = ["% Generated table fragment; see captions.md for protocol and uncertainty.", "\\begin{tabular}{lllllrr}", "Task & Network & Condition & Method & Split & N & Primary \\\\", "\\hline"]
    for row in rows:
        v = row[row["primary_metric"]]
        lines.append(" & ".join(escape(str(row[k])) for k in ("task", "network", "condition", "method", "split"))
                     + f" & {row['n']} & " + (f"{v:.4g}" if v is not None else "--") + " \\\\")
    lines.append("\\end{tabular}")
    (evaluation / "table_fragment.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (evaluation / "captions.md").write_text("\n\n".join(captions), encoding="utf-8")
    return evaluation
