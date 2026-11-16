"""Sequential inference with durable per-attempt traces and immutable run identities."""
import datetime as dt
import ipaddress
import json
import os
import shutil
import socket
import time
import urllib.error
from pathlib import Path
from urllib.parse import urlparse

from .methods import INSTRUCTIONS, SYSTEM, TOOL, fit, predict, prompt, request, summary, validate, visible, supports
from .storage import canonical, code_hash, digest, environment, read, verify_dataset, write
from .backend import IdentityError, guard, provenance, response_identity, utc


def endpoint_config(path):
    value = read(path)
    parsed = urlparse(value["base_url"])
    if parsed.scheme not in ("http", "https") or parsed.username or parsed.password or parsed.query:
        raise ValueError("Expected credential-free HTTP(S) base_url")
    if value.get("provider_scope") != "local":
        raise ValueError("Cloud disabled. Separate authorized budget/protocol required.")
    addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 80, type=socket.SOCK_STREAM)
    if not all(ipaddress.ip_address(a[4][0]).is_private or ipaddress.ip_address(a[4][0]).is_loopback for a in addresses):
        raise ValueError("Local endpoint must resolve only to loopback/private addresses")
    if any(k.lower() in ("api_key", "token", "password") for k in value):
        raise ValueError("Store credentials in the named environment variable")
    if value["mode"] not in ("direct", "tools"):
        raise ValueError("Unsupported LLM mode")
    if not 0 < value["timeout_s"] <= 3600 or not 1 <= value["max_tool_calls"] <= 4:
        raise ValueError("Invalid time/tool budgets")
    if value.get("context_limit_tokens", 0) <= 0:
        raise ValueError("Declare context capacity; verify tokenizer on server")
    if type(value["generation"].get("max_tokens")) is not int or value["generation"]["max_tokens"] <= 0:
        raise ValueError("Declare a positive max_tokens output budget")
    if not value.get("label") or not value.get("model"):
        raise ValueError("Endpoint label/model required")
    if {"model", "messages", "tools", "stream", "enable_tools", "enabled_tools", "session_id"} & value["generation"].keys():
        raise ValueError("generation cannot override identity/messages/tools or enable server tools/streaming")
    if value.get("loaded_guard") and not isinstance(value["loaded_guard"].get("fingerprint"), dict):
        raise ValueError("Invalid discovery fingerprint")
    if "response_model_ids" in value and (not isinstance(value["response_model_ids"], list) or not value["response_model_ids"]
                                         or not all(isinstance(x, str) and x for x in value["response_model_ids"])):
        raise ValueError("response_model_ids must be a nonempty list of explicit IDs")
    return value


def infer(endpoint, obs, trace_path):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(obs)}]
    trace = {"messages": messages.copy(), "responses": [], "tools": [], "usage": [], "mode": endpoint["mode"],
             "effective_generation": endpoint.get("generation", {})}
    used = 0
    while True:
        if "context_limit_tokens" in endpoint:
            # shortcut: conservative UTF-8 size bound; verify exact backend tokenizer in the pilot.
            budget = len(canonical(messages).encode()) + len(canonical([TOOL] if endpoint["mode"] == "tools" else []).encode())
            if budget + endpoint["generation"].get("max_tokens", 0) > endpoint["context_limit_tokens"]:
                raise ValueError("Conservative context bound exceeded; no silent truncation")
        # Durable 'started' record precedes every possibly billable/request-producing call.
        trace["inflight"] = True
        write(trace_path, trace)
        before = guard(endpoint)
        trace.setdefault("model_checks", []).append({"before": before})
        write(trace_path, trace)
        request_start = time.perf_counter()
        response = request(endpoint, messages, [TOOL] if endpoint["mode"] == "tools" else None)
        trace.setdefault("http_request_elapsed_s", []).append(time.perf_counter() - request_start)
        trace["responses"].append(response)
        trace["usage"].append(response.get("usage"))
        trace["inflight"] = False
        write(trace_path, trace)
        response_identity(endpoint, response)
        trace["model_checks"][-1]["after"] = guard(endpoint)
        trace.setdefault("response_models", []).append(response.get("model"))
        write(trace_path, trace)
        message = response["choices"][0]["message"]
        if response["choices"][0].get("finish_reason") == "length":
            raise ValueError("Output truncated at token budget")
        calls = message.get("tool_calls") or []
        if not calls:
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("Response content must be text JSON")
            output = json.loads(content)
            return validate(output, obs), trace
        if endpoint["mode"] != "tools" or used + len(calls) > endpoint["max_tool_calls"]:
            raise ValueError("Tool budget/mode violation")
        messages.append({k: v for k, v in message.items() if k in ("role", "content", "tool_calls")})
        for call in calls:
            function = call["function"]
            if function["name"] != "observation_summary" or json.loads(function["arguments"]) != {}:
                raise ValueError("Unsupported tool/arguments")
            tool_start = time.perf_counter()
            result = summary(obs)
            trace["tools"].append({"id": call["id"], "name": function["name"], "arguments": {}, "output": result})
            trace.setdefault("tool_elapsed_s", []).append(time.perf_counter() - tool_start)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": canonical(result)})
            used += 1
        trace["messages"] = messages.copy()
        write(trace_path, trace)


def run(dataset, output_root, methods, splits, tasks, networks, baseline_cfg_path,
        endpoint_path=None, repetition=0, max_cases=None, stop_after=None, session_id=None, session_name=None, generation_seed=None):
    dataset = Path(dataset)
    data_manifest = verify_dataset(dataset)
    baseline_cfg = read(baseline_cfg_path)
    endpoint = endpoint_config(endpoint_path) if endpoint_path else None
    if repetition < 0 or (generation_seed is not None and (not endpoint or endpoint.get("seed_supported") is not True)):
        raise ValueError("Nonnegative repetition required; generation seed requires explicitly verified seed_supported:true")
    if generation_seed is not None and "seed" in endpoint["generation"]:
        raise ValueError("Use either fixed endpoint generation.seed or --generation-seed, not both")
    if "llm" in methods and endpoint is None:
        raise ValueError("--endpoint required for llm; no mock substitutions")
    if "test" in splits and data_manifest["config"]["stage"] == "core-candidate":
        freeze = read(dataset / "freeze.json")
        if freeze["dataset_id"] != data_manifest["id"] or freeze["code_hash"] != code_hash(exclude_scoring=True):
            raise ValueError("Frozen dataset/code changed")
        baseline_equal = {k:v for k,v in freeze["baseline_config"].items() if k!="training_seed"} == {k:v for k,v in baseline_cfg.items() if k!="training_seed"}
        if not baseline_equal or baseline_cfg["training_seed"] not in freeze.get("baseline_seeds", [freeze["baseline_config"]["training_seed"]]) or (endpoint and endpoint not in freeze["endpoints"]):
            raise ValueError("Methods not part of frozen protocol")
        if "llm" in methods and generation_seed not in freeze.get("generation_seeds", [None]):
            raise ValueError("Generation seed not part of frozen protocol")
    cases = [c for c in read(dataset / "cases.json") if c["split"] in splits and c["task"] in tasks and c["network"] in networks]
    if max_cases is not None:
        if max_cases <= 0:
            raise ValueError("max-cases must be positive")
        cases = cases[:max_cases]
    if not cases or not any(supports(m, c["task"]) for m in methods for c in cases):
        raise ValueError("Empty case/method selection")
    env = environment()
    env.pop("code_hash")  # Scorer-only edits must permit offline re-evaluation without fresh inference.
    identity = {"schema": 1, "dataset_id": data_manifest["id"], "code_hash": code_hash(exclude_scoring=True), "environment": env,
                "methods": sorted(methods), "cases": cases, "baseline_config": baseline_cfg,
                "endpoint": endpoint, "repetition": repetition,
                "prompt_version": digest({"system": SYSTEM, "instructions": INSTRUCTIONS})}
    if generation_seed is not None:
        identity["generation_seed"] = generation_seed
    runid = digest(identity)
    root = Path(output_root) / runid[:20]
    root.mkdir(parents=True, exist_ok=True)
    lockpath = root / "writer.lock"
    try:
        lock = lockpath.open("x", encoding="utf-8")
    except FileExistsError:
        raise RuntimeError(f"Writer lock present: {lockpath}; verify process stopped before removing stale lock") from None
    completed = 0
    try:
        lock.write(canonical({"pid": os.getpid(), "hostname": socket.gethostname()})); lock.flush()
        execution = provenance(session_id, session_name)
        execution["initial_model_check"] = guard(endpoint) if endpoint and "llm" in methods else None
        if endpoint and endpoint.get("loaded_guard"):
            snapshot_path = Path(str(endpoint_path) + ".snapshot.json")
            snapshot = read(snapshot_path)
            if snapshot["endpoint_config_hash"] != digest(endpoint):
                raise ValueError("Endpoint changed after discovery; create a new immutable configuration")
            write(root / "model_snapshot.json", snapshot)
        write(root / "executions" / (digest(execution)[:24] + ".json"), execution)
        if (root / "manifest.json").exists() and read(root / "manifest.json")["id"] != runid:
            raise ValueError("Run identity collision")
        write(root / "manifest.json", {"id": runid, "identity": identity,
              "dataset_path_hint": str(dataset), "purpose": data_manifest["config"]["stage"],
              "timing_note": "end-to-end per decision includes HTTP/network/tools and discovery checks; trace HTTP timing isolates completions; no hardware efficiency inference",
              "memory_peak": "not measured; inventory is not process peak VRAM", "retry_policy": "zero retries; interrupted requests are failures"})
        states = {}
        for case in cases:
            obs_all = read(dataset / "observations" / f"{case['id']}.json")
            hours = case.get("decision_hours", [None])
            for method in methods:
                if not supports(method, case["task"]):
                    continue
                key = (case["task"], case["network"], method)
                if method != "llm" and key not in states:
                    t0 = time.perf_counter()
                    cpu0 = time.process_time()
                    states[key] = fit(dataset, *key[:2], method, baseline_cfg)
                    metadata = {"method": method, "network": case["network"], "task": case["task"],
                                "fit_s": time.perf_counter() - t0,
                                "fit_cpu_s": time.process_time() - cpu0,
                                "parameters": {k: v for k, v in states[key].items() if k not in ("model", "detector", "localizer")},
                                "training_seed": baseline_cfg["training_seed"],
                                "training_splits": states[key].get("training_splits", ["train", "validation"] if case["task"] == "leak" else ["train"])}
                    artifact = root / "fits" / ("_".join(key) + ".json")
                    if not artifact.exists():
                        write(artifact, metadata)
                for hour in hours:
                    obs = visible(obs_all, hour)
                    recordid = digest([runid, case["id"], method, hour])[:24]
                    path = root / "records" / f"{recordid}.json"
                    if path.exists():
                        record = read(path)
                        if record["status"] == "started":
                            record.update(status="interrupted", error="Request may have executed; no automatic retry")
                            write(path, record)
                        continue
                    size = sum(p.stat().st_size for p in root.rglob("*.json"))
                    if size > baseline_cfg.get("max_run_mb", 2048) * 1024**2 or shutil.disk_usage(root).free < 1024**3:
                        raise OSError("Run disk budget exceeded or less than 1 GiB free; resume after resolving space")
                    record = {"id": recordid, "run_id": runid, "case": case, "method": method,
                              "method_label": endpoint["label"] + ":" + endpoint["mode"] if method == "llm" else method,
                              "hour": hour, "status": "started", "prediction": None,
                              "prompt_hash": digest({"system": SYSTEM, "user": prompt(obs)}),
                              "observation_hash": digest(obs), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
                    write(path, record)
                    t0 = time.perf_counter()
                    cpu0 = time.process_time()
                    try:
                        if method == "llm":
                            effective = endpoint if generation_seed is None else {**endpoint, "generation": {**endpoint["generation"], "seed": generation_seed}}
                            result, trace = infer(effective, obs, root / "traces" / f"{recordid}.json")
                            record.update(response_models=trace["response_models"], usage=trace["usage"],
                                          http_request_elapsed_s=trace["http_request_elapsed_s"], retries=0)
                        else:
                            result = validate(predict(states[key], obs), obs)
                        record.update(status="ok", prediction=result)
                    except IdentityError as exc:
                        record.update(status="identity_error", error=str(exc), elapsed_s=time.perf_counter()-t0, ended_utc=utc())
                        write(path, record)
                        raise  # Never continue a campaign against an unverified model.
                    except (TimeoutError, socket.timeout):
                        record.update(status="timeout", error="Timeout")
                    except urllib.error.HTTPError as exc:
                        record.update(status="http_error", error=f"HTTP {exc.code}")
                    except urllib.error.URLError:
                        record.update(status="connection_error", error="Endpoint connection failed")
                    except (ValueError, KeyError, TypeError, IndexError) as exc:
                        record.update(status="invalid", error=type(exc).__name__ + ": " + str(exc)[:300])
                    record["elapsed_s"] = time.perf_counter() - t0
                    record["client_cpu_s"] = time.process_time() - cpu0
                    record["ended_utc"] = utc()
                    write(path, record)
                    print(f"{case['network']} {case['task']} {method} {case['split']} h={hour} {record['status']}", flush=True)
                    completed += 1
                    if stop_after and completed >= stop_after:
                        return root
    finally:
        lock.close()
        lockpath.unlink()
    return root
