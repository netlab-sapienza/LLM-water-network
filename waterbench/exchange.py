"""Offline exchange of exact prompts with subscription chat products. No network calls."""
import json
from pathlib import Path

from .methods import SYSTEM, INSTRUCTIONS, prompt, visible, validate
from .storage import code_hash, digest, environment, read, verify_dataset, write


def frozen_test(dataset, manifest, endpoint=None):
    if not (Path(dataset)/"freeze.json").exists():
        raise ValueError("Subscription final test requires freeze.json")
    freeze = read(Path(dataset)/"freeze.json")
    if manifest["config"]["stage"] != "core-candidate" or freeze["dataset_id"]!=manifest["id"] or freeze["code_hash"]!=code_hash(exclude_scoring=True):
        raise ValueError("Subscription test requires matching frozen core/code")
    subscriptions = [e for e in freeze["endpoints"] if e.get("provider_scope")=="subscription"]
    if not subscriptions or (endpoint is not None and endpoint not in subscriptions):
        raise ValueError("Subscription condition not registered in freeze")
    return freeze


def export_cases(dataset, output, baseline_config, splits, tasks, networks, max_cases=None):
    dataset, output = Path(dataset), Path(output)
    manifest = verify_dataset(dataset)
    if output.exists():
        raise ValueError("Packet immutable; choose a new directory")
    if "test" in splits:
        freeze = frozen_test(dataset, manifest)
        if freeze["baseline_config"] != read(baseline_config):
            raise ValueError("Baseline configuration differs from freeze")
    cases = [c for c in read(dataset/"cases.json") if c["split"] in splits and c["task"] in tasks and c["network"] in networks]
    if max_cases is not None:
        if max_cases <= 0: raise ValueError("max-cases must be positive")
        cases = cases[:max_cases]
    if not cases: raise ValueError("Empty selection")
    requests = []
    for case in cases:
        obs_all = read(dataset/"observations"/(case["id"]+".json"))
        for hour in case.get("decision_hours", [None]):
            obs = visible(obs_all, hour)
            messages = [{"role":"system", "content":SYSTEM}, {"role":"user", "content":prompt(obs)}]
            ph = digest({"system":SYSTEM, "user":prompt(obs)})
            requests.append({"request_id":digest([manifest["id"], case["id"], hour, ph])[:24], "case_id":case["id"],
                             "hour":hour, "observation_hash":digest(obs), "prompt_hash":ph, "messages":messages})
    packet = {"dataset_id":manifest["id"], "code_hash":code_hash(exclude_scoring=True),
              "prompt_version":digest({"system":SYSTEM, "instructions":INSTRUCTIONS}),
              "baseline_config":read(baseline_config), "cases":cases, "requests":requests,
              "policy":"Fresh conversation per request; no browsing, code execution, files, tools, retries or hints; retain first raw reply. Product hidden instructions/controls may differ."}
    packet["id"] = digest(packet)
    write(output/"packet.json", packet)
    write(output/"responses.template.json", [{"request_id":r["request_id"], "raw_text":""} for r in requests])
    for r in requests:
        content = "SYSTEM INSTRUCTION\n"+r["messages"][0]["content"]+"\n\n"+r["messages"][1]["content"]
        (output/(r["request_id"]+".txt")).write_text(content, encoding="utf-8")
    (output/"README.md").write_text("Use packet.json messages where a client supports system/user roles. In a chat GUI, paste one .txt in a NEW conversation; the role wrapper differs from API and is disclosed. Copy the first answer verbatim into raw_text in a copy of responses.template.json. Do not strip Markdown or repair JSON. Missing replies remain missing. Never attach the dataset/evaluator or use this project discussion as a test chat.\n", encoding="utf-8")
    return output


def import_responses(dataset, packet_path, responses_path, metadata_path, output, repetition=0):
    dataset = Path(dataset)
    manifest = verify_dataset(dataset)
    packet, replies, endpoint = read(packet_path), read(responses_path), read(metadata_path)
    if digest({k:v for k,v in packet.items() if k!="id"}) != packet["id"] or packet["dataset_id"] != manifest["id"]:
        raise ValueError("Packet identity/dataset mismatch")
    if packet["code_hash"] != code_hash(exclude_scoring=True):
        raise ValueError("Use original export code; packet code changed")
    if repetition < 0 or endpoint.get("provider_scope") != "subscription" or endpoint.get("mode") != "direct" or not endpoint.get("model") or not endpoint.get("label"):
        raise ValueError("Need nonnegative repetition and subscription/direct metadata with explicit model and label")
    if any(k.lower() in ("api_key", "token", "password", "authorization") for k in endpoint):
        raise ValueError("No credentials in exchange metadata")
    allowed_cases = {c["id"]:c for c in read(dataset/"cases.json")}
    if any(c!=allowed_cases.get(c["id"]) for c in packet["cases"]):
        raise ValueError("Foreign case or unsupported final test")
    if any(c["split"]=="test" for c in packet["cases"]):
        freeze = frozen_test(dataset, manifest, endpoint)
        if freeze["baseline_config"]!=packet["baseline_config"]:
            raise ValueError("Packet baseline configuration differs from freeze")
    requests = {r["request_id"]:r for r in packet["requests"]}
    if len(requests)!=len(packet["requests"]) or not isinstance(replies, list):
        raise ValueError("Invalid request/reply list")
    expected_pairs = {(c["id"],h) for c in packet["cases"] for h in c.get("decision_hours", [None])}
    if {(r["case_id"],r["hour"]) for r in requests.values()}!=expected_pairs or len(requests)!=len(expected_pairs):
        raise ValueError("Packet request coverage mismatch")
    observations = {}
    for r in requests.values():
        obs = visible(read(dataset/"observations"/(r["case_id"]+".json")), r["hour"])
        ph = digest({"system":SYSTEM, "user":prompt(obs)})
        if r["observation_hash"]!=digest(obs) or r["prompt_hash"]!=ph or r["request_id"]!=digest([manifest["id"],r["case_id"],r["hour"],ph])[:24] or r["messages"]!=[{"role":"system", "content":SYSTEM},{"role":"user", "content":prompt(obs)}]:
            raise ValueError("Packet observations/prompts changed")
        observations[r["request_id"]] = obs
    seen = set()
    for reply in replies:
        rid = reply.get("request_id")
        if rid not in requests or rid in seen or not isinstance(reply.get("raw_text"), str):
            raise ValueError("Duplicate/foreign reply or nontext raw answer")
        seen.add(rid)
    env = environment(); env.pop("code_hash")
    identity = {"schema":1, "dataset_id":manifest["id"], "code_hash":code_hash(exclude_scoring=True), "environment":env,
                "methods":["llm"], "cases":packet["cases"], "baseline_config":packet["baseline_config"],
                "endpoint":endpoint, "repetition":repetition, "prompt_version":packet["prompt_version"],
                "exchange_packet_id":packet["id"], "raw_replies_hash":digest(replies)}
    runid = digest(identity); root = Path(output)/runid[:20]
    if root.exists(): raise ValueError("Import immutable; output already exists")
    write(root/"manifest.json", {"id":runid, "identity":identity, "purpose":manifest["config"]["stage"],
          "retry_policy":"first raw answer only, operator declared; no retries by importer", "timing_note":"unknown; no API latency measured",
          "comparison_track":"supplementary subscription product; hidden system prompts, tokenizer, effective generation and tools cannot be independently verified"})
    cases = {c["id"]:c for c in packet["cases"]}
    for reply in replies:
        rid = reply["request_id"]; req = requests[rid]
        recordid = digest([runid, req["case_id"], "llm", req["hour"]])[:24]
        record = {"id":recordid, "run_id":runid, "case":cases[req["case_id"]], "method":"llm",
                  "method_label":endpoint["label"]+":"+endpoint["mode"], "hour":req["hour"], "status":"invalid",
                  "prediction":None, "prompt_hash":req["prompt_hash"], "observation_hash":req["observation_hash"],
                  "elapsed_s":None, "retries":None, "timing_source":"not measured"}
        write(root/"traces"/(recordid+".json"), {"messages":req["messages"], "raw_reply":reply, "metadata":endpoint})
        try:
            record.update(prediction=validate(json.loads(reply["raw_text"]), observations[rid]), status="ok")
        except (ValueError, KeyError, TypeError) as exc:
            record["error"] = type(exc).__name__+": "+str(exc)[:300]
        write(root/"records"/(recordid+".json"), record)
    return root
