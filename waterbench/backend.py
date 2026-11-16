"""Read-only loaded-model discovery and provenance. Never loads or changes models."""
import datetime as dt
import json
import os
import subprocess
import urllib.request
from pathlib import Path

from .storage import digest, read, write


class IdentityError(RuntimeError):
    """Stop the run when its served model cannot be verified."""


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def redact(value):
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()
                if not any(x in k.lower() for x in ("api_key", "apikey", "password", "secret", "authorization", "access_token"))}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def inventory(endpoint):
    headers = {}
    key = os.environ.get(endpoint.get("api_key_env", "WATERBENCH_API_KEY"))
    if not key:
        raise IdentityError("Set the named API-key environment variable")
    headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(endpoint["base_url"].rstrip("/") + "/models", headers=headers)
    with urllib.request.urlopen(req, timeout=endpoint["timeout_s"]) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise IdentityError("Model inventory exceeds 2 MB")
    result = redact(json.loads(raw))
    if not isinstance(result, dict) or not isinstance(result.get("data"), list):
        raise IdentityError("Expected /models data array")
    return result


# Only fields with model/configuration meaning; exclude usage, uptime and timestamps.
STABLE_FIELDS = {"id", "name", "display_name", "model_name", "family", "architecture",
                 "parameter_count", "parameters", "quantization", "quant", "dtype", "precision",
                 "revision", "checkpoint", "model_path", "file_path", "filename", "sha256",
                 "backend", "backend_version", "context_length", "max_model_len", "max_context_length",
                 "context_window", "native_context_length", "chat_template", "tokenizer_revision",
                 "reasoning", "reasoning_effort", "thinking", "tool_calling", "settings", "config"}


def select(payload, model=None):
    loaded = [m for m in payload["data"] if isinstance(m, dict) and m.get("loaded") is True]
    candidates = [m for m in loaded if model is None or m.get("id") == model]
    if len(candidates) != 1 or not candidates[0].get("id"):
        raise IdentityError("Select exactly one loaded:true model with --model; no auto-load/fallback")
    return candidates[0]


def fingerprint(model):
    return {k: v for k, v in model.items() if k in STABLE_FIELDS}


def guard(endpoint):
    expected = endpoint.get("loaded_guard")
    if not expected:
        return None  # Legacy generic endpoints retain their previous behavior.
    try:
        payload = inventory(endpoint)
        selected = select(payload, endpoint["model"])
    except (OSError, ValueError, KeyError) as exc:
        raise IdentityError("Cannot verify loaded model inventory; stop without inference/fallback") from exc
    actual = fingerprint(selected)
    if actual != expected["fingerprint"]:
        raise IdentityError("Loaded model/configuration differs from discovery snapshot; stop and investigate")
    return {"checked_utc": utc(), "fingerprint": actual, "fingerprint_hash": digest(actual)}


def response_identity(endpoint, response):
    if endpoint.get("loaded_guard"):
        allowed = endpoint.get("response_model_ids", [endpoint["model"]])
        if response.get("model") not in allowed:
            raise IdentityError("API response model absent or differs from explicitly allowed IDs")


def discover(endpoint, output, label, model=None, metadata=None):
    output = Path(output)
    snapshot_path = Path(str(output) + ".snapshot.json")
    if output.exists() or snapshot_path.exists():
        raise ValueError("Discovery files already exist; reuse for resume or choose a new path")
    payload = inventory(endpoint)
    selected = select(payload, model)
    declared = read(metadata) if metadata else {}
    if not isinstance(declared, dict):
        raise ValueError("Metadata must be an object")
    forbidden = {"base_url", "model", "label", "loaded_guard", "generation", "mode", "api_key_env", "response_model_ids"}
    if forbidden & declared.keys() or redact(declared) != declared:
        raise ValueError("Metadata cannot override transport/generation/identity or contain credentials")
    value = dict(endpoint)
    value.update(declared)
    value.update(label=label, model=selected["id"], checkpoint=declared.get("checkpoint", selected["id"]),
                 loaded_guard={"fingerprint": fingerprint(selected)}, response_model_ids=endpoint.get("response_model_ids", [selected["id"]]))
    value["model_metadata"] = {"api_observed": fingerprint(selected), "operator_declared": declared,
                               "effective_sampling": "requested generation saved; server application not verified",
                               "limitations": "Only exposed inventory fields checked; unexposed changes and change-back races cannot be detected"}
    for field in ("revision", "quantization", "backend", "backend_version", "chat_template", "tokenizer_revision"):
        observed = selected.get("quant", selected.get("quantization", "UNKNOWN")) if field == "quantization" else selected.get(field, "UNKNOWN")
        manual = declared.get(field)
        if field == "quantization" and observed != "UNKNOWN" and manual not in (None, "UNKNOWN", observed):
            raise ValueError("Declared quantization disagrees with /models quant")
        value[field] = observed if observed not in (None, "UNKNOWN") else (manual or "UNKNOWN")
    value["display_name"] = declared.get("display_name") or selected.get("display_name") or label
    value["configured_context_tokens"] = selected.get("context_length")
    value["max_context_tokens"] = selected.get("max_context_length")
    value["native_context_tokens"] = selected.get("native_context_length", declared.get("native_context_tokens"))
    if type(value["configured_context_tokens"]) is int and value["configured_context_tokens"] > 0:
        value["context_limit_tokens"] = min(value["context_limit_tokens"], value["configured_context_tokens"])
    snapshot = {"captured_utc": utc(), "inventory": payload, "selected_model": selected,
                "endpoint_config_hash": digest(value), "source": "authenticated GET /v1/models; no inference"}
    write(snapshot_path, snapshot)
    write(output, value)
    return output


def provenance(session_id=None, session_name=None):
    root = Path(__file__).resolve().parent.parent
    git = {"available": False, "commit": None, "dirty": None}
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10)
        if head.returncode == 0:
            status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=10)
            git = {"available": True, "commit": head.stdout.strip(),
                   "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None}
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"started_utc": utc(), "session_id": session_id, "session_name": session_name,
            "git": git, "pid": os.getpid(), "note": "Session IDs supplied by operator"}
