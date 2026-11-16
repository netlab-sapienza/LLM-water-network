"""Local HTTP fixtures only: not models or experimental predictions."""
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from waterbench.backend import IdentityError, discover, guard, redact, select
from waterbench.runner import endpoint_config, infer, run
from waterbench.storage import digest, file_hash, read, write


class BackendChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = {"id": "fixture", "loaded": True, "quantization": "BF16", "context_length": 32768}
        self.returned = "fixture"
        self.switch_after_post = False
        self.posts = 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, value):
                self.send_response(200); self.end_headers()
                self.wfile.write(json.dumps(value).encode())

            def do_GET(self):
                if self.headers.get("Authorization") != "Bearer fixture-key":
                    self.send_response(401); self.end_headers(); return
                self.reply({"data": [owner.model], "api_key": "must-not-be-saved"})

            def do_POST(self):
                owner.posts += 1
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if payload["model"] != "fixture" or "enable_tools" in payload:
                    self.send_response(400); self.end_headers(); return
                if owner.switch_after_post:
                    owner.model["quantization"] = "Q4"
                self.reply({"model": owner.returned, "usage": {"total_tokens": 4}, "choices": [
                    {"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps({"forecast_L_s": [1]*24})}}]})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.auth = patch.dict(os.environ, WATERBENCH_API_KEY="fixture-key")
        self.auth.start()
        self.endpoint = {"label": "template", "provider_scope": "local",
                         "base_url": f"http://127.0.0.1:{self.server.server_port}/v1", "model": "fixture",
                         "mode": "direct", "timeout_s": 2, "max_tool_calls": 1, "context_limit_tokens": 32768,
                         "generation": {"temperature": 0, "max_tokens": 100}}
        self.obs = {"task": "forecast", "history_L_s": [1]*168, "horizon": 24}

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()
        self.auth.stop(); self.temp.cleanup()

    def config(self):
        path = self.root / "endpoint.json"
        discover(self.endpoint, path, "fixture-bf16")
        return path, endpoint_config(path)

    def test_discovery_read_only_immutable_and_redacted(self):
        path, endpoint = self.config()
        self.assertEqual(self.posts, 0)
        self.assertEqual(endpoint["quantization"], "BF16")
        self.assertNotIn("api_key", read(str(path)+".snapshot.json")["inventory"])
        with self.assertRaises(ValueError): discover(self.endpoint, path, "again")
        self.model["uptime"] = 999
        self.assertIsNotNone(guard(endpoint))
        self.model["quantization"] = "Q4"
        with self.assertRaises(IdentityError): guard(endpoint)

    def test_no_loaded_or_ambiguous_inventory(self):
        for models in ([], [{"id": "f", "loaded": False}], [self.model, {"id": "other", "loaded": True}]):
            with self.assertRaises(IdentityError): select({"data": models})
        self.assertEqual(select({"data": [self.model, {"id": "other", "loaded": True}]}, "fixture")["id"], "fixture")

    def test_actual_unsloth_field_mapping_and_optional_fields(self):
        self.model = {"id": "fixture", "object": "model", "created": 1791021539,
                      "owned_by": "unsloth-studio", "quant": "BF16", "loaded": True,
                      "context_length": 262144, "max_context_length": 262144, "native_context_length": 262144}
        _, endpoint = self.config()
        self.assertEqual(endpoint["quantization"], "BF16")
        self.assertEqual(endpoint["configured_context_tokens"], 262144)
        self.assertEqual(endpoint["native_context_tokens"], 262144)
        self.assertEqual(endpoint["context_limit_tokens"], 32768)
        self.model["created"] += 100
        self.assertIsNotNone(guard(endpoint))
        self.model["context_length"] = 8192
        with self.assertRaises(IdentityError): guard(endpoint)

    def test_unknown_metadata_does_not_hide_api_quant_and_alias_is_explicit(self):
        write(self.root/"metadata.json", {"quantization": "UNKNOWN", "display_name": None})
        path = self.root/"declared.json"
        discover({**self.endpoint, "response_model_ids": ["explicit-alias"]}, path, "fixture",
                 metadata=self.root/"metadata.json")
        endpoint = endpoint_config(path)
        self.assertEqual(endpoint["quantization"], "BF16")
        self.returned = "explicit-alias"
        self.assertEqual(infer(endpoint, self.obs, self.root/"alias.json")[0]["forecast_L_s"], [1]*24)
        write(self.root/"metadata.json", {"quantization": "Q4"})
        with self.assertRaises(ValueError):
            discover(self.endpoint, self.root/"contradiction.json", "fixture", metadata=self.root/"metadata.json")

    def test_http_inference_and_response_identity(self):
        _, endpoint = self.config()
        result, trace = infer(endpoint, self.obs, self.root/"trace.json")
        self.assertEqual(result["forecast_L_s"], [1]*24)
        self.assertEqual(trace["response_models"], ["fixture"])
        self.returned = "unexpected"
        with self.assertRaises(IdentityError): infer(endpoint, self.obs, self.root/"bad.json")
        self.assertEqual(read(self.root/"bad.json")["responses"][0]["model"], "unexpected")

    def test_reserved_generation_and_redaction(self):
        write(self.root/"unsafe.json", {**self.endpoint, "generation": {"max_tokens": 10, "model": "other"}})
        with self.assertRaises(ValueError): endpoint_config(self.root/"unsafe.json")
        self.assertEqual(redact({"settings": {"password": "x", "temperature": 0}}), {"settings": {"temperature": 0}})

    def dataset(self):
        dataset = self.root/"data"
        cases = [{"id": c, "unit": c, "network": "grid", "task": "forecast", "split": "validation", "condition": "fixture"} for c in ("a", "b")]
        write(dataset/"cases.json", cases)
        for case in cases: write(dataset/"observations"/(case["id"]+".json"), self.obs)
        m = {"config": {"stage": "smoke"}, "files": {p.relative_to(dataset).as_posix(): file_hash(p) for p in dataset.rglob("*.json")}}
        m["id"] = digest(m); write(dataset/"manifest.json", m)
        write(self.root/"baseline.json", {"training_seed": 1})
        return dataset

    def test_session_resume_preserves_identity(self):
        path, _ = self.config()
        args = (self.dataset(), self.root/"runs", ["llm"], ["validation"], ["forecast"], ["grid"], self.root/"baseline.json", path)
        root = run(*args, stop_after=1, session_id="first")
        record = next((root/"records").glob("*.json")); before = record.read_bytes()
        self.assertEqual(run(*args, session_id="second"), root)
        self.assertEqual(record.read_bytes(), before)
        self.assertEqual(self.posts, 2)
        sessions = {read(p)["session_id"] for p in (root/"executions").glob("*.json")}
        self.assertEqual(sessions, {"first", "second"})

    def test_mid_request_switch_stops_entire_run(self):
        path, _ = self.config()
        self.switch_after_post = True
        with self.assertRaises(IdentityError):
            run(self.dataset(), self.root/"runs", ["llm"], ["validation"], ["forecast"], ["grid"], self.root/"baseline.json", path)
        records = list((self.root/"runs").glob("*/records/*.json"))
        self.assertEqual(len(records), 1)
        self.assertEqual(read(records[0])["status"], "identity_error")
        self.assertEqual(self.posts, 1)
        self.assertFalse(list((self.root/"runs").glob("*/writer.lock")))


if __name__ == "__main__":
    unittest.main()
