"""Software fixtures only. These are never experimental results."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from waterbench import methods
from waterbench.runner import infer, run
from waterbench.scoring import case_score, metrics, bootstrap
from waterbench.storage import code_hash, digest, file_hash, read, verify_dataset, write


class BenchmarkChecks(unittest.TestCase):
    def setUp(self):
        self.obs = {"task": "leak", "hours": list(range(24)), "sensor_ids": ["J"],
                    "pressure_m": [[30.0]]*24, "nominal_pressure_m": [[30.0]]*24,
                    "zones": {"Z00": ["J"]}, "nominal_signatures_m": {"Z00": [1.0]}}

    def test_causal_prefix_and_hidden_keys(self):
        observed = methods.visible(self.obs, 8)
        changed = copy.deepcopy(self.obs)
        changed["pressure_m"][20] = [999.0]
        self.assertEqual(observed, methods.visible(changed, 8))
        self.assertEqual(len(observed["pressure_m"]), 9)
        self.assertNotIn("seed", methods.prompt(observed))
        self.assertNotIn("future", methods.prompt(observed))

    def test_scorer_version_does_not_change_inference_identity(self):
        from waterbench.storage import file_hash as original_hash
        before = code_hash(exclude_scoring=True)
        with patch("waterbench.storage.file_hash", side_effect=lambda p: "changed" if Path(p).name == "scoring.py" else original_hash(p)):
            self.assertEqual(code_hash(exclude_scoring=True), before)
            self.assertNotEqual(code_hash(), before)

    def test_strict_schema(self):
        methods.validate({"alarm": True, "score": 0.9, "ranking": ["Z00"]}, self.obs)
        for result in ({"alarm": 1, "score": 0.5, "ranking": []},
                       {"alarm": True, "score": float("nan"), "ranking": ["Z00"]},
                       {"alarm": True, "score": 0.8, "ranking": ["unknown"]},
                       {"alarm": False, "score": 0.1, "ranking": ["Z00"]}):
            with self.assertRaises(ValueError): methods.validate(result, self.obs)
        with self.assertRaises(ValueError):
            methods.validate({"forecast_L_s": [True]*24}, {"task": "forecast"})

    def test_known_forecast_and_degenerate_mase(self):
        case = {"id": "f", "unit": "u", "network": "grid", "task": "forecast", "split": "test", "condition": "c"}
        records = [{"hour": None, "status": "ok", "prediction": {"forecast_L_s": [2.0]*24}}]
        result = case_score(case, records, {"future_L_s": [1.0]*24}, 2)
        self.assertEqual((result["mae"], result["rmse"], result["mase"]), (1.0, 1.0, 0.5))
        self.assertIsNone(case_score(case, records, {"future_L_s": [1.0]*24}, 0)["mase"])
        self.assertEqual(case_score(case, [], {"future_L_s": [1.0]*24}, 2)["success"], 0)

    def test_event_not_windows_and_failures(self):
        case = {"id": "l", "unit": "u", "network": "grid", "task": "leak", "split": "test", "condition": "c", "decision_hours": [8,12,16,20,23]}
        truth = {"has_leak": True, "zone": "Z00", "start_hour": 10, "end_hour": 24}
        records = [{"hour": h, "status": "ok", "prediction": {"alarm": h>=12, "score": 0.9, "ranking": ["Z00"] if h>=12 else []}} for h in case["decision_hours"]]
        result = case_score(case, records, truth)
        self.assertEqual((result["tp"], result["fn"], result["delay_h"], result["localized3"]), (1,0,2,1))
        self.assertEqual(metrics([result])["f1"], 1)
        records[0]["status"] = "invalid"
        failed = case_score(case, records, truth)
        combined = metrics([result, failed])
        self.assertEqual(combined["validity"], 0.5)
        self.assertEqual(combined["overall_detection_localization_top1"], 0.5)
        self.assertEqual(bootstrap([result], "f1", 100, np.random.default_rng(1)), [1,1])

    def test_tool_allowlist_and_original_response(self):
        endpoint = {"mode": "tools", "max_tool_calls": 1}
        response = {"choices": [{"message": {"role": "assistant", "content": None,
                    "tool_calls": [{"id": "x", "type": "function", "function": {"name": "read_labels", "arguments": "{}"}}]}}]}
        with tempfile.TemporaryDirectory() as d, patch("waterbench.runner.request", return_value=response):
            with self.assertRaises(ValueError): infer(endpoint, self.obs, Path(d)/"trace.json")
            self.assertEqual(read(Path(d)/"trace.json")["responses"], [response])

    def test_tools_round_trip(self):
        endpoint = {"mode": "tools", "max_tool_calls": 1}
        first = {"choices": [{"message": {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "x", "type": "function", "function": {"name": "observation_summary", "arguments": "{}"}}]}}]}
        last = {"choices": [{"message": {"role": "assistant", "content": '{"alarm":false,"score":0.1,"ranking":[]}'}}]}
        with tempfile.TemporaryDirectory() as d, patch("waterbench.runner.request", side_effect=[first,last]):
            result, trace = infer(endpoint, self.obs, Path(d)/"trace.json")
            self.assertFalse(result["alarm"])
            self.assertEqual(len(trace["tools"]), 1)

    def test_archive_resume_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d); dataset = base/"data"; dataset.mkdir()
            cases = [{"id": x, "unit": x, "network": "grid", "task": "forecast", "split": "validation", "condition": "c"} for x in ("a","b")]
            write(dataset/"cases.json", cases)
            for c in cases:
                write(dataset/"observations"/(c["id"]+".json"), {"task": "forecast", "history_L_s": [1.0]*168, "horizon": 24})
            manifest = {"config": {"stage": "smoke"}, "files": {p.relative_to(dataset).as_posix():file_hash(p) for p in dataset.rglob("*.json")}}
            manifest["id"] = digest(manifest); write(dataset/"manifest.json", manifest)
            write(base/"baseline.json", {"training_seed": 1})
            args = (dataset, base/"runs", ["seasonal-naive"], ["validation"], ["forecast"], ["grid"], base/"baseline.json")
            with patch("waterbench.runner.fit", return_value={"name": "seasonal-naive", "scale": 1}):
                root = run(*args, stop_after=1)
                first = next((root/"records").glob("*.json")); before = first.read_bytes()
                self.assertEqual(run(*args), root)
                self.assertEqual(first.read_bytes(), before)
                self.assertEqual(len(list((root/"records").glob("*.json"))), 2)
                record = read(first); record["status"] = "started"; write(first, record)
                run(*args)
                self.assertEqual(read(first)["status"], "interrupted")
            (dataset/"cases.json").write_text("[]")
            with self.assertRaises(ValueError): verify_dataset(dataset)


if __name__ == "__main__":
    unittest.main()
