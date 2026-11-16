"""Contract and end-to-end checks. Fake answers are never scientific results."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from waterbench.clustering import observation, augment, predict, scores
from waterbench.methods import validate, prompt, SYSTEM
from waterbench.exchange import export_cases, import_responses
from waterbench.runner import run
from waterbench.scoring import evaluate, bootstrap, case_score
from waterbench.storage import digest, file_hash, write, read, verify_dataset


class ThreeTaskChecks(unittest.TestCase):
    def fixture(self, root):
        data = root/"parent"
        cases = [{"id":s, "unit":s, "network":"grid", "task":"forecast", "split":s, "condition":"nominal"} for s in ("train","validation","test")]
        write(data/"cases.json", cases)
        history = (15+3*np.sin(np.arange(168)*2*np.pi/24)).tolist()
        for c in cases:
            write(data/"observations"/(c["id"]+".json"), {"task":"forecast", "history_L_s":history})
            write(data/"evaluator"/(c["id"]+".json"), {"future_L_s":history[-24:]})
            if c["split"]=="train": write(data/"training"/(c["id"]+".json"), {"series_L_s":history*2})
        (data/"source").mkdir()
        m = {"config":{"stage":"pilot", "networks":["grid"]}, "files":{p.relative_to(data).as_posix():file_hash(p) for p in data.rglob("*") if p.is_file()}}
        m["id"] = digest(m); write(data/"manifest.json", m)
        write(root/"baseline.json", {"training_seed":17})
        return data

    def test_shapes_seed_schema_and_permutation(self):
        history = np.full(168, 20.)
        a,t = observation(history,17); b,_ = observation(history,17); c,_ = observation(history,18)
        self.assertEqual(a,b); self.assertNotEqual(a,c)
        self.assertNotIn("assignments", a); self.assertNotIn("seed", prompt(a))
        for name in ("shape-kmeans", "shape-dtw-average"):
            result = validate(predict(name,a,17),a)
            self.assertTrue(-1<=scores(result,t,a["consumer_ids"])["ari"]<=1)
        permuted = {"assignments":{i:(v+1)%3 for i,v in t["assignments"].items()}}
        self.assertEqual(scores(permuted,t,a["consumer_ids"])["ari"],1)
        for value in ({"assignments":{}}, {"assignments":{i:True for i in a["consumer_ids"]}}, {"assignments":{i:0 for i in a["consumer_ids"]}}):
            with self.assertRaises(ValueError): validate(value,a)

    def test_parent_immutable_and_exchange_scoring(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); parent=self.fixture(root); before=verify_dataset(parent)
            data=augment(parent,root/"three",23)
            m=verify_dataset(data)
            self.assertEqual(m["parent_dataset_id"],before["id"])
            self.assertEqual(verify_dataset(parent),before)
            for name,h in before["files"].items():
                if name!="cases.json": self.assertEqual(file_hash(data/name),h)
            with self.assertRaises(ValueError): augment(parent,parent/"nested")
            packet=export_cases(data,root/"packet",root/"baseline.json",["validation"],["cluster"],["grid"])
            p=read(packet/"packet.json"); r=p["requests"][0]
            self.assertNotIn("evaluator",json.dumps(p)); self.assertNotIn("synthetic latent archetype",json.dumps(r["messages"]))
            cid=r["case_id"]; truth=read(data/"evaluator"/(cid+".json"))
            replies=[{"request_id":r["request_id"], "raw_text":json.dumps({"assignments":truth["assignments"]})}]
            write(root/"replies.json",replies)
            write(root/"metadata.json",{"provider_scope":"subscription", "mode":"direct", "label":"fixture", "model":"fake"})
            external=import_responses(data,packet/"packet.json",root/"replies.json",root/"metadata.json",root/"runs")
            baseline=run(data,root/"runs",["shape-kmeans","shape-dtw-average"],["validation"],["cluster"],["grid"],root/"baseline.json")
            record=read(next((external/"records").glob("*.json")))
            self.assertEqual(record["prompt_hash"],r["prompt_hash"])
            self.assertIsNone(record["elapsed_s"])
            ev=evaluate(data,[external,baseline],root/"eval",100)
            result=read(ev/"evaluation.json")
            self.assertTrue(all(c["missing"]==c["failed"]==0 for c in result["coverage"]))
            self.assertEqual(next(a for a in result["aggregates"] if a["method"]=="fixture:direct")["ari"],1)
            write(root/"replies.json",replies*2)
            with self.assertRaises(ValueError): import_responses(data,packet/"packet.json",root/"replies.json",root/"metadata.json",root/"runs")
            with self.assertRaises(ValueError): export_cases(data,root/"test-packet",root/"baseline.json",["test"],["cluster"],["grid"])

    def test_invalid_cloud_answer_retained_and_missing_counted(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); data=augment(self.fixture(root),root/"three",23)
            packet=export_cases(data,root/"packet",root/"baseline.json",["validation"],["forecast","cluster"],["grid"])
            requests=read(packet/"packet.json")["requests"]
            write(root/"replies.json",[{"request_id":requests[0]["request_id"],"raw_text":"```json\n{}\n```"}])
            write(root/"metadata.json",{"provider_scope":"subscription", "mode":"direct", "label":"invalid-fixture", "model":"fake"})
            run_dir=import_responses(data,packet/"packet.json",root/"replies.json",root/"metadata.json",root/"runs")
            ev=evaluate(data,[run_dir],root/"eval",100)
            coverage=read(ev/"evaluation.json")["coverage"][0]
            self.assertEqual((coverage["expected"],coverage["observed"],coverage["missing"],coverage["failed"]),(2,1,1,1))

    def test_replicas_remain_one_bootstrap_unit(self):
        case={"id":"c","unit":"u","network":"grid","task":"cluster","split":"validation","condition":"synthetic"}
        truth={"assignments":{"a":0,"b":1,"c":2}}
        row=case_score(case,[{"hour":None,"status":"ok","prediction":truth,"elapsed_s":1}],truth)
        self.assertEqual(bootstrap([row]*3,"ari",100,np.random.default_rng(1)),[1,1])

    def test_generation_seed_forwarding_identity_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); data=augment(self.fixture(root),root/"three",23)
            ep={"provider_scope":"local", "base_url":"http://127.0.0.1:62888/v1", "mode":"direct", "timeout_s":10,
                "max_tool_calls":1, "context_limit_tokens":262144, "label":"fake-seeded", "model":"fake",
                "generation":{"temperature":0.2,"max_tokens":2048}, "seed_supported":True}
            write(root/"endpoint.json",ep)
            cid=next(c["id"] for c in read(data/"cases.json") if c["task"]=="cluster" and c["split"]=="validation")
            obs=read(data/"observations"/(cid+".json")); answer=predict("shape-kmeans",obs,17)
            response={"model":"fake","choices":[{"finish_reason":"stop","message":{"content":json.dumps(answer)}}]}
            args=(data,root/"runs",["llm"],["validation"],["cluster"],["grid"],root/"baseline.json",root/"endpoint.json")
            with patch("waterbench.runner.request",return_value=response) as call:
                a=run(*args,generation_seed=17)
                self.assertEqual(call.call_args.args[0]["generation"]["seed"],17)
                self.assertEqual(run(*args,generation_seed=17),a)
                self.assertEqual(call.call_count,1)
                b=run(*args,generation_seed=18)
                self.assertNotEqual(a,b)
                self.assertEqual(call.call_args.args[0]["generation"]["seed"],18)
            self.assertEqual(read(root/"endpoint.json"),ep)
            trace=read(next((a/"traces").glob("*.json")))
            self.assertEqual(trace["effective_generation"]["seed"],17)
            ep["seed_supported"]=False; write(root/"endpoint.json",ep)
            with self.assertRaises(ValueError): run(*args,generation_seed=19)


if __name__=="__main__": unittest.main()
