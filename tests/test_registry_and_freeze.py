"""Regression: exact prompt register, immutable freeze and subscription test boundaries."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from waterbench.catalog import catalog
from waterbench.exchange import export_cases, import_responses
from waterbench.runner import run
from waterbench.storage import code_hash, digest, read, write, file_hash


class RegistryChecks(unittest.TestCase):
    def fixture(self,root):
        data=root/"data"; cases=[]
        obs={"task":"cluster","consumer_ids":["a","b","c"],"profiles":{"a":[0.0]*24,"b":[1.0]*24,"c":[2.0]*24},"k":3}
        for split in ("validation","test"):
            case={"id":split,"unit":split,"network":"grid","task":"cluster","split":split,"condition":"synthetic"}
            cases.append(case);write(data/"observations"/(split+".json"),obs)
            write(data/"evaluator"/(split+".json"),{"assignments":{"a":0,"b":1,"c":2}})
        write(data/"cases.json",cases)
        m={"config":{"stage":"core-candidate","networks":["grid"]},"files":{p.relative_to(data).as_posix():file_hash(p) for p in data.rglob("*.json")}}
        m["id"]=digest(m);write(data/"manifest.json",m)
        baseline={"training_seed":1};write(root/"baseline.json",baseline)
        meta={"provider_scope":"subscription","mode":"direct","label":"fixture-cloud","model":"fake","platform":"fixture",
              "reasoning_mode":"disabled","operator_declaration":"fixture","comparison_track":"supplementary-product-level"}
        write(root/"metadata.json",meta)
        return data,m,baseline,meta

    def test_frozen_cloud_test_and_exact_register(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); data,m,baseline,meta=self.fixture(root)
            with self.assertRaises(ValueError): export_cases(data,root/"no-freeze",root/"baseline.json",["test"],["cluster"],["grid"])
            write(data/"freeze.json",{"dataset_id":m["id"],"code_hash":code_hash(exclude_scoring=True),"baseline_config":baseline,
                  "baseline_seeds":[1,2],"endpoints":[meta]})
            packet=export_cases(data,root/"packet",root/"baseline.json",["test"],["cluster"],["grid"])
            req=read(packet/"packet.json")["requests"][0]
            write(root/"responses.json",[{"request_id":req["request_id"],"raw_text":'{"assignments":{"a":0,"b":1,"c":2}}'}])
            cloud=import_responses(data,packet/"packet.json",root/"responses.json",root/"metadata.json",root/"runs")
            cat=catalog(data,[cloud],root/"catalog")
            actual=json.loads((cat/"inputs_and_prompts.jsonl").read_text())
            self.assertEqual(actual["messages"],req["messages"])
            self.assertEqual(actual["prompt_hash"],req["prompt_hash"])
            changed={**meta,"reasoning_mode":"enabled"};write(root/"metadata.json",changed)
            with self.assertRaises(ValueError): import_responses(data,packet/"packet.json",root/"responses.json",root/"metadata.json",root/"other-runs")
            record=next((cloud/"records").glob("*.json"));value=read(record);value["prompt_hash"]="changed";write(record,value)
            with self.assertRaises(ValueError): catalog(data,[cloud],root/"bad-catalog")

    def test_baseline_seed_freeze_and_rejection_before_inference(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data,m,baseline,meta=self.fixture(root)
            write(data/"freeze.json",{"dataset_id":m["id"],"code_hash":code_hash(exclude_scoring=True),"baseline_config":baseline,
                  "baseline_seeds":[1,2],"generation_seeds":[910,911,912],"endpoints":[]})
            write(root/"baseline.json",{"training_seed":2})
            result=run(data,root/"runs",["shape-dtw-average"],["test"],["cluster"],["grid"],root/"baseline.json")
            self.assertEqual(read(next((result/"records").glob("*.json")))["status"],"ok")
            write(root/"baseline.json",{"training_seed":3})
            with self.assertRaises(ValueError): run(data,root/"bad",["shape-dtw-average"],["test"],["cluster"],["grid"],root/"baseline.json")


if __name__=="__main__":unittest.main()
