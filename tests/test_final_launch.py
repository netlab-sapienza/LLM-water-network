import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.freeze_final import freeze
from scripts.run_pilot3 import main as campaign
from waterbench.methods import TASKS
from waterbench.storage import code_hash,digest,file_hash,read,write


class FinalLaunchChecks(unittest.TestCase):
    def fixture(self,root,status="invalid"):
        data = root/"data"; cases = []
        for split in ("validation","test"):
            for task in TASKS:
                cases.append({"id":split+task,"unit":split,"network":"grid","task":task,"split":split,"condition":"nominal"})
        write(data/"cases.json",cases)
        m = {"config":{"stage":"core-candidate","networks":["grid"]},"files":{"cases.json":file_hash(data/"cases.json")}}
        m["id"] = digest(m); write(data/"manifest.json",m)
        ep = {"base_url":"http://127.0.0.1:1/v1","model":"fixture","label":"fixture","provider_scope":"local",
              "mode":"direct","timeout_s":180,"max_tool_calls":1,"context_limit_tokens":100000,
              "generation":{"max_tokens":2048,"temperature":0},"checkpoint":"fixture","revision":"UNKNOWN",
              "quantization":"BF16","backend_version":"UNKNOWN","chat_template":"UNKNOWN"}
        write(root/"endpoint.json",ep); write(root/"baseline.json",{"training_seed":1})
        write(root/"declaration.json",{"operator_declaration":"fixture unknown fields not exposed",
              "unavailable_fields":dict.fromkeys(("revision","backend_version","chat_template"),"not exposed")})
        write(root/"selection.json",{"specialists":{"cluster/grid":"shape-kmeans"}})
        (root/"notes.md").write_text("Measured protocol; invalid LLM JSON retained as model outcome.")
        identity = {"fixture":1}; rid = digest(identity); runs = root/"pilot-runs"; run = runs/rid[:20]
        write(run/"manifest.json",{"id":rid,"identity":identity})
        write(run/"records/decision.json",{"method":"llm","status":status})
        sources = {p.relative_to(runs).as_posix():file_hash(p) for p in run.rglob("*.json")}
        pilot = {"id":"pilot","stage":"pilot","dataset_config":{"stage":"pilot","networks":["grid"]},
                 "inference_code_hash":code_hash(exclude_scoring=True),"coverage":[{"run":rid,"expected":1,"observed":1,"missing":0,"failed":int(status!="ok")}],
                 "source_hashes":sources,"method_configurations":{"fixture:direct":ep,"shape-kmeans":{}},
                 "aggregates":[{"method":"fixture:direct","task":task,"network":"grid","n":1,"validity":0 if task=="cluster" else 1} for task in TASKS]}
        write(root/"evaluation/evaluation.json",pilot)
        return data,runs

    def do_freeze(self,root,data,runs):
        return freeze(data,root/"baseline.json",[root/"endpoint.json"],root/"evaluation",root/"notes.md",
                      root/"declaration.json",root/"selection.json",[1,2,3],runs)

    def test_invalid_model_and_declared_unknowns_retained_but_changes_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); data,runs=self.fixture(root)
            before=code_hash(exclude_scoring=True)
            output=self.do_freeze(root,data,runs); result=read(output)
            self.assertEqual(result["pilot_status_counts"],{"invalid":1})
            self.assertEqual(result["baseline_seeds"],[1,2,3])
            self.assertIn("revision",result["unavailable_metadata"]["fixture:direct"])
            self.assertEqual(before,code_hash(exclude_scoring=True))
            with self.assertRaises(ValueError): self.do_freeze(root,data,runs)  # immutable
        for change in ("generation","missing","provenance","hash","source"):
            with tempfile.TemporaryDirectory() as t:
                root=Path(t); data,runs=self.fixture(root)
                if change=="generation":
                    e=read(root/"endpoint.json"); e["generation"]["max_tokens"]=999; write(root/"endpoint.json",e)
                elif change=="provenance": write(root/"declaration.json",{"operator_declaration":"fixture","unavailable_fields":{}})
                elif change=="source": next((runs).rglob("decision.json")).write_text("{}")
                else:
                    e=read(root/"evaluation/evaluation.json")
                    if change=="missing": e["coverage"][0]["missing"]=1
                    else: e["inference_code_hash"]="changed"
                    write(root/"evaluation/evaluation.json",e)
                with self.assertRaises(ValueError): self.do_freeze(root,data,runs)
                self.assertFalse((data/"freeze.json").exists())

    def test_infrastructure_failures_block_freeze(self):
        for status in ("identity_error","timeout","interrupted","started","http_error"):
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);data,runs=self.fixture(root,status)
                with self.assertRaises(ValueError): self.do_freeze(root,data,runs)

    def test_test_campaign_reaches_test_runner_and_final_bootstrap(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);data,_=self.fixture(root)
            args=["run_pilot3","--dataset",str(data),"--endpoint",str(root/"endpoint.json"),"--baseline-config",str(root/"baseline.json"),
                  "--output",str(root/"campaigns"),"--split","test","--execute"]
            with patch("sys.argv",args), patch("scripts.run_pilot3.run") as runner:
                with self.assertRaises(SystemExit): campaign()
                runner.assert_not_called()
            write(data/"freeze.json",{"fixture":True})
            with patch("sys.argv",args),patch("scripts.run_pilot3.run",return_value=root/"run") as runner,\
                 patch("scripts.run_pilot3.evaluate",return_value=root/"eval") as evaluate,\
                 patch("scripts.run_pilot3.report"),patch("scripts.run_pilot3.catalog"):
                campaign()
                self.assertEqual(runner.call_count,6)
                self.assertTrue(all(c.args[3]==["test"] for c in runner.call_args_list))
                self.assertEqual(evaluate.call_args.kwargs["bootstrap_count"],10000)
            # Default remains validation and planning causes no inference.
            args=[a for a in args if a!="--execute"]
            args[args.index("test")]="validation"
            with patch("sys.argv",args),patch("scripts.run_pilot3.run") as runner:
                campaign();runner.assert_not_called()


if __name__=="__main__":unittest.main()
