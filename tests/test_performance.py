import tempfile
import unittest
from pathlib import Path
import numpy as np
from waterbench.clustering import msm, medoids, observation, predict
from waterbench.methods import validate
from waterbench.performance import decision_cost
from waterbench.scoring import case_score
from waterbench.storage import write


class PerformanceChecks(unittest.TestCase):
    def test_msm_and_exact_objective(self):
        self.assertEqual(msm([1,2,3],[1,2,3]),0)
        self.assertEqual(msm([0,0],[2,2]),4)
        self.assertEqual(msm([0,0,0],[0]),2)
        a,b = [0,1,4,0],[2,0,3]
        self.assertEqual(msm(a,b),msm(b,a))
        distances = np.array([[0,1,10,11],[1,0,9,10],[10,9,0,1],[11,10,1,0]])
        labels = medoids(distances,2)
        self.assertEqual(labels.tolist(),[0,0,1,1])
        obs,_ = observation(np.full(168,20.),17)
        out = validate(predict("shape-msm-medoids",obs,17),obs)
        self.assertEqual(out,predict("shape-msm-medoids",obs,18))
        with self.assertRaises(ValueError): msm(a,b,0)

    def test_truncated_usage_is_not_lost(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            r = {"id":"x", "run_id":"run", "case":{"id":"c","task":"forecast","network":"grid","condition":"nominal","split":"validation"},
                 "method_label":"llm", "hour":None, "status":"invalid", "elapsed_s":3.0}
            write(root/"traces/x.json", {"usage":[{"prompt_tokens":10,"completion_tokens":2048,"total_tokens":2058,
                "completion_tokens_details":{"reasoning_tokens":0}}], "http_request_elapsed_s":[2.0],
                "responses":[{"choices":[{"finish_reason":"length","message":{"reasoning_content":"unfinished thought"}}]}]})
            row = decision_cost(root,r)
            self.assertEqual(row["completion_tokens"],2048)
            self.assertEqual(row["truncated_calls"],1)
            self.assertEqual(row["reasoning_accounting_anomaly"],1)
            self.assertIsNone(row["peak_vram_bytes"])
            missing = decision_cost(root,{**r,"id":"missing"})
            self.assertIsNone(missing["completion_tokens"])
            self.assertIsNone(missing["http_s"])

    def test_forecast_zero_and_scaled_metrics(self):
        case = {"id":"x","unit":"x","network":"grid","task":"forecast","split":"test","condition":"nominal"}
        record = {"hour":None,"status":"ok","prediction":{"forecast_L_s":[0.]*24},"elapsed_s":None}
        row = case_score(case,[record],{"future_L_s":[0.]*24},1)
        self.assertEqual(row["smape_pct"],0)
        self.assertIsNone(row["wape_pct"])
        row = case_score(case,[record],{"future_L_s":[2.]*24},1)
        self.assertEqual(row["smape_pct"],200)
        self.assertEqual(row["wape_pct"],100)
        self.assertEqual(row["mase"],2)


if __name__=="__main__": unittest.main()
