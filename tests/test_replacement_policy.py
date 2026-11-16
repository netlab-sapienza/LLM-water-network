"""Correzione 0.2.2: rifiuto e sostituzione deterministica delle unita' stocastiche
inammissibili per criteri idraulici prestabiliti, con classe/condizione invariate.
Software fixtures only. These are never experimental results."""
import tempfile
import unittest
from pathlib import Path

from waterbench.data import MAX_UNIT_ATTEMPTS, _candidate_indices, _unit_class, prepare
from waterbench.storage import read, write

CFG_CITOWN = {"stage": "pilot", "seed": 20261003, "networks": ["ctown"],
              "counts": {"train": 8, "validation": 8, "test": 4}, "sensor_count": 8,
              "conditions": ["nominal", "stress"],
              "pressure_noise_m": {"nominal": 0.05, "stress": 0.2},
              "leak_fraction": [0.05, 0.2], "signature_fraction": 0.1,
              "decision_hours": [8, 12, 16, 20, 23]}

CFG_GRID = {"stage": "pilot", "seed": 20261003, "networks": ["grid"],
            "counts": {"train": 4, "validation": 4, "test": 4}, "sensor_count": 8,
            "conditions": ["nominal", "stress"],
            "pressure_noise_m": {"nominal": 0.05, "stress": 0.2},
            "leak_fraction": [0.05, 0.2], "signature_fraction": 0.1,
            "decision_hours": [8, 12, 16, 20, 23]}


class CandidateChainChecks(unittest.TestCase):
    def test_chain_preserves_class_and_cap(self):
        cfg = {"conditions": ["nominal", "stress"]}
        chain = _candidate_indices(cfg, 5, set())  # slot di classe (nominal, leak)
        self.assertEqual(len(chain), MAX_UNIT_ATTEMPTS)
        self.assertEqual(chain[0], 5)
        for j in chain:
            self.assertEqual(_unit_class(cfg, j), _unit_class(cfg, 5))

    def test_chain_skips_used_indices(self):
        cfg = {"conditions": ["nominal", "stress"]}
        chain = _candidate_indices(cfg, 5, {5, 9})
        self.assertNotIn(5, chain)
        self.assertNotIn(9, chain)
        self.assertEqual(chain[0], 13)

    def test_replacement_collision_is_avoided(self):
        # Se lo slot 0 rifiuta il proprio indice e consuma il candidato dello slot 4,
        # lo slot 4 deve saltarlo: nessun indice viene mai usato due volte.
        cfg = {"conditions": ["nominal", "stress"]}
        used = set()
        chain0 = _candidate_indices(cfg, 0, used)
        used.add(chain0[0])                      # idx 0 rifiutato
        taken0 = [j for j in chain0 if j not in used][0]
        used.add(taken0)                         # slot 0 consuma idx 4
        chain4 = _candidate_indices(cfg, 4, used)
        self.assertNotIn(4, chain4)
        self.assertEqual(chain4[0], 8)


class ReplacementIntegrationChecks(unittest.TestCase):
    def _balance(self, cases, units_by_caseid):
        # per split: conteggio unita' e bilanciamento normali/perdite entro condizione
        for split in ("train", "validation", "test"):
            leak_cases = [c for c in cases if c["split"] == split and c["task"] == "leak"]
            self.assertEqual(len({c["unit"] for c in leak_cases}), CFG_CITOWN["counts"][split])
            for cond in ("nominal", "stress"):
                sub = [c for c in leak_cases if c["condition"] == cond]
                n_leak = sum(1 for c in sub if read(units_by_caseid[c["id"]])["has_leak"])
                self.assertEqual(n_leak, len(sub) // 2)

    def test_ctown_rejections_are_replaced_within_class(self):
        # Runtime ~2.5 min: preparazione completa su C-Town (seed 20261003, conteggi 8/8/4).
        # Deterministico e pre-validato: train slot5 e validation slot7 sono rifiutati dalla
        # validazione globale e sostituiti da candidati della medesima classe.
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            cfgpath = base / "cfg.json"
            write(cfgpath, CFG_CITOWN)
            out = base / "ds"
            prepare(cfgpath, Path("data/networks"), out)
            self.assertTrue((out / "manifest.json").exists())
            cases = read(out / "cases.json")
            units = {}
            for c in cases:
                units.setdefault(c["unit"], []).append(c)
            self.assertEqual(len(units), 20)  # 8+8+4 unita', due task ciascuna, senza duplicati
            self.assertTrue(all(len(v) == 2 for v in units.values()))
            evaldir = {p.stem: p for p in (out / "evaluator").glob("*.json") if p.name not in ("generation.json", "rejections.json")}
            self._balance(cases, evaldir)
            rejs = read(out / "evaluator/rejections.json")
            self.assertEqual(sorted((r["net"], r["split"], r["slot"]) for r in rejs),
                             [("ctown", "train", 5), ("ctown", "validation", 7)])
            gen = {g["unit"]: g for g in read(out / "evaluator/generation.json")}

            def as_dict(cls):
                return {"condition": cls[0], "leak": cls[1]}

            for r in rejs:
                self.assertEqual(r["class"], as_dict(_unit_class(CFG_CITOWN, r["slot"])))
                for cand in r["candidates"]:
                    self.assertFalse(cand["accepted"])
                    self.assertIn(cand["reason"], ("Negative junction pressure; reject scenario",
                                                   "Incomplete/nonfinite hydraulic simulation"))
                    self.assertLess(cand["global_min_m"], -0.01)
                    self.assertEqual(as_dict(_unit_class(CFG_CITOWN, cand["index"])), r["class"])
                repl = r["replacement"]
                self.assertEqual(as_dict(_unit_class(CFG_CITOWN, repl["index"])), r["class"])
                g = gen[repl["unit"]]
                self.assertEqual(g["slot"], r["slot"])
                self.assertEqual(g["source_index"], repl["index"])

    def test_grid_preparation_has_empty_rejections(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            cfgpath = base / "cfg.json"
            write(cfgpath, CFG_GRID)
            out = base / "ds"
            prepare(cfgpath, Path("data/networks"), out)
            self.assertEqual(read(out / "evaluator/rejections.json"), [])
            cases = read(out / "cases.json")
            self.assertEqual(len({c["unit"] for c in cases}), 12)


if __name__ == "__main__":
    unittest.main()
