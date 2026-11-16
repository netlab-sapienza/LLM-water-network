"""Correzione 0.2.1: test mirati per normalizzazione domanda (mu_net) e validazione
globale delle pressioni su tutte le giunzioni, con osservazioni sensore separate.
Software fixtures only. These are never experimental results."""
import copy
import unittest

import numpy as np
import wntr.network
import wntr.sim

from waterbench.data import configure, demands, demand_scale, layout, simulate


class DemandScaleChecks(unittest.TestCase):
    def _junction(self, w, name, base, elevation=5.0):
        w.add_junction(name, base_demand=base, elevation=elevation)

    def test_weighted_formula_with_global_multiplier(self):
        # J1: base 1.0 con pattern costante 2.0 (media 2.0); J2: base 3.0 senza pattern.
        # mu = DM * (1*2 + 3*1) / (1+3) = 0.5 * 5/4 = 0.625.
        # La media semplice dei pattern darebbe (2+1)/2=1.5 -> 0.75: il test la scarta.
        w = wntr.network.WaterNetworkModel()
        w.add_reservoir("R", base_head=50.0)
        self._junction(w, "J1", 1.0)
        self._junction(w, "J2", 3.0)
        w.add_pipe("P1", "R", "J1", length=1000, diameter=0.3)
        w.add_pipe("P2", "J1", "J2", length=1000, diameter=0.3)
        w.add_pattern("PAT", [2.0, 2.0])
        list(w.get_node("J1").demand_timeseries_list)[0].pattern_name = "PAT"
        w.options.hydraulic.demand_multiplier = 0.5
        self.assertAlmostEqual(demand_scale(w), 0.625, places=12)

    def test_categories_and_native_pattern_resolution(self):
        # J1 categoria A: base 1.0, pattern [1,3] a passo 1800 s (media oraria 2.0);
        # J1 categoria B: base 2.0 costante; J2: base 4.0 costante. DM = 1.
        # mu = (1*2 + 2*1 + 4*1) / 7 = 8/7.
        from wntr.network.elements import TimeSeries
        w = wntr.network.WaterNetworkModel()
        w.add_reservoir("R", base_head=50.0)
        self._junction(w, "J1", 1.0)
        self._junction(w, "J2", 4.0)
        w.add_pipe("P1", "R", "J1", length=1000, diameter=0.3)
        w.add_pipe("P2", "J1", "J2", length=1000, diameter=0.3)
        w.options.time.pattern_timestep = 1800
        w.add_pattern("PAT", [1.0, 3.0] * 24)
        list(w.get_node("J1").demand_timeseries_list)[0].pattern_name = "PAT"
        w.get_node("J1").demand_timeseries_list.append(TimeSeries(w.patterns, 2.0))
        self.assertAlmostEqual(demand_scale(w), 8.0 / 7.0, places=12)

    def test_no_pattern_is_unit_multiplier(self):
        w = wntr.network.WaterNetworkModel()
        w.add_reservoir("R", base_head=50.0)
        self._junction(w, "J1", 2.0)
        w.add_pipe("P1", "R", "J1", length=100, diameter=0.3)
        self.assertAlmostEqual(demand_scale(w), 1.0, places=12)


class GlobalPressureValidationChecks(unittest.TestCase):
    def test_healthy_scenario_passes_and_returns_sensor_only(self):
        w = wntr.network.WaterNetworkModel("data/networks/grid.inp")
        configure(w, 24)
        sensors, _ = layout(w, 4)
        demands(w, 1 + 0.25 * np.sin(2 * np.pi * (np.arange(24) - 7) / 24))
        result, pressure = simulate(copy.deepcopy(w), sensors)
        self.assertEqual(pressure.shape, (24, len(sensors)))
        self.assertGreater(float(pressure.min()), -0.01)
        # le osservazioni restano limitate ai nodi sensore: uguaglianza con il sottoinsieme
        raw = wntr.sim.WNTRSimulator(copy.deepcopy(w)).run_sim(convergence_error=True)
        self.assertTrue(np.allclose(pressure, raw.node["pressure"][sensors].to_numpy()))

    def test_negative_pressure_on_unobserved_node_is_rejected(self):
        # Caso reale di regressione (diagnosi 2026-10-03): nominale C-Town con profilo
        # a media 1.0x base -> 27 giunzioni non osservate < -0.01 m mentre i sensori del
        # layout-4 restano positivi (+25 m). La validazione pre-0.2.1 (solo sensori)
        # lasciava passare lo scenario; la validazione globale deve rifiutarlo.
        w = wntr.network.WaterNetworkModel("data/networks/ctown.inp")
        configure(w, 24)
        sensors, _ = layout(w, 4)
        demands(w, 1 + 0.25 * np.sin(2 * np.pi * (np.arange(24) - 7) / 24))
        raw = wntr.sim.WNTRSimulator(copy.deepcopy(w)).run_sim(convergence_error=True)
        p_all = raw.node["pressure"][w.junction_name_list].to_numpy()
        p_sens = raw.node["pressure"][sensors].to_numpy()
        self.assertGreater(float(p_sens.min()), -0.01)  # il vecchio controllo passava
        self.assertLess(float(p_all.min()), -0.01)       # ma esistono giunzioni negative non osservate
        with self.assertRaises(ValueError):
            simulate(copy.deepcopy(w), sensors)


if __name__ == "__main__":
    unittest.main()
