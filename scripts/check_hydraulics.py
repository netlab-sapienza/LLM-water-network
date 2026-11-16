"""Compare nominal WNTR/EPANET 2.2 initial pressures. No scientific result fixture."""
import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import wntr

from waterbench.data import configure, demands
from waterbench.storage import write


def main():
    rows = []
    for path in sorted(Path("data/networks").glob("*.inp")):
        w = wntr.network.WaterNetworkModel(str(path))
        configure(w, 24)
        demands(w, 1 + 0.25*np.sin(2*np.pi*(np.arange(24)-7)/24))
        w.options.time.duration = 0
        a = wntr.sim.WNTRSimulator(copy.deepcopy(w)).run_sim(convergence_error=True)
        with tempfile.TemporaryDirectory(prefix="waterbench_epanet_") as d:
            b = wntr.sim.EpanetSimulator(copy.deepcopy(w)).run_sim(file_prefix=str(Path(d)/"nominal"), version=2.2)
        pa = a.node["pressure"][w.junction_name_list].iloc[0].to_numpy()
        pb = b.node["pressure"][w.junction_name_list].iloc[0].to_numpy()
        assert np.isfinite(pa).all() and np.isfinite(pb).all()
        error = float(np.max(np.abs(pa-pb)))
        assert error < 0.5, (path.name, error)
        row = {"network": path.stem, "initial_pressure_max_difference_m": error,
               "wntr_min_pressure_m": float(pa.min()), "epanet_min_pressure_m": float(pb.min()),
               "scope": "initial nominal state only; transient/tank agreement not inferred"}
        rows.append(row); print(row)
    write("results/checks/hydraulics.json", rows)


if __name__ == "__main__":
    main()
