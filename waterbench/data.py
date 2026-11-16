"""Network acquisition and independent simulated episodes; evaluator data stay separate."""
import copy
import datetime as dt
import hashlib
import math
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path

import networkx as nx
import numpy as np
import wntr

from .storage import code_hash, digest, environment, file_hash, read, write

SOURCES = {
    "ctown": {"url": "https://www.batadal.net/data/CTOWN.INP", "source": "https://www.batadal.net/data.html",
              "license": "CC BY 4.0 indicated by source page; retain attribution", "version": "BATADAL public CTOWN.INP"},
    "ltown": {"url": "https://zenodo.org/records/4017659/files/L-TOWN.inp?download=1",
              "source": "https://zenodo.org/records/4017659", "version": "Zenodo 4017659 v1",
              "md5": "bfbb16b8b463e0576200ba6ad360f68e", "license": "CC BY 4.0 (Zenodo record metadata); BattLeDIM code separately EUPL-1.2"},
}


def grid(size=4):
    w = wntr.network.WaterNetworkModel()
    for r in range(size):
        for c in range(size):
            w.add_junction(f"J{r}_{c}", base_demand=0.001, elevation=5 + r, coordinates=(c * 100, r * 100))
    w.add_reservoir("R", base_head=55, coordinates=(-100, 0))
    w.add_pipe("supply", "R", "J0_0", length=100, diameter=0.3, roughness=120)
    for r in range(size):
        for c in range(size):
            for rr, cc in ((r + 1, c), (r, c + 1)):
                if rr < size and cc < size:
                    w.add_pipe(f"P{r}_{c}_{rr}_{cc}", f"J{r}_{c}", f"J{rr}_{cc}", length=100,
                               diameter=0.15, roughness=120)
    return w


def acquire(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    gpath = directory / "grid.inp"
    if not gpath.exists():
        wntr.network.write_inpfile(grid(), str(gpath), units="LPS", version=2.2)
    provenance = {"grid": {"version": "4x4-v1", "source": "waterbench.data.grid", "license": "project-generated"}}
    for name, source in SOURCES.items():
        path = directory / f"{name}.inp"
        if not path.exists():
            with urllib.request.urlopen(source["url"], timeout=60) as response:
                content = response.read(2_000_001)
            if len(content) > 2_000_000:
                raise ValueError("Network download exceeded 2 MB cap")
            path.write_bytes(content)
        if source.get("md5") and hashlib.md5(path.read_bytes()).hexdigest() != source["md5"]:
            raise ValueError("L-Town checksum differs from Zenodo v1")
        provenance[name] = source
    for name in provenance:
        path = directory / f"{name}.inp"
        w = wntr.network.WaterNetworkModel(str(path))
        provenance[name] = dict(provenance[name], sha256=file_hash(path), acquired_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                                nodes=w.num_nodes, junctions=w.num_junctions, pipes=w.num_pipes,
                                tanks=w.num_tanks, pumps=w.num_pumps, reservoirs=w.num_reservoirs,
                                modifications="none; simulation copies configured separately")
    write(directory / "manifest.json", provenance)
    return provenance


def layout(w, count):
    """Deterministic geometric sensor/zone placement, independent of events."""
    names = sorted(w.junction_name_list)
    xy = np.array([w.get_node(n).coordinates for n in names])
    chosen = [0]
    while len(chosen) < min(count, len(names)):
        distances = ((xy[:, None] - xy[chosen]) ** 2).sum(axis=2).min(axis=1)
        distances[chosen] = -1
        chosen.append(int(distances.argmax()))
    assignment = ((xy[:, None] - xy[chosen]) ** 2).sum(axis=2).argmin(axis=1)
    zones = {f"Z{i:02}": [n for n, group in zip(names, assignment) if group == i] for i in range(len(chosen))}
    return [names[i] for i in chosen], zones


def demand_scale(w):
    """Correzione 0.2.1: fattore di ri-ancoraggio della domanda per rete.

    mu = DM * mean_h( sum_jc b_jc * m_jc(h) ) / base_total, h = 0..23,
    dove DM e' il moltiplicatore globale INP, b_jc la domanda base della categoria c
    della giunzione j, m_jc(h) il moltiplicatore di pattern valutato alla risoluzione
    nativa del pattern (pattern_timestep) e mediato dentro l'ora h; le categorie senza
    pattern hanno m = 1. La media temporale e' ponderata per domanda base e categorie
    (non e' la media semplice dei pattern). Ancorare i profili WaterBench a mu riporta
    il carico medio al punto di funzionamento pubblicato della rete.
    """
    dm = float(w.options.hydraulic.demand_multiplier or 1.0)
    pt = max(int(w.options.time.pattern_timestep), 1)
    steps = max(3600 // pt, 1)
    per_hour = np.zeros(24)
    base_total = 0.0
    for _, node in w.junctions():
        for item in node.demand_timeseries_list:
            b = float(item.base_value)
            base_total += b
            if item.pattern_name and item.pattern_name in w.pattern_name_list:
                mult = np.asarray(w.get_pattern(item.pattern_name).multipliers, dtype=float)
                for h in range(24):
                    idx = ((h * 3600) // pt + np.arange(steps)) % len(mult)
                    per_hour[h] += b * mult[idx].mean()
            else:
                per_hour += b
    if base_total <= 0:
        return 1.0
    return dm * float(per_hour.mean()) / base_total


def configure(w, hours):
    w.options.time.duration = (hours - 1) * 3600
    w.options.time.hydraulic_timestep = 900
    w.options.time.report_timestep = 3600
    w.options.time.pattern_timestep = 3600
    w.options.hydraulic.demand_model = "PDD"
    w.options.hydraulic.minimum_pressure = 0
    w.options.hydraulic.required_pressure = 20
    w.options.hydraulic.pressure_exponent = 0.5


def profile(rng, hours, phase=0):
    time = np.arange(hours) + phase
    daily = rng.lognormal(-0.003, 0.08, math.ceil(hours / 24) + 1)
    innovations = rng.normal(0, 0.025, hours)
    ar = np.zeros(hours)
    for t in range(1, hours):
        ar[t] = 0.8 * ar[t - 1] + innovations[t]
    return np.clip((1 + 0.25 * np.sin(2 * np.pi * (time - 7) / 24)
                    + 0.10 * np.sin(4 * np.pi * (time - 5) / 24))
                   * (1 + 0.06 * np.cos(2 * np.pi * time / 168))
                   * daily[np.arange(hours) // 24] * np.exp(ar), 0.25, 2.5)


def demands(w, values, rng=None):
    """Preserve base demands/categories, replace repeating INP demand patterns explicitly."""
    for i, (_, node) in enumerate(w.junctions()):
        curve = values.copy()
        if rng is not None:
            curve *= rng.lognormal(-0.0002, 0.02, len(values))
        name = f"WB{i}"
        if name in w.pattern_name_list:
            w.get_pattern(name).multipliers = curve.tolist()
        else:
            w.add_pattern(name, curve.tolist())
        for item in node.demand_timeseries_list:
            item.pattern_name = name
    w.options.hydraulic.demand_multiplier = 1.0


def simulate(w, sensors):
    # WNTR's native leak law supports time-controlled pressure-dependent leaks.
    result = wntr.sim.WNTRSimulator(w).run_sim(convergence_error=True)
    # Correzione 0.2.1: validazione globale su TUTTE le giunzioni (indipendente dai
    # sensori); le osservazioni restano limitate ai nodi sensore. Soglia e PDD invariati.
    junc = w.junction_name_list
    p_all = result.node["pressure"][junc].to_numpy()
    if p_all.shape != (24, len(junc)) or not np.isfinite(p_all).all():
        raise ValueError("Incomplete/nonfinite hydraulic simulation")
    if p_all.min() < -0.01:
        raise ValueError("Negative junction pressure; reject scenario")
    return result, result.node["pressure"][sensors].to_numpy()


def leak(w, node, area, start, end=24):
    w.get_node(node).add_leak(w, area=area, discharge_coeff=0.75,
                            start_time=start * 3600, end_time=end * 3600)


def base_total(w):
    return sum(sum(x.base_value for x in node.demand_timeseries_list) for _, node in w.junctions())


# Correzione 0.2.2: rifiuto e sostituzione deterministica delle unita' stocastiche
# inammissibili per criteri idraulici prestabiliti. Solo questi messaggi contano come
# rifiuto; ogni altro eccezione (errore di programmazione) interrompe la generazione.
MAX_UNIT_ATTEMPTS = 10
_HYDRAULIC_VALUE_ERRORS = ("Incomplete/nonfinite hydraulic simulation",
                           "Negative junction pressure; reject scenario")
_HYDRAULIC_RUNTIME_PREFIXES = ("Simulation did not converge", "Exceeded maximum number of trials")


class UnitRejected(Exception):
    """Rifiuto idraulico prestabilito di un'unita' stocastica (non errore di programmazione)."""

    def __init__(self, reason, global_min_m, leak_info):
        super().__init__(reason)
        self.reason = reason
        self.global_min_m = global_min_m
        self.leak_info = leak_info


def _is_hydraulic_rejection(exc):
    if isinstance(exc, ValueError) and str(exc) in _HYDRAULIC_VALUE_ERRORS:
        return True
    if isinstance(exc, RuntimeError) and str(exc).startswith(_HYDRAULIC_RUNTIME_PREFIXES):
        return True
    return False


def _unit_class(cfg, index):
    """Classe di un caso: (condizione nominal/stress, con/senza perdita), invariata dalle sostituzioni."""
    return (cfg["conditions"][(index // 2) % len(cfg["conditions"])], index % 2 == 1)


def _candidate_indices(cfg, slot, used, max_attempts=MAX_UNIT_ATTEMPTS):
    """Catena deterministica di candidati per uno slot: indici non ancora consumati, a partire
    dallo slot stesso, della medesima classe (stessa condizione e stessa classe con/senza perdita)."""
    out = []
    j = slot
    while len(out) < max_attempts:
        if j not in used and _unit_class(cfg, j) == _unit_class(cfg, slot):
            out.append(j)
        j += 1
    return out


def prepare(config_path, network_dir, output):
    cfg = read(config_path)
    if cfg["stage"] not in ("smoke", "pilot", "core-candidate"):
        raise ValueError("Unknown campaign stage")
    if set(cfg["counts"]) != {"train", "validation", "test"} or min(cfg["counts"].values()) < 2:
        raise ValueError("Need at least two independent episodes per split")
    if any(type(n) is not int or n % (2 * len(cfg["conditions"])) for n in cfg["counts"].values()):
        raise ValueError("Counts must balance normal/leak episodes within each condition")
    if not cfg["decision_hours"] or cfg["decision_hours"] != sorted(set(cfg["decision_hours"])) or any(type(h) is not int or not 8 <= h <= 23 for h in cfg["decision_hours"]):
        raise ValueError("Unique ordered decision hours from 8 to 23 required")
    if set(cfg["networks"]) - {"grid", "ctown", "ltown"} or not cfg["networks"]:
        raise ValueError("Unsupported/empty network selection")
    if not 0 < cfg["signature_fraction"] < 1 or not 0 < cfg["leak_fraction"][0] <= cfg["leak_fraction"][1] < 1:
        raise ValueError("Leak fractions outside (0,1)")
    if cfg["sensor_count"] < 1 or any(not np.isfinite(s) or s < 0 for s in cfg["pressure_noise_m"].values()):
        raise ValueError("Invalid sensor/noise configuration")
    output, network_dir = Path(output), Path(network_dir)
    if output.exists():
        raise FileExistsError("Dataset immutable; choose a new output directory")
    source = read(network_dir / "manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=output.name + ".", dir=output.parent))
    (temp / "source").mkdir()
    for module in Path(__file__).parent.glob("*.py"):
        (temp / "source" / module.name).write_bytes(module.read_bytes())
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    cases, generation, demand_scales, rejections = [], [], {}, []
    try:
        for net in cfg["networks"]:
            netpath = network_dir / f"{net}.inp"
            if file_hash(netpath) != source[net]["sha256"]:
                raise ValueError("Network provenance mismatch")
            w = wntr.network.WaterNetworkModel(str(netpath))
            configure(w, 24)
            mu = demand_scale(w)  # correzione 0.2.1: ancoraggio al carico medio pubblicato
            demand_scales[net] = mu
            total = base_total(w)
            sensors, zones = layout(w, cfg["sensor_count"])
            # deterministic nominal profile: remove stochastic component, identical for all methods.
            # Correzione 0.2.1: il fattore mu e' applicato moltiplicativamente una sola volta.
            nominal_profile = mu * (1 + 0.25 * np.sin(2 * np.pi * (np.arange(24) - 7) / 24))
            demands(w, nominal_profile)
            _, nominal = simulate(copy.deepcopy(w), sensors)
            nominal_area = total * mu * cfg["signature_fraction"] / (0.75 * np.sqrt(2 * 9.81 * 30))
            signatures = {}
            for zone, nodes in zones.items():
                candidate = copy.deepcopy(w)
                leak(candidate, nodes[0], nominal_area, 6)
                _, p = simulate(candidate, sensors)
                signatures[zone] = np.round((nominal - p)[8:].mean(axis=0), 6).tolist()
            graph = nx.Graph(w.to_graph(link_weight={name: getattr(link, "length", 0.0) for name, link in w.links()}))
            representatives = {z: nodes[0] for z, nodes in zones.items()}
            distances = {z: {zz: nx.shortest_path_length(graph, n, nn, weight="weight")
                             for zz, nn in representatives.items()} for z, n in representatives.items()}
            write(temp / "evaluator" / f"{net}_geometry.json", {"zones": zones, "distances_m": distances})
            def build_unit(split, index, slot):
                # Correzione 0.2.2: l'unita' dello slot `slot` e' generata al candidato `index`
                # (stessa classe); la sequenza RNG e' identica alla versione 0.2.1 (stesso seed).
                case_start = time.perf_counter()
                size = sum(p.stat().st_size for p in temp.rglob("*.json"))
                if size > cfg.get("max_dataset_mb", 2048) * 1024**2 or shutil.disk_usage(temp).free < 1024**3:
                    raise OSError("Dataset disk budget exceeded or less than 1 GiB free; preserve partial data")
                seed = int(digest([cfg["seed"], net, split, index])[:16], 16)
                rng = np.random.default_rng(seed)
                unit = digest(["unit", net, split, seed])[:20]
                condition = cfg["conditions"][(index // 2) % len(cfg["conditions"])]
                sigma = cfg["pressure_noise_m"][condition]
                truth_w = copy.deepcopy(w)
                real_profile = mu * profile(rng, 24)  # correzione 0.2.1: un'unica applicazione di mu
                demands(truth_w, real_profile, rng)
                for _, pipe in truth_w.pipes():
                    pipe.roughness *= rng.uniform(0.95, 1.05)
                active = index % 2 == 1
                zone = list(zones)[int(rng.integers(len(zones)))] if active else None
                node = str(rng.choice(zones[zone])) if active else None
                start = int(rng.integers(6, 19)) if active else None
                fraction = float(rng.uniform(*cfg["leak_fraction"])) if active else 0
                area = total * mu * fraction / (0.75 * np.sqrt(2 * 9.81 * 30))  # domanda media risultante
                leak_info = {"node": node, "start_hour": start, "fraction": fraction, "area_m2": area}
                if active:
                    leak(truth_w, node, area, start)
                evidence_w = copy.deepcopy(truth_w)  # stato pre-simulazione per l'evidenza del rifiuto
                try:
                    result, pressures = simulate(truth_w, sensors)
                except Exception as exc:  # filtrato sotto: solo criteri idraulici prestabiliti; gli altri interrompono
                    if not _is_hydraulic_rejection(exc):
                        raise
                    gmin = None
                    try:  # evidenza del rifiuto: ricalcolo deterministico dello stato pre-simulazione
                        raw = wntr.sim.WNTRSimulator(evidence_w).run_sim(convergence_error=True)
                        cand = float(raw.node["pressure"][w.junction_name_list].to_numpy().min())
                        gmin = cand if np.isfinite(cand) else None
                    except Exception:  # non convergente o non finito: nessuna pressione disponibile
                        pass
                    raise UnitRejected(str(exc), gmin, leak_info) from exc
                measured = pressures + rng.normal(0, sigma, pressures.shape)
                leakflow = result.node.get("leak_demand")
                flow = leakflow[node].to_numpy().tolist() if active and leakflow is not None else [0.0] * 24
                observation = {"task": "leak", "units": {"pressure": "m", "time": "hour"},
                               "sensor_ids": sensors, "zones": zones, "nominal_signatures_m": signatures,
                               "pressure_m": np.round(measured, 4).tolist(),
                               "nominal_pressure_m": np.round(nominal, 4).tolist(), "hours": list(range(24))}
                caseid = digest([unit, "leak"])[:20]
                write(temp / "observations" / f"{caseid}.json", observation)
                write(temp / "evaluator" / f"{caseid}.json", {"has_leak": active, "zone": zone, "node": node,
                      "start_hour": start, "end_hour": 24, "leak_area_m2": area, "leak_m3_s": flow,
                      "base_demand_m3_s": total, "reference_fraction_at_30m": fraction,
                      "pressure_min_m": float(pressures.min()), "pressure_max_m": float(pressures.max())})
                cases.append({"id": caseid, "unit": unit, "task": "leak", "network": net, "split": split,
                              "condition": condition, "decision_hours": cfg["decision_hours"]})
                # Independent 36-day realization per unit: supervised training uses separate train units.
                hourly = total * 1000 * mu * profile(rng, 36 * 24, phase=int(rng.integers(0, 168)))
                history, future = hourly[-192:-24], hourly[-24:]
                caseid = digest([unit, "forecast"])[:20]
                write(temp / "observations" / f"{caseid}.json", {"task": "forecast", "units": {"demand": "L/s", "time": "hour"},
                      "history_L_s": np.round(history, 6).tolist(), "horizon": 24})
                write(temp / "evaluator" / f"{caseid}.json", {"future_L_s": future.tolist()})
                if split == "train":
                    write(temp / "training" / f"{caseid}.json", {"series_L_s": hourly.tolist()})
                cases.append({"id": caseid, "unit": unit, "task": "forecast", "network": net, "split": split,
                              "condition": "synthetic-demand"})
                generation.append({"unit": unit, "seed": seed, "split": split, "network": net, "hydraulic_solver": "WNTRSimulator PDD",
                                   "demand_scale": mu, "slot": slot, "source_index": index,
                                   "generation_elapsed_s": time.perf_counter() - case_start})
                return unit

            for split, count in cfg["counts"].items():
                # Correzione 0.2.2: uno slot per caso previsto; i candidati dello slot mantengono
                # classe e condizione. I rifiuti (solo idraulici) sono loggati con evidenza e la
                # generazione prosegue al candidato successivo; cap MAX_UNIT_ATTEMPTS, poi stop.
                used = set()
                for slot in range(count):
                    chain, placed, accepted_index = [], None, None
                    for index in _candidate_indices(cfg, slot, used):
                        try:
                            placed = build_unit(split, index, slot)
                            accepted_index = index
                            break
                        except UnitRejected as rej:
                            used.add(index)  # deterministico: un candidato rifiutato non verra' mai accettato
                            chain.append({"index": index, "seed": int(digest([cfg["seed"], net, split, index])[:16], 16),
                                          "reason": rej.reason, "global_min_m": rej.global_min_m,
                                          **rej.leak_info, "accepted": False})
                            print(f"REJECTED {net} {split} slot {slot + 1}/{count} index {index}: {rej.reason} "
                                  f"(min {rej.global_min_m} m)", flush=True)
                    if placed is None:
                        raise RuntimeError(f"Quota non raggiunta per {net}/{split} slot {slot}: "
                                           f"{len(chain)} candidati rifiutati (cap {MAX_UNIT_ATTEMPTS}); "
                                           "generazione interrotta senza manifest finale")
                    used.add(accepted_index)
                    if chain:
                        rejections.append({"net": net, "split": split, "slot": slot,
                                           "class": {"condition": cfg["conditions"][(slot // 2) % len(cfg["conditions"])],
                                                     "leak": slot % 2 == 1},
                                           "candidates": chain,
                                           "replacement": {"index": accepted_index, "unit": placed}})
                    print(f"prepared {net} {split} slot {slot + 1}/{count} (index {accepted_index})", flush=True)
        write(temp / "evaluator" / "generation.json", generation)
        write(temp / "evaluator" / "rejections.json", rejections)  # correzione 0.2.2: catena di rifiuti/sostituzioni
        write(temp / "cases.json", cases)
        files = {p.relative_to(temp).as_posix(): file_hash(p) for p in sorted(temp.rglob("*")) if p.is_file()}
        manifest = {"schema": 1, "config": cfg, "code_hash": code_hash(), "environment": environment(),
                    "started_utc": started, "networks": {n: source[n] for n in cfg["networks"]},
                    "demand_scales": demand_scales, "files": files}
        manifest["id"] = digest(manifest)
        write(temp / "manifest.json", manifest)
        temp.rename(output)
    except BaseException:
        # Keep partial simulations inspectable, never label as a completed dataset.
        print(f"Preparation interrupted; incomplete directory: {temp}", flush=True)
        raise
    return manifest
