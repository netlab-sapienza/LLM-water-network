"""Synthetic consumption-shape partitioning, with a shared, explicit observation contract."""
import shutil
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .storage import code_hash, digest, file_hash, read, verify_dataset, write

METHODS = {"shape-kmeans", "shape-dtw-average", "shape-msm-medoids"}


def observation(history, seed, consumers=12):
    history = np.asarray(history, dtype=float)
    if history.shape != (168,) or not np.isfinite(history).all() or (history < 0).any() or consumers < 3 or consumers % 3:
        raise ValueError("Need 168 finite nonnegative hourly demands and a positive multiple of 3 consumers")
    rng = np.random.default_rng(seed)
    h = np.arange(168) % 24
    # Synthetic archetypes, not inferred customer demographics or field measurements.
    shapes = np.array([0.45 + np.exp(-((h-8)/2.5)**2) + 0.8*np.exp(-((h-19)/3)**2),
                       0.4 + 1.1*np.exp(-((h-14)/4)**2),
                       0.8 + 0.2*np.cos(2*np.pi*h/24)])
    families = np.repeat(np.arange(3), consumers//3)
    rng.shuffle(families)
    weights = []
    for family in families:
        weights.append(np.roll(shapes[family], int(rng.integers(-1, 2))) * rng.lognormal(0, 0.12, 168))
    weights = np.asarray(weights)
    # Each hour's synthetic consumers sum exactly to the parent's intended demand.
    series = weights / weights.sum(axis=0) * np.asarray(history)
    daily = series.reshape(consumers, 7, 24).mean(axis=1)
    mean, std = daily.mean(axis=1, keepdims=True), daily.std(axis=1, keepdims=True)
    normalized = np.divide(daily-mean, std, out=np.zeros_like(daily), where=std>1e-12)
    ids = [f"C{i:02}" for i in range(consumers)]
    obs = {"task": "cluster", "consumer_ids": ids, "k": 3,
           "profiles": dict(zip(ids, np.round(normalized, 6).tolist())),
           "representation": "z-normalized 24-hour mean profiles from seven past days; synthetic consumers",
           "units": {"profile": "dimensionless z-score", "time": "hour of day"}}
    truth = {"assignments": dict(zip(ids, families.tolist())), "source": "synthetic latent archetype, not field label"}
    return obs, truth


def augment(dataset, output, seed=20261005):
    source, target = Path(dataset), Path(output)
    if source.resolve() == target.resolve() or source.resolve() in target.resolve().parents:
        raise ValueError("Output must be outside the parent dataset")
    original = verify_dataset(source)
    if target.exists():
        raise ValueError("Dataset immutable; choose a new output")
    cases = read(source/"cases.json")
    if any(c["task"] == "cluster" for c in cases):
        raise ValueError("Clustering already present")
    shutil.copytree(source, target)
    # A copied parent manifest must not certify a partial extension.
    (target/"manifest.json").unlink()
    if (target/"freeze.json").exists():
        (target/"freeze.json").unlink()  # New dataset requires its own freeze.
    for case in list(cases):
        if case["task"] != "forecast":
            continue
        history = read(source/"observations"/(case["id"]+".json"))["history_L_s"]
        scenario_seed = int(digest([seed, case["unit"], "cluster-v1"])[:16], 16)
        obs, truth = observation(history, scenario_seed)
        cid = digest([case["unit"], "cluster", seed])[:20]
        write(target/"observations"/(cid+".json"), obs)
        write(target/"evaluator"/(cid+".json"), {**truth, "seed": scenario_seed})
        cases.append({**{k:v for k,v in case.items() if k != "decision_hours"}, "id": cid,
                      "task": "cluster", "condition": "synthetic-consumption-shapes"})
    write(target/"cases.json", cases)
    shutil.copyfile(__file__, target/"source"/"clustering_extension.py")
    files = {p.relative_to(target).as_posix(): file_hash(p) for p in sorted(target.rglob("*")) if p.is_file()}
    m = {**{k:v for k,v in original.items() if k not in ("id", "files", "config", "code_hash")},
         "config": {**original["config"], "cluster_seed": seed}, "code_hash": code_hash(), "files": files,
         "parent_dataset_id": original["id"], "extension": "cluster-v1; parent hydraulics unchanged; synthetic shapes"}
    m["id"] = digest(m); write(target/"manifest.json", m)
    return target


def dtw(a, b):
    cost = np.full((len(a)+1, len(b)+1), np.inf); cost[0,0] = 0
    for i, x in enumerate(a, 1):
        for j, y in enumerate(b, 1):
            cost[i,j] = (x-y)**2 + min(cost[i-1,j], cost[i,j-1], cost[i-1,j-1])
    return float(np.sqrt(cost[-1,-1]))


def msm(a, b, c=1.0):
    """Univariate move/split/merge recurrence (Stefan et al., TKDE 2013)."""
    if c <= 0 or not math.isfinite(c): raise ValueError("MSM cost must be finite and positive")
    def edit(x, y, z):
        return c if min(y,z) <= x <= max(y,z) else c + min(abs(x-y), abs(x-z))
    d = np.empty((len(a), len(b)))
    d[0,0] = abs(a[0]-b[0])
    for i in range(1,len(a)): d[i,0] = d[i-1,0]+edit(a[i],a[i-1],b[0])
    for j in range(1,len(b)): d[0,j] = d[0,j-1]+edit(b[j],a[0],b[j-1])
    for i in range(1,len(a)):
        for j in range(1,len(b)):
            d[i,j] = min(d[i-1,j-1]+abs(a[i]-b[j]), d[i-1,j]+edit(a[i],a[i-1],b[j]),
                         d[i,j-1]+edit(b[j],a[i],b[j-1]))
    return float(d[-1,-1])


def medoids(distances, k):
    # Exactly 220 candidates for the benchmark's n=12, k=3; no heuristic initialization.
    if math.comb(len(distances), k) > 50000: raise ValueError("Exact medoids size cap exceeded")
    chosen = min(combinations(range(len(distances)), k), key=lambda m: float(distances[:,m].min(axis=1).sum()))
    labels = distances[:,chosen].argmin(axis=1)
    for label, index in enumerate(chosen): labels[index] = label  # zero-distance ties keep k nonempty
    return labels


def predict(name, obs, seed, msm_cost=1.0):
    if name not in METHODS:
        raise ValueError("Unknown clustering baseline")
    ids = obs["consumer_ids"]
    x = np.array([obs["profiles"][i] for i in ids])
    if name == "shape-kmeans":
        labels = KMeans(n_clusters=obs["k"], n_init=10, random_state=seed).fit_predict(x)
    elif name == "shape-msm-medoids":
        distances = np.zeros((len(x),len(x)))
        for i in range(len(x)):
            for j in range(i): distances[i,j] = distances[j,i] = msm(x[i],x[j],msm_cost)
        labels = medoids(distances, obs["k"])
    else:
        distances = np.array([[dtw(a,b) for b in x] for a in x])
        labels = AgglomerativeClustering(n_clusters=obs["k"], metric="precomputed", linkage="average").fit_predict(distances)
    return {"assignments": dict(zip(ids, labels.tolist()))}


def scores(prediction, truth, ids):
    a, b = ([v["assignments"][i] for i in ids] for v in (prediction, truth))
    return {"ari": float(adjusted_rand_score(b,a)), "nmi": float(normalized_mutual_info_score(b,a))}
