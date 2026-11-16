"""Small baselines. They are adaptations, not certified SOTA reproductions."""
import json
import math
import os
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

from .storage import read
from . import clustering

TASKS = ["leak", "forecast", "cluster"]
BASELINE_TASKS = {"physics-residual": "leak", "extra-trees": "leak", "seasonal-naive": "forecast",
                  "hist-gradient-boosting": "forecast", "shape-kmeans": "cluster", "shape-dtw-average": "cluster"}
BASELINE_TASKS.update({"residual-svm": "leak", "lasso-direct": "forecast",
                       "forecast-ensemble": "forecast", "shape-msm-medoids": "cluster"})
PRIMARY = {"leak": "f1", "forecast": "mae", "cluster": "ari"}


def supports(method, task):
    return method == "llm" or BASELINE_TASKS.get(method) == task

SYSTEM = "You are evaluating water-network observations. Use only the supplied observations. Return exactly the specified JSON object, no Markdown or commentary."
INSTRUCTIONS = {
    "leak": "Decide whether a leak is ACTIVE at the last observed hour. Historical nominal pressures and nominal zone leak signatures are imperfect model predictions, not ground truth. Use only past and present observations. Output {\"alarm\":boolean,\"score\":number in [0,1],\"ranking\":[up to 3 unique zone IDs]}. If alarm is false ranking must be []. Zones group physical junction leak locations.",
    "forecast": "Forecast the next 24 hourly intended aggregate consumer demands, in L/s, from the 168 hourly past values. This is synthetic consumer demand, not inlet flow or delivered PDD demand. Output {\"forecast_L_s\":[24 finite nonnegative numbers]}.",
    "cluster": "Partition the supplied synthetic consumer profiles by daily consumption shape into exactly k nonempty clusters. Profiles are already z-normalized; use only the supplied profiles. Output {\"assignments\":{consumer_ID:integer cluster label}} with every supplied consumer ID exactly once, labels 0 through k-1. Cluster label order is arbitrary.",
}


def prompt(obs):
    return INSTRUCTIONS[obs["task"]] + "\nOBSERVATIONS\n" + json.dumps(obs, sort_keys=True, separators=(",", ":"), allow_nan=False)


def visible(obs, hour=None):
    if obs["task"] != "leak":
        return obs.copy()
    return {k: v[:hour + 1] if k in ("pressure_m", "nominal_pressure_m", "hours") else v for k, v in obs.items()}


def validate(value, obs):
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    def numeric(x):
        return type(x) in (int, float) and math.isfinite(x)
    if obs["task"] == "forecast":
        if set(value) != {"forecast_L_s"}:
            raise ValueError("Unexpected forecast fields")
        values = value["forecast_L_s"]
        if not isinstance(values, list) or len(values) != 24 or not all(numeric(x) and x >= 0 for x in values):
            raise ValueError("Need 24 finite nonnegative L/s values")
    elif obs["task"] == "cluster":
        assignments = value.get("assignments")
        if set(value) != {"assignments"} or not isinstance(assignments, dict) or set(assignments) != set(obs["consumer_ids"]):
            raise ValueError("Need exactly one assignment per supplied consumer")
        if not all(type(v) is int and 0 <= v < obs["k"] for v in assignments.values()) or set(assignments.values()) != set(range(obs["k"])):
            raise ValueError("Need exactly k nonempty integer-labeled clusters")
    elif obs["task"] == "leak":
        if set(value) != {"alarm", "score", "ranking"} or type(value["alarm"]) is not bool:
            raise ValueError("Invalid alarm schema")
        if not numeric(value["score"]) or not 0 <= value["score"] <= 1:
            raise ValueError("Invalid probability score")
        ranks = value["ranking"]
        if not isinstance(ranks, list) or len(ranks) > 3 or not all(isinstance(z, str) and z in obs["zones"] for z in ranks):
            raise ValueError("Unknown zone or excessive ranking")
        if len(set(ranks)) != len(ranks) or (not value["alarm"] and ranks):
            raise ValueError("Duplicate ranks or ranking without alarm")
        if value["alarm"] and not ranks:
            raise ValueError("Alarm requires localization ranking")
    else:
        raise ValueError("Unknown task")
    return value


def residual(obs):
    r = np.asarray(obs["nominal_pressure_m"]) - np.asarray(obs["pressure_m"])
    return r[-4:].mean(axis=0) - r[:6].mean(axis=0)


def features(obs):
    r = residual(obs)
    return np.r_[r, np.abs(r), r.mean(), r.std()]


def physics(obs, threshold):
    r = residual(obs)
    score = max(0.0, float(np.linalg.norm(r)))
    ranking = sorted(obs["zones"], key=lambda z: float(np.linalg.norm(
        r - np.asarray(obs["nominal_signatures_m"][z]) * max(0, np.dot(r, obs["nominal_signatures_m"][z]))
        / max(1e-12, np.dot(obs["nominal_signatures_m"][z], obs["nominal_signatures_m"][z])))))
    alarm = score > threshold
    return {"alarm": alarm, "score": float(score / (score + max(threshold, 1e-6))), "ranking": ranking[:3] if alarm else []}


def forecast_features(history, horizon):
    h = np.asarray(history)
    # Same 168-value history as the LLM; no future covariates.
    return np.r_[h[-24:], h[-48:-24], h[:24], h.mean(), h.std(), horizon]


def fit(dataset, task, network, name, cfg):
    if task == "cluster" and name in clustering.METHODS:
        return {"name": name, "seed": cfg["training_seed"], "msm_cost": cfg.get("msm_cost", 1.0),
                "training_splits": [], "information_regime": "observation-only"}
    dataset = Path(dataset)
    cases = [c for c in read(dataset / "cases.json") if c["network"] == network and c["task"] == task]
    if task == "forecast":
        training = [read(dataset / "training" / f"{c['id']}.json")["series_L_s"] for c in cases if c["split"] == "train"]
        scale = float(np.mean(np.concatenate([np.abs(np.asarray(s)[24:] - np.asarray(s)[:-24]) for s in training])))
        if name == "seasonal-naive":
            return {"name": name, "scale": scale, "training_splits": [], "scale_splits": ["train"]}
        if name == "forecast-ensemble":
            names = ["seasonal-naive", "hist-gradient-boosting", "lasso-direct"]
            members = [fit(dataset, task, network, n, cfg) for n in names]
            # Fixed equal weights, selected before results; no validation/test label tuning.
            return {"name": name, "scale": scale, "model": members, "members": names,
                    "weights": [1/3]*3, "training_splits": ["train"]}
        if name == "lasso-direct":
            x, y = [], []
            level = max(float(np.mean(np.concatenate(training))), 1e-12)
            for series in training:
                for t in range(168, len(series)-24+1, 24):
                    x.append(np.asarray(series[t-168:t])/level)
                    y.append(np.asarray(series[t:t+24])/level)
            model = make_pipeline(StandardScaler(), Lasso(alpha=cfg.get("lasso_alpha", 0.001),
                                  max_iter=10000, tol=1e-4)).fit(x, y)
            return {"name": name, "scale": scale, "level": level, "model": model,
                    "alpha": cfg.get("lasso_alpha", 0.001), "training_splits": ["train"]}
        if name != "hist-gradient-boosting":
            raise ValueError("Unknown forecasting baseline")
        x, y = [], []
        for series in training:
            for t in range(168, len(series) - 24 + 1, 24):
                for horizon in range(24):
                    x.append(forecast_features(series[t-168:t], horizon))
                    y.append(series[t+horizon])
        model = HistGradientBoostingRegressor(max_iter=cfg["boosting_iterations"], max_leaf_nodes=15, l2_regularization=1,
                                              random_state=cfg["training_seed"]).fit(x, y)
        return {"name": name, "scale": scale, "model": model}
    if name not in ("physics-residual", "extra-trees", "residual-svm"):
        raise ValueError("Unknown leak baseline")
    normal_scores, x, y, zone_x, zone_y = [], [], [], [], []
    for case in cases:
        if case["split"] != ("validation" if name == "physics-residual" else "train"):
            continue
        obs = read(dataset / "observations" / f"{case['id']}.json")
        truth = read(dataset / "evaluator" / f"{case['id']}.json")
        for hour in case["decision_hours"]:
            part = visible(obs, hour)
            active = bool(truth["has_leak"] and truth["start_hour"] <= hour < truth["end_hour"])
            if case["split"] == "validation" and not truth["has_leak"]:
                normal_scores.append(float(np.linalg.norm(residual(part))))
            if case["split"] == "train":
                x.append(features(part)); y.append(active)
                if active:
                    zone_x.append(features(part)); zone_y.append(truth["zone"])
    threshold = float(np.quantile(normal_scores, 0.95)) if normal_scores else None
    if name == "physics-residual" and threshold is None:
        raise ValueError("Physics calibration needs nominal validation cases")
    state = {"name": name, "threshold": threshold}
    if name == "extra-trees":
        state["training_splits"] = ["train"]
        state["detector"] = ExtraTreesClassifier(n_estimators=cfg["trees"], min_samples_leaf=2, class_weight="balanced",
                                                n_jobs=1, random_state=cfg["training_seed"]).fit(x, y)
        state["localizer"] = ExtraTreesClassifier(n_estimators=cfg["trees"], n_jobs=1,
                                                 random_state=cfg["training_seed"]).fit(zone_x, zone_y)
    if name == "residual-svm":
        def classifier(labels, balanced=False):
            estimator = (DummyClassifier(strategy="most_frequent") if len(set(labels)) == 1 else
                         SVC(C=cfg.get("svm_c", 1.0), kernel="rbf", gamma="scale",
                             class_weight="balanced" if balanced else None, decision_function_shape="ovr"))
            return make_pipeline(StandardScaler(), estimator)
        state["detector"] = classifier(y, True).fit(x, y)
        state["localizer"] = classifier(zone_y).fit(zone_x, zone_y)
        state["score_semantics"] = "sigmoid of SVM margin; not a calibrated probability"
        state["training_splits"] = ["train"]
    if name == "physics-residual": state["training_splits"] = ["validation"]
    return state


def predict(state, obs):
    name = state["name"]
    if name in clustering.METHODS:
        return clustering.predict(name, obs, state["seed"], state.get("msm_cost", 1.0))
    if name == "seasonal-naive":
        return {"forecast_L_s": obs["history_L_s"][-24:]}
    if name == "hist-gradient-boosting":
        output = state["model"].predict([forecast_features(obs["history_L_s"], h) for h in range(24)])
        return {"forecast_L_s": np.maximum(output, 0).tolist()}
    if name == "lasso-direct":
        output = state["model"].predict([np.asarray(obs["history_L_s"])/state["level"]])[0]*state["level"]
        return {"forecast_L_s": np.maximum(output, 0).tolist()}
    if name == "forecast-ensemble":
        values = [predict(member, obs)["forecast_L_s"] for member in state["model"]]
        return {"forecast_L_s": np.average(values, axis=0, weights=state["weights"]).tolist()}
    if name == "physics-residual":
        return physics(obs, state["threshold"])
    f = features(obs).reshape(1, -1)
    if name == "residual-svm":
        detector, local = state["detector"], state["localizer"]
        alarm = bool(detector.predict(f)[0])
        score = (float(1/(1+np.exp(-np.clip(detector.decision_function(f)[0], -50, 50))))
                 if len(detector.classes_) > 1 else float(alarm))
        if len(local.classes_) == 1:
            order = [0]
        else:
            margins = np.asarray(local.decision_function(f)).ravel()
            if len(local.classes_) == 2: margins = np.array([-margins[0], margins[0]])
            order = np.argsort(-margins, kind="stable")[:3]
        return {"alarm": alarm, "score": score, "ranking": [str(local.classes_[i]) for i in order] if alarm else []}
    classes = list(state["detector"].classes_)
    score = float(state["detector"].predict_proba(f)[0][classes.index(True)]) if True in classes else 0.0
    local = state["localizer"]
    order = np.argsort(-local.predict_proba(f)[0], kind="stable")
    alarm = score >= 0.5
    return {"alarm": alarm, "score": score, "ranking": [str(local.classes_[i]) for i in order[:3]] if alarm else []}


def request(endpoint, messages, tools=None):
    payload = {"model": endpoint["model"], "messages": messages, **endpoint["generation"]}
    if tools:
        payload["tools"] = tools
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(endpoint.get("api_key_env", "WATERBENCH_API_KEY"))
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(endpoint["base_url"].rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload, allow_nan=False).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=endpoint["timeout_s"]) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("API response exceeded 2 MB cap")
    def reject_constant(value):
        raise ValueError("Nonfinite JSON constant")
    try:
        result = json.loads(raw, parse_constant=reject_constant)
    except (ValueError, UnicodeDecodeError):
        return {"raw_body": raw.decode("utf-8", errors="replace"), "parse_error": "Non-JSON API body"}
    if not isinstance(result, dict):
        return {"raw_body": raw.decode("utf-8", errors="replace"), "parse_error": "Non-object API body"}
    result["raw_body"] = raw.decode("utf-8", errors="replace")
    return result


TOOL = {"type": "function", "function": {"name": "observation_summary", "description": "Compute descriptive statistics of already visible observations. No solver, hidden data or future access.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}}


def summary(obs):
    if obs["task"] == "cluster":
        return {"consumer_ids": obs["consumer_ids"], "profiles": obs["profiles"], "k": obs["k"]}
    if obs["task"] == "forecast":
        h = np.asarray(obs["history_L_s"])
        return {"mean_L_s": float(h.mean()), "std_L_s": float(h.std()), "last_day_L_s": h[-24:].tolist()}
    return {"mean_pressure_deficit_after_initial_history_m": residual(obs).tolist(), "sensor_ids": obs["sensor_ids"]}
