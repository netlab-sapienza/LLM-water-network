# Setup and running experiments

Run commands from the repository root with Python 3.12 or 3.13. The standard installation is in the main README. Core hydraulic simulation uses WNTR and its bundled EPANET libraries; no global simulator service is required. The original lock file records the scientific environment rather than a guarantee of wheels for every platform.

## 1. Generate a small dataset

```bash
python -m waterbench prepare --config configs/smoke.json --output data/datasets/demo
python -m waterbench audit --dataset data/datasets/demo
```

The smoke configuration has four sensors and small split counts on Grid, for a quick functionality check. It is not the paper's complete experiment. `configs/pilot.json` illustrates eight sensors and all three networks. Choose seeds and split counts explicitly for a new study. Included INP files avoid downloads; `python -m waterbench acquire` can acquire/check the public networks separately.

Dataset directories are immutable. Choose a new output path when changing the configuration. Observations, evaluator targets, training data, generation logs, source snapshots, and checksums are stored separately.

## 2. Add consumption-shape clustering

The hydraulic generator creates leak and forecast cases. Add the third task without modifying the parent:

```bash
python -m waterbench augment-cluster --dataset data/datasets/demo --output data/datasets/demo-three-tasks --seed 42
python -m waterbench audit --dataset data/datasets/demo-three-tasks
```

Consumer profiles come from synthetic daily shape families; their latent labels stay in evaluator files.

## 3. Run and score baselines

```bash
python -m waterbench run --dataset data/datasets/demo-three-tasks --methods seasonal-naive shape-kmeans --tasks forecast cluster --networks grid --split validation
```

The command prints its run directory. Insert that path below; then insert the evaluation path printed by the evaluator:

```bash
python -m waterbench evaluate --dataset data/datasets/demo-three-tasks --runs RUN_DIRECTORY --bootstrap 1000
python -m waterbench report --evaluation EVALUATION_DIRECTORY
```

Use `--methods` to select other core specialists. See `python -m waterbench run --help` for task, network, repetition, and seed controls. Repeating the same invocation resumes the same run without replacing completed decisions. Separate explicit repetition IDs create separate runs. Do not reuse a failed attempt under a new repetition as an undocumented retry.

## 4. Connect a local language model

Start your own compatible service separately. Copy `configs/endpoint.example.json` to `configs/model.local.json`, then fill in the actual model ID, endpoint URL, generation settings, and provenance fields. Keep `mode` set to `direct`. Unknown revision or template information must remain unknown; an example value is not evidence of provenance.

Credentials are read only from the environment variable named by `api_key_env` (default `WATERBENCH_API_KEY`). Never put a token in JSON, command-line arguments, or logs. The local runner accepts private/loopback HTTP(S) endpoints and rejects embedded URL credentials.

```bash
python -m waterbench run --dataset data/datasets/demo-three-tasks --methods llm --split validation --endpoint configs/model.local.json
```

Set `context_limit_tokens`, `max_tokens`, and `timeout_s` for the actual service. The code checks context conservatively and rejects truncated or invalid outputs; it does not silently repair them. Discovery via `python -m waterbench discover --help` supports services that explicitly expose loaded-model inventory. A generic endpoint need not support that extended inventory contract.

## 5. Export prompts or import offline answers

```bash
python -m waterbench export-cases --dataset data/datasets/demo-three-tasks --output generated/requests --split validation
python -m waterbench import-responses --help
```

Export writes exact messages and observation/prompt hashes. Import validates the first raw answer for each request and preserves invalid responses and missing coverage. `configs/subscription.example.json` describes the required metadata. Private cloud-service automation is deliberately outside this repository; the import/export contract is service-independent.

## 6. Freeze a test campaign

Training and validation must precede test execution. The generic CLI and the final campaign helper enforce provenance and freeze checks. Start with:

```bash
python scripts/freeze_final.py --help
python scripts/run_pilot3.py --help
```

The freeze helper requires a completed pilot evaluation, its original run files, a specialist-selection document, notes, explicit baseline seeds, endpoints, and a declaration explaining unavailable metadata. It retains invalid model outcomes but rejects missing/infrastructure-failed pilot evidence. Do not manufacture these files or bypass the guard.

`run_pilot3.py` prints a plan without inference by default; `--execute` is required to run it. Specify `--split test` explicitly for a frozen final test. Its default repetition count is an example workflow setting, not a claim about the full server campaign in the paper. Use at least 10,000 bootstrap resamples for final reporting, and preserve independent scenario blocks.

## 7. Supplementary specialists

LightGBM, Chronos-2, and KASBA use a separate dependency environment to avoid replacing core packages:

```bash
python -m venv .venv-supplementary
# Activate this environment using the same platform-specific command as above.
python -m pip install -r requirements-supplementary.lock.txt
python scripts/run_supplementary_baselines.py --help
```

The adapter preserves the original batch layout: `data/datasets/core-20261031`, `core-20261101`, and `core-20261102`, each with all three tasks and its own valid `freeze.json`. These are expected paths for datasets you generate and freeze; no experimental datasets are distributed. Change the explicit `DATA`, `OUT`, seed list, and `CFG` in a study-specific copy if using a different experiment layout.

```bash
python scripts/run_supplementary_baselines.py --prepare-model
python scripts/run_supplementary_baselines.py --check
# After preparing and freezing all expected datasets:
python scripts/run_supplementary_baselines.py
```

The first command downloads Chronos-2 weights and records their revision and checksums. The check exercises one validation input per task. Full execution runs validation and test and writes local results. These are explicit computational steps, not part of installing or testing the core package.

## Verify the software

```bash
python scripts/verify_release.py
python -m unittest discover -s tests -v
```

Tests use simulated HTTP services and generated fixtures, never an external LLM. Some hydraulic tests take a few minutes. Fixtures demonstrate software behavior and are not paper results. Generated datasets and outputs are ignored by Git.
