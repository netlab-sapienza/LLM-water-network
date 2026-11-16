# Prompts and example requests

The authoritative implementation is `waterbench/methods.py`: `SYSTEM`, `INSTRUCTIONS`, `visible`, and `prompt`.

| File | Content |
|---|---|
| `system.txt` | Shared system message |
| `leak.txt`, `forecast.txt`, `cluster.txt` | Exact task instructions |
| `examples/leak.json` | Complete request with a causal Grid pressure observation |
| `examples/forecast.json` | Complete request with 168 hourly demand values |
| `examples/cluster.json` | Complete request with twelve normalized profiles |

Examples contain inputs only, without model answers, evaluator targets, or experimental results.

The user message is the task instruction followed by the literal separator `\nOBSERVATIONS\n` and the observation serialized with `json.dumps(observation, sort_keys=True, separators=(',', ':'), allow_nan=False)`.

Each request starts with fresh context. For leak detection, `visible(observation, hour)` truncates `hours`, `pressure_m`, and `nominal_pressure_m` at the current decision hour. Forecast and clustering observations are supplied intact. Evaluator targets are never included.

Instruction files have a final newline for readability; it is not part of the request message. Example JSON files preserve the exact message strings. `python scripts/verify_release.py` checks them against the code.

To export exact messages for a newly generated validation dataset:

```bash
python -m waterbench export-cases --dataset data/datasets/demo-three-tasks --output generated/requests --split validation
```

See `docs/reproduction.md` for preparation. Test export additionally requires a valid freeze. Cloud services may apply hidden instructions; these files specify only the benchmark-controlled messages.
