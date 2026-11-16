# Models and baselines

The paper considers six local configurations and three subscription-cloud configurations. Model names identify the recorded families; weights and private service launchers are not distributed here.

| Local model | Representation described in the paper |
|---|---|
| Ministral 3 3B | Q4_K_M |
| Gemma 4 E4B | Q4_K_M |
| DeepSeek-R1-0528-Qwen3-8B | BF16 |
| Qwen3.5 9B | Q8 |
| Qwen3.8 27B | BF16 |
| Qwen3.8 Flash Next | Q2 |

Cloud configurations are GPT-6 Luna, GPT-5.6 Terra, and GPT-6.1 Sol. Subscription products may hide weight revisions, templates, effective sampling, and internal execution behavior. Offline import/export keeps these comparisons separate from controlled local APIs.

`configs/endpoint.example.json` is an editable local-endpoint template; model IDs must match the actual server. `configs/model_metadata.example.json` lists provenance fields. Examples are not complete reproductions of private server settings. Record actual settings for each new run and leave unavailable fields unknown.

## Thirteen adapted specialists

| Task | Methods | Implementation |
|---|---|---|
| Leak | Physics residual, Extra Trees, residual SVM | `waterbench/methods.py` |
| Leak | LightGBM residual classifier | `scripts/run_supplementary_baselines.py` |
| Forecast | Seasonal naive, histogram gradient boosting, direct Lasso, fixed ensemble | `waterbench/methods.py` |
| Forecast | Chronos-2 median | `scripts/run_supplementary_baselines.py` |
| Cluster | Shape k-means, DTW average linkage, MSM k-medoids | `waterbench/methods.py`, `waterbench/clustering.py` |
| Cluster | KASBA | `scripts/run_supplementary_baselines.py` |

Core hyperparameters are in `configs/baselines.json`. The supplementary adapter records its own settings and provenance. These are adaptations to shared observations, not full reproductions of published systems.

Chronos-2 uses externally pretrained weights. Download is explicit (`--prepare-model`) and stores the resolved revision and checksums. The basic example and software tests do not download weights.
