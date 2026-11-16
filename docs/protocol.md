# Protocol and scoring

## Observations and information access

WaterBench generates independent scenarios with disjoint train, validation, and test units. Configuration controls seeds, counts, networks, noise conditions, leak fractions, sensor count, and decision hours. Training windows stay within their parent realization. Forecast and clustering can share parent demand realizations, so they must not be pooled as independent tasks.

Leak episodes use WNTR pressure-dependent simulation. Hydraulic steps are 900 seconds with hourly observations. Incomplete, nonfinite, nonconvergent, or materially negative-pressure scenarios are rejected and deterministically replaced within their scenario stratum. Generation logs retain accepted and rejected candidates.

Each method receives the same held-out observation object. LLMs are zero-shot. Supervised specialists additionally use labeled training data; the physics residual detector calibrates its threshold on nominal validation episodes. Shape clustering is observation-only. This compares trained specialists with zero-shot generalists rather than equal total training information.

## Task contracts

- **Leak:** 24-hour episodes, zero or one leak, five independent causal decisions at hours 8, 12, 16, 20, and 23. Output is a Boolean alarm, finite score in [0, 1], and up to three distinct known zones. An alarm requires a ranking; no alarm requires an empty ranking.
- **Forecast:** 168 past hourly intended-demand values; exactly 24 finite nonnegative predictions in L/s. The target is synthetic intended demand, not delivered pressure-dependent demand or inlet flow.
- **Clustering:** twelve normalized daily profiles and known k = 3. Each consumer appears once; integer labels 0 through k−1 form exactly k nonempty groups.

Direct conditions use no external tools, repair, silent truncation, or wrapper retries. Invalid outputs remain terminal attempts. Optional tool-mode library support is separate from the paper's direct conditions.

## Metrics

**Validity** is complete valid case-replicates divided by all case-replicates. A leak episode is complete only if all five decisions are valid. Failed and missing attempts remain unsuccessful for unconditional metrics.

**Leak F1 and recall** use complete episodes and event-level counts. False-alarm sequences count as false positives; decision windows are not independent observations. Unconditional top-1 localization divides correctly detected and localized leak events by all leak events, including failures. Conditional detection delay must accompany event coverage.

**Forecast MAE** is in L/s. MASE uses network-specific training-only lag-24 absolute differences. WAPE is calculated within each valid case and then averaged equally across cases. Undefined scales stay undefined. Cross-network MAE changes with demand scale.

**Clustering ARI and NMI** use complete partitions, invariant to label permutation. Exact synthetic shape recovery does not establish performance on real customer categories.

**All-case success** requires detection and correct first-ranked localization for leak events or no false alarm for normal episodes, a valid forecast with MASE ≤ 1, or exact clustering recovery. Conditional metrics and unconditional success answer different questions.

## Repetitions, freeze, and efficiency

Repeated executions of one scenario are not independent scenarios. Bootstrap resampling retains repetitions within each independent dataset/scenario block. Conditional paired comparisons use complete units and must accompany all-case validity and success. Pointwise intervals do not control family-wise error.

Choose settings on training/validation data, then freeze the test protocol. `scripts/freeze_final.py` verifies pilot evidence, source identity, baseline seeds, endpoint settings, and declarations for unavailable metadata. Changed prompts, inference code, settings, or information access require a new freeze.

Local API HTTP latency and cloud-product wall time have different boundaries. Client CPU time excludes remote inference. Fit costs are separate from prediction time. Unknown usage, energy, memory, parameter counts, and financial costs stay unknown rather than inferred from product names or subscriptions.
