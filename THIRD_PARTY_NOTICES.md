# Network sources and attribution

Only network models are reused; challenge event labels and published challenge results are not included.

| Network | Source | Recorded attribution and terms |
|---|---|---|
| Grid | `waterbench.data.grid` | Project-generated 4 × 4 network |
| C-Town | [BATADAL](https://www.batadal.net/data.html), [CTOWN.INP](https://www.batadal.net/data/CTOWN.INP) | BATADAL public network; acquisition metadata records CC BY 4.0 |
| L-Town | [BattLeDIM dataset, Zenodo 4017659](https://doi.org/10.5281/zenodo.4017659) | Vrachimis, Eliades, Taormina, Ostfeld, Kapelan, and collaborators; dataset metadata records CC BY 4.0 |

Original INP files are unchanged. Simulation copies are configured separately. `data/networks/manifest.json` records source versions, URLs, and checksums. The L-Town README and Zenodo metadata retain additional attribution. BattLeDIM software has separate licensing and is not bundled here.

Dependencies retain their respective licenses and are installed separately, rather than vendored. Pretrained weights remain subject to their publishers' terms and are not included.

A repository-wide license for original project code has not yet been assigned by the authors. These notices do not grant a new license over that code or any model weights.
