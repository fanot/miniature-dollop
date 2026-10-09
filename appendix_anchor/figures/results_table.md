Seeds: 1 · episodes per split per seed: 200

| method | in-distribution | new positions | new colour×shape | swapped places | new table colour | air-grasp, target removed | wrong object, swapped |
|---|---|---|---|---|---|---|---|
| BC (FiLM-CNN) | 0.61 ± 0.00 | 0.12 ± 0.00 | 0.08 ± 0.00 | 0.06 ± 0.00 | 0.11 ± 0.00 | 0.00 ± 0.00 | 0.38 ± 0.00 |
| BC + pointing co-train | 0.72 ± 0.00 | 0.28 ± 0.00 | 0.17 ± 0.00 | 0.20 ± 0.00 | 0.18 ± 0.00 | 0.00 ± 0.00 | 0.28 ± 0.00 |
| Guidance (JPM-like) | 0.72 ± 0.00 | 0.33 ± 0.00 | 0.22 ± 0.00 | 0.24 ± 0.00 | 0.27 ± 0.00 | 0.00 ± 0.00 | 0.07 ± 0.00 |
| Point injection | 0.71 ± 0.00 | 0.50 ± 0.00 | 0.42 ± 0.00 | 0.43 ± 0.00 | 0.19 ± 0.00 | 0.00 ± 0.00 | 0.03 ± 0.00 |
| Anchor frame (ours) | 0.94 ± 0.00 | 0.96 ± 0.00 | 0.91 ± 0.00 | 0.94 ± 0.00 | 0.41 ± 0.00 | 0.02 ± 0.00 | 0.01 ± 0.00 |
| AF + global image | 0.92 ± 0.00 | 0.96 ± 0.00 | 0.92 ± 0.00 | 0.94 ± 0.00 | 0.40 ± 0.00 | 0.02 ± 0.00 | 0.01 ± 0.00 |
| AF w/o local crop | 0.05 ± 0.00 | 0.07 ± 0.00 | 0.04 ± 0.00 | 0.06 ± 0.00 | 0.05 ± 0.00 | 0.07 ± 0.00 | 0.01 ± 0.00 |

| method (probe) | pointing acc., new positions | success w/ oracle point, new positions | success w/ oracle point, new colour×shape | follows moved anchor | follows language |
|---|---|---|---|---|---|
| Guidance (JPM-like) | 1.00 | 0.32 ± 0.00 | 0.23 ± 0.00 | 0.14 ± 0.00 | 0.07 ± 0.00 |
| Point injection | 1.00 | 0.49 ± 0.00 | 0.42 ± 0.00 | 0.39 ± 0.00 | 0.01 ± 0.00 |
| Anchor frame (ours) | 1.00 | 0.94 ± 0.00 | 0.91 ± 0.00 | 0.73 ± 0.00 | 0.00 ± 0.00 |
| AF + global image | 1.00 | 0.95 ± 0.00 | 0.93 ± 0.00 | 0.74 ± 0.00 | 0.00 ± 0.00 |
| AF w/o local crop | 1.00 | 0.06 ± 0.00 | 0.05 ± 0.00 | 0.56 ± 0.00 | 0.00 ± 0.00 |

| method | σ=0 | σ=0.02 | σ=0.04 | σ=0.06 | σ=0.08 |
|---|---|---|---|---|---|
| Point injection | 0.49 | 0.50 | 0.48 | 0.49 | 0.41 |
| Anchor frame (ours) | 0.94 | 0.85 | 0.58 | 0.37 | 0.26 |
| AF + global image | 0.95 | 0.91 | 0.56 | 0.36 | 0.27 |
| AF w/o local crop | 0.06 | 0.04 | 0.04 | 0.04 | 0.04 |
| Guidance (JPM-like) | 0.32 | 0.33 | 0.34 | 0.31 | 0.29 |
