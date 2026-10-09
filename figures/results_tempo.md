E1: measured − predicted tempo ratio, blend 1.2 s: mean -0.033, |max| 0.093; blend 0.6 s: mean -0.062, |max| 0.154 (n=20+20)

E2 (seeds: 3), learned policy, score per episode (max 2) / subtasks completed:

| robot lag | speed 1 | best uniform speed | score there | best expire | best adaptive | best arm shift | formula speed |
|---|---|---|---|---|---|---|---|
| 0.08 s | 0.61 / 0.97 | 1.80 | 1.05 / 0.93 | 1.08 @ 2 | 0.61 @ 2.5 | 0.69 @ 0.05 | 1.06 |
| 0.13 s | 0.52 / 0.92 | 1.60 | 0.88 / 0.87 | 0.91 @ 1.8 | 0.51 @ 2.5 | 0.66 @ 0.1 | 1.16 |
| 0.18 s | 0.42 / 0.84 | 1.30 | 0.63 / 0.85 | 0.63 @ 1.3 | 0.43 @ 1.5 | 0.60 @ 0.1 | 1.26 |

Measured speed at which executed/planned crosses 1 (linear interpolation of the mean curve) vs formula:

- lag 0.08 s: tempo at speed 1 = 0.856; crosses 1 at s = 1.22; formula 1.06
- lag 0.13 s: tempo at speed 1 = 0.775; crosses 1 at s = 1.40; formula 1.16
- lag 0.18 s: tempo at speed 1 = 0.708; crosses 1 at s = 1.53; formula 1.26

E3 (seeds: 3), mean over seeds:

| model | open-loop error, teleop val | open-loop error, scripted val | planned tempo | closed-loop success @1 | closed-loop score @1 | calibrated speed | score @ calibrated | best score in sweep (speed) |
|---|---|---|---|---|---|---|---|---|
| base_2k | 0.0493 | 0.0370 | 0.90 | 0.50 | 0.28 | 1.18 | 0.39 | 0.57 (1.8) |
| base_4k | 0.0422 | 0.0256 | 0.91 | 0.92 | 0.55 | 1.16 | 0.70 | 0.92 (1.8) |
| base | 0.0376 | 0.0178 | 0.93 | 0.97 | 0.61 | 1.14 | 0.74 | 1.06 (1.8) |
| ft_25 | 0.0318 | 0.0266 | 0.79 | 0.68 | 0.33 | 1.35 | 0.56 | 0.61 (1.6) |
| ft_50 | 0.0313 | 0.0278 | 0.77 | 0.63 | 0.30 | 1.38 | 0.52 | 0.57 (1.6) |
| ft_100 | 0.0280 | 0.0382 | 0.62 | 0.41 | 0.15 | 1.72 | 0.51 | 0.53 (1.8) |

Share of the sweep gain (best − speed 1) recovered by the calibrated speed:
- base_2k: 37 %
- base_4k: 41 %
- base: 29 %
- ft_25: 84 %
- ft_50: 82 %
- ft_100: 93 %

Spearman ρ with closed-loop score @1 over all checkpoints × seeds: err_teleop_val +0.40, err_scripted_val -0.97, planned_tempo +0.85

E1b: blend on/off and blend life on the task (seed 0; score / subtasks / executed÷planned speed):

| planner | robot lag | no blend | life 0.4 s | life 0.8 s | life 1.2 s (served) |
|---|---|---|---|---|---|
| oracle | 0.08 s | 0.03 / 0.16 / 0.50 | 0.34 / 0.80 / 0.71 | 0.45 / 0.92 / 0.80 | 0.49 / 0.94 / 0.82 |
| oracle | 0.13 s | 0.00 / 0.00 / 0.31 | 0.12 / 0.44 / 0.59 | 0.30 / 0.73 / 0.70 | 0.36 / 0.83 / 0.73 |
| oracle | 0.18 s | 0.00 / 0.00 / 0.15 | 0.02 / 0.14 / 0.52 | 0.16 / 0.50 / 0.63 | 0.23 / 0.63 / 0.66 |
| learned | 0.08 s | 0.31 / 0.55 / 0.78 | 0.43 / 0.75 / 0.81 | 0.58 / 0.95 / 0.85 | 0.61 / 0.97 / 0.86 |
| learned | 0.13 s | 0.16 / 0.35 / 0.63 | 0.30 / 0.62 / 0.70 | 0.46 / 0.85 / 0.75 | 0.52 / 0.92 / 0.78 |
| learned | 0.18 s | 0.08 / 0.23 / 0.53 | 0.20 / 0.48 / 0.62 | 0.36 / 0.76 / 0.69 | 0.43 / 0.85 / 0.71 |

E2b (3 seeds): rows flagged as a gripper event

| chunks | row-jitter detector | crossing detector |
|---|---|---|
| demo_chunks | 0.023 | 0.023 |
| policy_plans | 0.599 | 0.028 |

| robot lag | detector | 1.5 | 2.0 | 2.5 | 3.0 | rows kept at 1x |
|---|---|---|---|---|---|---|
| 0.08 s | jitter | 0.61 / 0.97 | 0.61 / 0.97 | 0.61 / 0.97 | 0.61 / 0.97 | 0.99 |
| 0.08 s | crossing | 0.81 / 1.00 | 0.91 / 0.99 | 0.95 / 0.96 | 0.92 / 0.90 | 0.48 |
| 0.13 s | jitter | 0.51 / 0.92 | 0.51 / 0.92 | 0.52 / 0.93 | 0.52 / 0.92 | 0.99 |
| 0.13 s | crossing | 0.73 / 0.99 | 0.84 / 0.99 | 0.88 / 0.97 | 0.91 / 0.95 | 0.49 |
| 0.18 s | jitter | 0.42 / 0.84 | 0.42 / 0.84 | 0.42 / 0.85 | 0.42 / 0.84 | 0.99 |
| 0.18 s | crossing | 0.64 / 0.98 | 0.73 / 0.94 | 0.76 / 0.90 | 0.77 / 0.87 | 0.49 |

| robot lag | speed | clamp on: executed÷planned / score | clamp off |
|---|---|---|---|
| 0.08 s | 1.0 | 0.855 / 0.61 | 0.847 / 0.63 |
| 0.08 s | 1.3 | 1.056 / 0.88 | 1.051 / 0.89 |
| 0.13 s | 1.0 | 0.773 / 0.52 | 0.774 / 0.53 |
| 0.13 s | 1.3 | 0.945 / 0.80 | 0.945 / 0.80 |
| 0.18 s | 1.0 | 0.712 / 0.42 | 0.714 / 0.43 |
| 0.18 s | 1.3 | 0.872 / 0.62 | 0.876 / 0.59 |
