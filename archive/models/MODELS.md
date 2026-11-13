# Trained models (saved artifacts)

_Generated 2026-06-15 13:54 by `train_and_save.py` from `paper/data/measurements_generalized.csv`._

| model | file | type | target | params | held-out accuracy (MRE %) |
|---|---|---|---|---|---|
| LINEAR | `models/linear_time.joblib` | sklearn | time | - | size-extrapolation=435.6, corpus-transfer=778.6 |
| RIDGE | `models/ridge_time.joblib` | sklearn | time | - | size-extrapolation=433.6, corpus-transfer=773.0 |
| GBM | `models/gbm_time.joblib` | sklearn | time | - | size-extrapolation=155.1, corpus-transfer=245.1 |
| MLP | `models/mlp_time.joblib` | sklearn | time | - | size-extrapolation=52.7, corpus-transfer=669.3 |
| FT-Transformer | `models/ftt_time.pt` | torch | time | 17569 | size-extrapolation=33.7, corpus-transfer=380.8 |
| LINEAR | `models/linear_energy.joblib` | sklearn | energy | - | size-extrapolation=278.2, corpus-transfer=482.9 |
| RIDGE | `models/ridge_energy.joblib` | sklearn | energy | - | size-extrapolation=277.1, corpus-transfer=479.2 |
| GBM | `models/gbm_energy.joblib` | sklearn | energy | - | size-extrapolation=153.0, corpus-transfer=330.1 |
| MLP | `models/mlp_energy.joblib` | sklearn | energy | - | size-extrapolation=110.9, corpus-transfer=3271.8 |
| FT-Transformer | `models/ftt_energy.pt` | torch | energy | 17569 | size-extrapolation=100.8, corpus-transfer=362.1 |
| Kalman-trained FT-Transformer | `models/kalman_transformer_energy.pt` | torch | energy(total,cold) | 10081 | cold_baseline%=127.9, cold_kalman%=117.5 |
