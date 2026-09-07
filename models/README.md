# Models

Everything the pipeline loads, in one place.

| file | params | role |
|---|---|---|
| `photon_corrector.pt` | 1,361,187 | the shipped dose corrector, baked by `submission/bake_model.py` |
| `tiny_fluence_tile_mean_epoch1.pt` | 39,081 | frozen fluence prior |
| `tiny_lateral_heterogeneity_tile_mean_fluence_e1.pt` | 48 | frozen lateral-heterogeneity prior, trained on top of the fluence prior |

The two priors are **source inputs** to corrector training, not runtime
dependencies: `photon_corrector.pt` embeds both state dictionaries and their
configurations, so evaluation and serving never read them. They are kept so the
corrector can be retrained from the same starting point.

`serve.py` reads this directory (`MODEL_DIR`, defaulting here) and picks a file
with `MODEL_NAME` / `MODEL_NAME_CT` / `MODEL_NAME_MR`.

Not kept here: the MR → synthetic CT TorchScript bundle, which is 390 MB and
lives on Hugging Face — see the root `README.md`.
