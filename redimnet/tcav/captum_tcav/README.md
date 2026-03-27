# Captum TCAV Docstring Skeleton

This folder keeps a simple Captum-style top-level TCAV flow while reading paths and names from `config.py`.

- Set `concept_root`, `concept_names`, `input_path`, `target_index`, and model settings in `config.py`.
- The expected concept layout is `<concept_root>/<concept_name>/*.npy`, such as `concepts/concept1/*.npy` and `concepts/random_0/*.npy`.
- The real WAV you want to explain loads from `input_path`.
- `random_0` is a single gaussian noise concept that is generated and saved automatically if it does not exist yet.
- The default model config is `IDRnD/ReDimNet` with `ReDimNet`, `b5`, `ptn`, and `vox2`.
- The entrypoint compares every configured concept against the same `random_0` baseline in one `interpret(...)` call.
- After a successful run, the entrypoint writes a CSV to `captum_tcav/output/tcav_scores.csv` and prints `CSV path: ...`.
- The CSV contains exactly these columns: `input_path`, `magnitude`, `sign_count`, `concept_name`, `speaker_id`.
- The full TCAV run is intended for the remote server after the expected data is present; local work here is only the skeleton and config contract.
- Future spectrogram and `redimnet` work should layer onto this generic surface later.
