# Captum TCAV Layout

This package is being split into shared TCAV utilities plus task-specific experiment folders.

- Root-level shared files are intended for concept loading, layer resolution, and CSV export.
- `vox2/` contains the current speaker-identification TCAV pipeline for ReDimNet + VoxCeleb2 speaker head.
- `asvspoof5/` contains the spoof-system TCAV workflow and supporting notebooks.

Current state:

- The original root-level speaker files still exist for compatibility during the refactor.
- A mirrored `vox2/` package now holds the speaker-ID-specific config, data loading, modeling, and entrypoint logic.
- `asvspoof5/` now has its own config, manifest/subset loader, logistic-head model wrapper, and entrypoint.
- The concept bank is still expected at `<concept_root>/<concept_name>/*.npy`, including `random_0`.
- `random_0` remains a generated gaussian baseline concept shared across runs.
