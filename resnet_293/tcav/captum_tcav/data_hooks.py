from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from captum_tcav.common import (
        chunk_paths,
        module_name_in_model,
        predict_argmax_target as predict_target_index,
        resolve_layers,
    )
    from captum_tcav.concepts import (
        concept_dir,
        concept_npy_paths,
        ensure_random_concept,
        infer_target_frames,
        load_npy_as_tensor,
        make_iter,
        normalize_frames,
    )
    from captum_tcav.asvspoof5.config import Config, load_config
    from captum_tcav.asvspoof5.data import load_input, validate_config
    from captum_tcav.asvspoof5.modeling import (
        ResNet293WithSpoofLogit,
        load_model,
    )
else:
    from .common import (
        chunk_paths,
        module_name_in_model,
        predict_argmax_target as predict_target_index,
        resolve_layers,
    )
    from .concepts import (
        concept_dir,
        concept_npy_paths,
        ensure_random_concept,
        infer_target_frames,
        load_npy_as_tensor,
        make_iter,
        normalize_frames,
    )
    from .asvspoof5.config import Config, load_config
    from .asvspoof5.data import load_input, validate_config
    from .asvspoof5.modeling import (
        ResNet293WithSpoofLogit,
        load_model,
    )

__all__ = [
    "Config",
    "ResNet293WithSpoofLogit",
    "chunk_paths",
    "concept_dir",
    "concept_npy_paths",
    "ensure_random_concept",
    "infer_target_frames",
    "load_config",
    "load_input",
    "load_model",
    "load_npy_as_tensor",
    "make_iter",
    "normalize_frames",
    "predict_target_index",
    "resolve_layers",
    "validate_config",
]
