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
    from captum_tcav.vox2.config import Config, load_config
    from captum_tcav.vox2.data import discover_speakers, load_input, validate_config
    from captum_tcav.vox2.modeling import (
        ReDimNetMelLogitsWrapper,
        SpeakerHead,
        load_model,
        load_speaker_head,
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
    from .vox2.config import Config, load_config
    from .vox2.data import discover_speakers, load_input, validate_config
    from .vox2.modeling import (
        ReDimNetMelLogitsWrapper,
        SpeakerHead,
        load_model,
        load_speaker_head,
    )

__all__ = [
    "Config",
    "ReDimNetMelLogitsWrapper",
    "SpeakerHead",
    "chunk_paths",
    "concept_dir",
    "concept_npy_paths",
    "discover_speakers",
    "ensure_random_concept",
    "infer_target_frames",
    "load_config",
    "load_input",
    "load_model",
    "load_npy_as_tensor",
    "load_speaker_head",
    "make_iter",
    "normalize_frames",
    "predict_target_index",
    "resolve_layers",
    "validate_config",
]
