"""Test configuration for TCAV - single speaker test run."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

ROOT_DIR = Path("/home/SpeakerRec/BioVoice")


@dataclass
class TestExperimentConfig:
    """Lightweight config for quick testing on single speaker."""

    # ===== TEST DATASET: Single speaker only =====
    dataset_csv_path: Path = (
        ROOT_DIR
        / "redimnet/grad_cam/2.0/output/speaker_ranking_test_single_speaker.csv"
    )
    output_dir: Path = ROOT_DIR / "redimnet/tcav/output"
    output_tag: str = "tcav_test_single_speaker_id00012"

    concept_root: Path = ROOT_DIR / "concept/final_concepts"
    excluded_concept_names: tuple[str, ...] = ("random",)
    min_concept_files_warning: int = 120

    head_type: str = "speaker"
    head_path: Path = ROOT_DIR / "data/heads/redim_speaker_head_vox2_10_20.pt"
    spoof_logreg_path: Optional[Path] = None
    spoof_scaler_path: Optional[Path] = None
    spoof_l2_norm_emb: bool = False
    target_layer_keys: tuple[str, ...] = ("stage4",)  # Test on stage4 only

    target_frames_override: Optional[int] = None
    frame_crop_mode: str = "center"
    frame_pad_mode: str = "center"

    cav_save_path: Optional[Path] = (
        ROOT_DIR / "redimnet/tcav/cav_cache_test_single_speaker"
    )
    model_id: str = "redimnet_test_single_speaker"

    negative_mode: str = "shuffled_real"
    random_concept_repeats: int = 1
    random_source_csv_path: Optional[Path] = None
    random_source_max_files: Optional[int] = None
    shuffle_time: bool = True
    shuffle_freq: bool = True

    path_column: str = "path"
    label_column: str = "speaker"

    seed: int = 1337
    # ===== REDUCED FOR FASTER TESTING =====
    concept_samples: int = 50  # was 300 - small for quick test
    augment_on_oversample: bool = False
    concept_augment_time_shift_max: int = 8
    concept_augment_freq_shift_max: int = 1
    concept_augment_gain_min: float = 0.97
    concept_augment_gain_max: float = 1.03
    concept_augment_noise_std: float = 0.005

    random_samples: int = 50  # was 300 - small for quick test
    batch_size: int = 8
    data_loader_num_workers: int = 2
    data_loader_pin_memory: bool = True

    force_av_load_cpu: bool = True
    force_train_cavs: bool = True
    cav_test_split_ratio: float = 0.1
    cav_quality_warn_threshold: float = 0.6
    tcav_device: str = "cpu"  # Use CPU to avoid CUDA OOM

    target_mode: str = "predicted"
    fixed_target_idx: Optional[int] = None
    effect_epsilon: float = 1e-8


# Use this config for testing
TEST_CONFIG = TestExperimentConfig()
