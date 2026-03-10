from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT_DIR = Path("/home/SpeakerRec/BioVoice")


@dataclass
class ExperimentConfig:
    dataset_csv_path: Path = (
        ROOT_DIR
        / "redimnet/grad_cam/2.0/output/speaker_similarity_ranking_vox2_10_20_ids.csv"
    )
    output_dir: Path = ROOT_DIR / "redimnet/tcav/output"
    output_tag: str = "tcav-v2-quality"

    concept_root: Path = ROOT_DIR / "concept/final_concepts"
    excluded_concept_names: tuple[str, ...] = ("random",)
    min_concept_files_warning: int = 120

    head_path: Path = ROOT_DIR / "data/heads/redim_speaker_head_vox2_10_20.pt"
    target_layer_keys: tuple[str, ...] = ("stage4",)

    target_frames_override: Optional[int] = None
    frame_crop_mode: str = "center"
    frame_pad_mode: str = "center"

    cav_save_path: Optional[Path] = ROOT_DIR / "redimnet/tcav/cav_cache"
    model_id: str = "redimnet_vox2"

    negative_mode: str = "shuffled_real"
    random_concept_repeats: int = 8
    random_source_csv_path: Optional[Path] = None
    random_source_max_files: Optional[int] = None
    shuffle_time: bool = True
    shuffle_freq: bool = True

    path_column: str = "path"
    label_column: str = "speaker"

    seed: int = 1337
    concept_samples: int = 360
    augment_on_oversample: bool = True
    concept_augment_time_shift_max: int = 8
    concept_augment_freq_shift_max: int = 1
    concept_augment_gain_min: float = 0.97
    concept_augment_gain_max: float = 1.03
    concept_augment_noise_std: float = 0.005

    random_samples: int = 360
    batch_size: int = 4
    data_loader_num_workers: int = 4
    data_loader_pin_memory: bool = True

    force_av_load_cpu: bool = True
    force_train_cavs: bool = False
    cav_test_split_ratio: float = 0.25
    cav_quality_warn_threshold: float = 0.6
    tcav_device: str = "auto"

    target_mode: str = "predicted"
    fixed_target_idx: Optional[int] = None
    effect_epsilon: float = 1e-8


CONFIG = ExperimentConfig()
