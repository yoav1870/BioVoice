from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from tcav_core.config import ExperimentConfig
from tcav_core.utils import output_path


def save_outputs(
    *,
    output_dir: Path,
    config: ExperimentConfig,
    rows_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    effect_summary_df: pd.DataFrame,
    cav_acc_df: pd.DataFrame,
    run_metadata: dict,
) -> None:
    if not rows_df.empty and not cav_acc_df.empty:
        rows_df = rows_df.merge(
            cav_acc_df,
            on=["layer_key", "concept name", "random concept", "layer name"],
            how="left",
        )
    elif "cav acc" not in rows_df.columns:
        rows_df["cav acc"] = float("nan")

    rows_csv = output_path(output_dir, config.output_tag, "tcav_rows.csv")
    summary_csv = output_path(output_dir, config.output_tag, "tcav_summary.csv")
    effect_summary_csv = output_path(
        output_dir,
        config.output_tag,
        "tcav_effect_summary.csv",
    )
    cav_csv = output_path(output_dir, config.output_tag, "cav_acc.csv")
    config_json = output_path(output_dir, config.output_tag, "run_config.json")

    rows_df.to_csv(rows_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    effect_summary_df.to_csv(effect_summary_csv, index=False)
    cav_acc_df.to_csv(cav_csv, index=False)

    metadata = {
        **asdict(config),
        **run_metadata,
    }
    config_json.write_text(json.dumps(metadata, indent=2))

    print("\nSaved:")
    print(f" - {rows_csv}")
    print(f" - {summary_csv}")
    print(f" - {effect_summary_csv}")
    print(f" - {cav_csv}")
    print(f" - {config_json}")
