#!/usr/bin/env python3
"""Run the full TCAV pipeline: activations -> CAVs -> scores -> significance -> heatmap."""
import sys
import yaml
import json
import csv
import os
import io
import tarfile
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# Resolve xai_root (consistent with all xai/ scripts)
XAI_ROOT = Path(__file__).parent.parent

from xai.experiments.runner import ExperimentRunner
from xai.tcav.hooks import ActivationExtractor, LAYER_NAMES, pool_activation
from xai.tcav.cav import train_cav
from xai.tcav.scorer import compute_tcav_score
from xai.tcav.stats import test_significance
from xai.tcav.viz import plot_emergence_heatmap
from xai.tcav.concepts import load_concept_clips, sample_random_baseline


def load_dev_clips(protocol_path: str, tar_dir: str, label: str,
                   n_clips: int, seed: int = 42) -> torch.Tensor:
    """
    Load n_clips from ASVspoof5 dev partition for a specific label.

    protocol_path: path to ASVspoof5.dev.track_1.tsv
    tar_dir:       path to flac_D tar directory
    label:         'bonafide' or 'spoof'
    n_clips:       number to sample
    seed:          for deterministic sampling

    Returns: (n_clips, 64600) tensor
    """
    # Read protocol, filter by label
    rows = []
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 9 and parts[8] == label:
                rows.append({'audio_id': parts[1], 'speaker_id': parts[0]})

    rng = np.random.RandomState(seed)
    selected = rng.choice(rows, size=min(n_clips, len(rows)), replace=False)
    selected = sorted(selected, key=lambda r: r['audio_id'])

    needed = {r['audio_id'] for r in selected}
    loaded = {}
    fixed_len = 64600  # RawNet2 input length

    for tar_path in sorted(Path(tar_dir).glob('*.tar')):
        with tarfile.open(str(tar_path), 'r') as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = os.path.basename(member.name)
                audio_id = name.replace('.flac', '')
                if audio_id in needed:
                    data = tar.extractfile(member).read()
                    audio, sr = sf.read(io.BytesIO(data))
                    loaded[audio_id] = torch.tensor(audio, dtype=torch.float32)
                    if len(loaded) == len(needed):
                        break
        if len(loaded) == len(needed):
            break

    # Pad/truncate to fixed_len, stack
    tensors = []
    for r in selected:
        aid = r['audio_id']
        if aid in loaded:
            w = loaded[aid]
            t = torch.zeros(fixed_len)
            T = min(len(w), fixed_len)
            t[:T] = w[:T]
            tensors.append(t)

    return torch.stack(tensors)


def run_tcav_pipeline(config_path: str = 'config/tcav_config.yaml'):
    cfg = yaml.safe_load(open(XAI_ROOT / config_path))

    tcav_cfg = cfg['tcav']
    layers = tcav_cfg['layers']
    n_clips = tcav_cfg['n_clips_per_concept']
    n_baselines = tcav_cfg['n_random_baselines']
    n_dev = tcav_cfg['n_dev_per_class']
    n_pairs = tcav_cfg['bonferroni_n_pairs']
    device = tcav_cfg['device']
    batch_size = tcav_cfg['batch_size']

    # Create results directories
    results_dir = XAI_ROOT / cfg['results']['output_dir']
    for subdir in ['activations', 'cavs', 'scores', 'figures']:
        (results_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Initialize ExperimentRunner for seeding and artifacts
    runner = ExperimentRunner(config_path)
    with runner.run() as exp_dir:
        model = runner.load_model()

        # ----- Step 1: Extract concept activations -----
        print("\n=== Step 1: Extracting concept activations ===")
        all_concepts = list(cfg['concepts'].keys())  # includes negative_control

        concept_activations = {}  # {concept: {layer: np.ndarray}}
        all_concept_audio_ids = set()

        for concept_name in tqdm(all_concepts, desc='Concepts'):
            manifest_path = str(XAI_ROOT / cfg['concepts'][concept_name])
            clips = load_concept_clips(manifest_path, cfg['data']['tar_dir_train'],
                                       n_clips=n_clips, seed=42)
            # Extract activations at all layers
            acts = {}
            with ActivationExtractor(model, LAYER_NAMES, retain_grad=False) as ext:
                model.eval()
                with torch.no_grad():
                    # Process in batches
                    for i in range(0, len(clips), batch_size):
                        batch = clips[i:i+batch_size].squeeze(1).to(device)
                        # Pad to 64600 if needed
                        if batch.shape[-1] < 64600:
                            pad = torch.zeros(batch.shape[0], 64600 - batch.shape[-1], device=device)
                            batch = torch.cat([batch, pad], dim=-1)
                        _ = model(batch)
                        for layer in layers:
                            act = ext.get(layer)
                            pooled = pool_activation(act).numpy()
                            if layer not in acts:
                                acts[layer] = []
                            acts[layer].append(pooled)
                        ext.activations.clear()  # clear for next batch

            concept_activations[concept_name] = {
                layer: np.vstack(acts[layer]) for layer in layers
            }

            # Save activations cache
            for layer in layers:
                np.savez_compressed(
                    str(results_dir / 'activations' / f'{concept_name}_{layer}.npz'),
                    activations=concept_activations[concept_name][layer]
                )

            # Collect audio_ids for exclusion in random baselines
            with open(str(XAI_ROOT / cfg['concepts'][concept_name]), 'r') as f:
                for row in csv.DictReader(f):
                    all_concept_audio_ids.add(row['audio_id'])

        # ----- Step 2: Extract random baseline activations -----
        print("\n=== Step 2: Extracting random baseline activations ===")
        random_activations = []  # list of {layer: np.ndarray}

        for baseline_idx in tqdm(range(n_baselines), desc='Random baselines'):
            clips = sample_random_baseline(
                manifest_dir=str(XAI_ROOT / 'data'),
                tar_dir=cfg['data']['tar_dir_train'],
                n_clips=n_clips, seed=baseline_idx,
                exclude_audio_ids=all_concept_audio_ids
            )
            acts = {}
            with ActivationExtractor(model, LAYER_NAMES, retain_grad=False) as ext:
                model.eval()
                with torch.no_grad():
                    for i in range(0, len(clips), batch_size):
                        batch = clips[i:i+batch_size].squeeze(1).to(device)
                        if batch.shape[-1] < 64600:
                            pad = torch.zeros(batch.shape[0], 64600 - batch.shape[-1], device=device)
                            batch = torch.cat([batch, pad], dim=-1)
                        _ = model(batch)
                        for layer in layers:
                            act = ext.get(layer)
                            pooled = pool_activation(act).numpy()
                            if layer not in acts:
                                acts[layer] = []
                            acts[layer].append(pooled)
                        ext.activations.clear()

            random_activations.append({
                layer: np.vstack(acts[layer]) for layer in layers
            })

        # ----- Step 3: Train CAVs -----
        print("\n=== Step 3: Training CAVs ===")
        cavs = {}  # {concept: {layer: {'cav': ndarray, 'accuracy': float, 'scaler': StandardScaler}}}

        for concept_name in tqdm(all_concepts, desc='CAV training'):
            cavs[concept_name] = {}
            for layer in layers:
                c_acts = concept_activations[concept_name][layer]
                r_acts = random_activations[0][layer]  # first random baseline for CAV training
                result = train_cav(c_acts, r_acts, random_state=42)
                cavs[concept_name][layer] = result
                # Save CAV
                np.savez(
                    str(results_dir / 'cavs' / f'{concept_name}_{layer}.npz'),
                    cav=result['cav'],
                    accuracy=np.array([result['accuracy']])
                )

        # Print CAV accuracy summary
        print("\nCAV Accuracies:")
        print(f"{'Concept':<25} " + " ".join(f"{l:>12}" for l in layers))
        for concept_name in all_concepts:
            accs = [f"{cavs[concept_name][l]['accuracy']:.3f}" for l in layers]
            print(f"{concept_name:<25} " + " ".join(f"{a:>12}" for a in accs))

        # ----- Step 4: Load dev clips for TCAV scoring -----
        print(f"\n=== Step 4: Loading {n_dev} bonafide + {n_dev} spoof dev clips ===")
        bonafide_dev = load_dev_clips(
            cfg['data']['dev_protocol'], cfg['data']['tar_dir_dev'],
            'bonafide', n_clips=n_dev, seed=42
        )
        spoof_dev = load_dev_clips(
            cfg['data']['dev_protocol'], cfg['data']['tar_dir_dev'],
            'spoof', n_clips=n_dev, seed=42
        )

        # ----- Step 5: Compute TCAV scores -----
        print("\n=== Step 5: Computing TCAV scores ===")
        scores = {}  # {concept: {layer: {'bonafide': dict, 'spoof': dict}}}

        for concept_name in tqdm(all_concepts, desc='TCAV scoring'):
            scores[concept_name] = {}
            for layer in layers:
                cav_vec = cavs[concept_name][layer]['cav']
                scaler = cavs[concept_name][layer]['scaler']

                bf_result = compute_tcav_score(
                    model, bonafide_dev, target_class=1,
                    layer_name=layer, cav=cav_vec, scaler=scaler,
                    device=device, batch_size=batch_size
                )
                sp_result = compute_tcav_score(
                    model, spoof_dev, target_class=0,
                    layer_name=layer, cav=cav_vec, scaler=scaler,
                    device=device, batch_size=batch_size
                )
                scores[concept_name][layer] = {
                    'bonafide': bf_result,
                    'spoof': sp_result,
                }

        # Also compute random baseline TCAV scores for significance testing
        print("\n=== Step 5b: Computing random baseline TCAV scores ===")
        random_tcav_scores = {}  # {layer: [score_per_baseline]}
        for layer in layers:
            random_tcav_scores[layer] = []
            for baseline_idx in tqdm(range(n_baselines), desc=f'Random baselines ({layer})', leave=False):
                r_acts = random_activations[baseline_idx][layer]
                # Train CAV for this random baseline vs another random baseline
                other_idx = (baseline_idx + 1) % n_baselines
                r_cav_result = train_cav(r_acts, random_activations[other_idx][layer],
                                         random_state=baseline_idx)
                # Compute TCAV score
                bf_r = compute_tcav_score(
                    model, bonafide_dev, target_class=1,
                    layer_name=layer, cav=r_cav_result['cav'],
                    scaler=r_cav_result['scaler'],
                    device=device, batch_size=batch_size
                )
                random_tcav_scores[layer].append(bf_r['tcav_score'])

        # ----- Step 6: Significance testing -----
        print("\n=== Step 6: Significance testing ===")
        sig_results = {}  # {(concept, layer): dict}

        for concept_name in all_concepts:
            for layer in layers:
                real_score = scores[concept_name][layer]['bonafide']['tcav_score']
                random_baseline_scores = [[random_tcav_scores[layer][i]]
                                          for i in range(n_baselines)]
                sig = test_significance(
                    [real_score], random_baseline_scores, n_pairs=n_pairs
                )
                sig_results[(concept_name, layer)] = sig

        # Print significance summary
        print("\nTCAV Significance Results (bonafide class):")
        print(f"{'Concept':<25} {'Layer':<15} {'Score':>8} {'p_corr':>8} {'Sig?':>5}")
        for concept_name in all_concepts:
            for layer in layers:
                s = scores[concept_name][layer]['bonafide']['tcav_score']
                sig = sig_results[(concept_name, layer)]
                star = '*' if sig['significant'] else ''
                print(f"{concept_name:<25} {layer:<15} {s:>8.3f} {sig['pval_corrected']:>8.4f} {star:>5}")

        # ----- Step 7: Generate heatmap -----
        print("\n=== Step 7: Generating emergence heatmap ===")
        accuracy_matrix = np.zeros((len(layers), len(all_concepts)))
        for i, layer in enumerate(layers):
            for j, concept_name in enumerate(all_concepts):
                accuracy_matrix[i, j] = cavs[concept_name][layer]['accuracy']

        heatmap_path = str(results_dir / 'figures' / 'emergence_heatmap.png')
        plot_emergence_heatmap(
            accuracy_matrix, layers, all_concepts, heatmap_path
        )
        print(f"Heatmap saved to: {heatmap_path}")

        # ----- Save all results -----
        results_summary = {
            'concepts': all_concepts,
            'layers': layers,
            'cav_accuracies': {
                c: {l: cavs[c][l]['accuracy'] for l in layers}
                for c in all_concepts
            },
            'tcav_scores': {
                c: {l: {
                    'bonafide': scores[c][l]['bonafide']['tcav_score'],
                    'spoof': scores[c][l]['spoof']['tcav_score'],
                } for l in layers}
                for c in all_concepts
            },
            'significance': {
                f"{c}_{l}": {
                    'pval': sig_results[(c, l)]['pval'],
                    'pval_corrected': sig_results[(c, l)]['pval_corrected'],
                    'significant': sig_results[(c, l)]['significant'],
                    'ci_95': list(sig_results[(c, l)]['ci_95']),
                    'mean_score': sig_results[(c, l)]['mean_score'],
                } for c in all_concepts for l in layers
            },
        }

        results_path = results_dir / 'scores' / 'tcav_results.json'
        with open(str(results_path), 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"\nResults saved to: {results_path}")

        # Also save via ExperimentRunner
        runner.save_results(results_summary)

    print("\n=== TCAV Pipeline Complete ===")
    return 0


if __name__ == '__main__':
    config = sys.argv[1] if len(sys.argv) > 1 else 'config/tcav_config.yaml'
    sys.exit(run_tcav_pipeline(config))
