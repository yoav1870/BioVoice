#!/usr/bin/env python3
"""Per-system TCAV analysis: stratify dev clips by synthesis system, compute TCAV scores per system.

Usage:
    cd /home/SpeakerRec/BioVoice
    .venv/bin/python xai/scripts/run_per_system_tcav.py [--config xai/config/per_system_config.yaml]
"""
import sys
import yaml
import json
import io
import os
import tarfile
import argparse
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

XAI_ROOT = Path(__file__).parent.parent

from xai.experiments.runner import ExperimentRunner
from xai.tcav.hooks import ActivationExtractor, LAYER_NAMES, pool_activation
from xai.tcav.cav import train_cav
from xai.tcav.scorer import compute_tcav_score
from xai.tcav.stats import test_significance, fdr_bh_correction
from xai.tcav.viz import plot_per_system_heatmap, plot_transferability


def parse_dev_protocol_by_system(protocol_path):
    """
    Parse ASVspoof5 dev protocol to group spoof clips by attack system.

    Protocol format (space-delimited, NOT tab despite .tsv extension -- Phase 3 bug fix 7e42e7c):
      col[0] = speaker_id
      col[1] = audio_id
      col[7] = attack_type (A09..A16 for spoof, '-' for bonafide)
      col[8] = 'bonafide' or 'spoof'

    Returns: (systems_dict, bonafide_list)
      systems_dict: {attack_type: [{'audio_id': str, 'speaker_id': str}, ...]}
      bonafide_list: [{'audio_id': str, 'speaker_id': str}, ...]
    """
    systems = defaultdict(list)
    bonafide = []
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()  # space-delimited, NOT split('\t')
            if len(parts) < 9:
                continue
            audio_id = parts[1]
            speaker_id = parts[0]
            attack_type = parts[7]
            key = parts[8]
            if key == 'bonafide':
                bonafide.append({'audio_id': audio_id, 'speaker_id': speaker_id})
            elif key == 'spoof' and attack_type != '-':
                systems[attack_type].append({'audio_id': audio_id, 'speaker_id': speaker_id})
    return dict(systems), bonafide


def check_system_sample_counts(systems, min_samples=50):
    """
    Check which systems meet the minimum sample threshold (SYS-04).

    systems: {system_id: [clip_dicts]}
    min_samples: minimum samples required (default 50)

    Returns: (valid_systems, excluded_systems)
      excluded_systems: {system_id: {'count': int, 'reason': str}}
    """
    valid = {}
    excluded = {}
    for system_id, clips in sorted(systems.items()):
        n = len(clips)
        if n >= min_samples:
            valid[system_id] = clips
        else:
            excluded[system_id] = {'count': n, 'reason': f'< {min_samples} samples'}
            print(f"EXCLUDED: {system_id} has only {n} dev examples (< {min_samples})")
    print(f"\nSystems included: {len(valid)} ({', '.join(sorted(valid.keys()))})")
    if excluded:
        print(f"Systems excluded: {len(excluded)} ({', '.join(sorted(excluded.keys()))})")
    return valid, excluded


def identify_concept_signatures(significance, systems, concepts, layers):
    """
    Identify which concepts are significantly elevated for each system (SYS-02).

    significance: {(system, concept, layer): bool}
    Returns: {system_id: {'significant_concepts': [concept_names]}}
    """
    signatures = {}
    for system in systems:
        sig_concepts = []
        for concept in concepts:
            # A concept is significant for a system if significant at ANY layer
            for layer in layers:
                if significance.get((system, concept, layer), False):
                    sig_concepts.append(concept)
                    break
        signatures[system] = {'significant_concepts': sig_concepts}
    return signatures


def classify_concept_transferability(significance, systems, concepts, layers):
    """
    Classify concepts as universal, system_specific, or intermediate (SYS-03).

    significance: {(system, concept, layer): bool}
    Thresholds: universal >= 6 systems; system_specific <= 2; intermediate 3-5.

    Returns: {concept: {'classification': str, 'n_significant_systems': int,
                         'significant_systems': list, 'best_layer': str}}
    """
    n_systems = len(systems)
    results = {}
    for concept in concepts:
        sig_systems = set()
        layer_sig_counts = {layer: 0 for layer in layers}
        for system in systems:
            for layer in layers:
                if significance.get((system, concept, layer), False):
                    sig_systems.add(system)
                    layer_sig_counts[layer] += 1
        n_sig = len(sig_systems)
        if n_sig >= 6:
            classification = 'universal'
        elif n_sig <= 2:
            classification = 'system_specific'
        else:
            classification = 'intermediate'
        best_layer = max(layer_sig_counts, key=layer_sig_counts.get)
        results[concept] = {
            'n_significant_systems': n_sig,
            'total_systems': n_systems,
            'classification': classification,
            'significant_systems': sorted(sig_systems),
            'best_layer': best_layer,
        }
    return results


def load_system_clips(tar_dir, clip_list, fixed_len=64600, max_clips=500, rng=None):
    """
    Load audio clips for a system from tar files.

    Returns: torch.Tensor of shape (n_clips, fixed_len) or None if no clips found.
    """
    if rng is not None and len(clip_list) > max_clips:
        indices = rng.choice(len(clip_list), size=max_clips, replace=False)
        clip_list = [clip_list[i] for i in sorted(indices)]

    needed = {c['audio_id'] for c in clip_list}
    loaded = {}

    for tar_path in sorted(Path(tar_dir).glob('*.tar')):
        with tarfile.open(str(tar_path), 'r') as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = os.path.basename(member.name)  # path traversal prevention (T-03-01)
                audio_id = name.replace('.flac', '')
                if audio_id in needed:
                    data = tar.extractfile(member).read()
                    audio, sr = sf.read(io.BytesIO(data))
                    audio = audio.astype(np.float32)
                    if len(audio) >= fixed_len:
                        audio = audio[:fixed_len]
                    else:
                        audio = np.pad(audio, (0, fixed_len - len(audio)))
                    loaded[audio_id] = torch.tensor(audio, dtype=torch.float32)
                    if len(loaded) == len(needed):
                        break
        if len(loaded) == len(needed):
            break

    if not loaded:
        return None
    tensors = [loaded[c['audio_id']] for c in clip_list if c['audio_id'] in loaded]
    return torch.stack(tensors, dim=0)


def run_per_system_analysis(config_path=None):
    """
    End-to-end per-system TCAV analysis pipeline.

    Steps:
      1. Load config
      2. Parse protocol by system
      3. Exclude systems below min_samples (SYS-04)
      4. Load model
      5. Per-system TCAV scoring with multiple CAV reruns
      6. BH FDR correction (SYS-02)
      7. Per-system concept signatures (SYS-02)
      8. Concept transferability classification (SYS-03)
      9. Save results JSON
      10. Generate visualizations

    Returns: 0 on success, 1 on failure
    """
    if config_path is None:
        config_path = str(XAI_ROOT / 'config' / 'per_system_config.yaml')

    print(f"\n{'='*60}")
    print("Per-System TCAV Analysis")
    print(f"Config: {config_path}")
    print(f"{'='*60}\n")

    # Step 1: Load config
    cfg = yaml.safe_load(open(config_path))
    layers = cfg['per_system']['layers']
    min_samples = cfg['per_system']['min_system_samples']
    n_reruns = cfg['per_system']['n_cav_reruns']
    n_clips_per_concept = cfg['per_system']['n_clips_per_concept']
    n_random_baselines = cfg['per_system']['n_random_baselines']
    n_clips_per_system = cfg['per_system']['n_clips_per_system']
    alpha = cfg['per_system']['alpha']
    device = cfg['per_system']['device']
    batch_size = cfg['per_system']['batch_size']
    seed = cfg['per_system']['seed']

    rng = np.random.RandomState(seed)
    concept_names = list(cfg['concepts'].keys())
    concept_paths = {k: XAI_ROOT / v for k, v in cfg['concepts'].items()}

    results_dir = XAI_ROOT / cfg['results']['output_dir']
    scores_dir = results_dir / 'scores'
    figures_dir = results_dir / 'figures'
    for d in [scores_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Step 2: Parse protocol by system
    print("Step 1: Parsing dev protocol by synthesis system...")
    dev_protocol = cfg['data']['dev_protocol']
    systems, bonafide = parse_dev_protocol_by_system(dev_protocol)
    print(f"Found {len(systems)} synthesis systems in protocol")
    for sys_id in sorted(systems.keys()):
        print(f"  {sys_id}: {len(systems[sys_id])} clips")
    print(f"  bonafide: {len(bonafide)} clips")

    # Step 3: Exclude systems below threshold
    print(f"\nStep 2: Checking system sample counts (min_samples={min_samples})...")
    valid_systems, excluded_systems = check_system_sample_counts(systems, min_samples)
    if not valid_systems:
        print("ERROR: No systems meet the minimum sample threshold.")
        return 1

    # Step 4: Load model
    print("\nStep 3: Loading RawNet2 model...")
    runner = ExperimentRunner(config_path)
    with runner.run() as exp_dir:
        model = runner.load_model()
        model = model.to(device)
        model.eval()

        # Step 5: Per-system TCAV scoring with multiple CAV reruns
        print(f"\nStep 4: Computing per-system TCAV scores ({n_reruns} CAV reruns)...")
        per_system_scores = {}
        score_distributions = {}

        from xai.tcav.concepts import load_concept_clips

        # Load concept activations once (system-agnostic)
        print("  Extracting concept activations...")
        concept_acts = {}
        for concept_name, manifest_path in concept_paths.items():
            concept_acts[concept_name] = {}
            concept_clips = load_concept_clips(str(manifest_path), n_clips=n_clips_per_concept, rng=rng)
            if concept_clips is None or len(concept_clips) == 0:
                print(f"  WARNING: No clips loaded for concept {concept_name}")
                continue
            concept_clips = concept_clips.to(device)
            with torch.no_grad():
                for layer in layers:
                    extractor = ActivationExtractor(model, LAYER_NAMES[layer])
                    acts = []
                    for i in range(0, len(concept_clips), batch_size):
                        batch = concept_clips[i:i+batch_size]
                        _ = model(batch)
                        act = pool_activation(extractor.activation.detach().cpu().numpy(), layer)
                        acts.append(act)
                    extractor.remove()
                    concept_acts[concept_name][layer] = np.concatenate(acts, axis=0)

        # Generate random baseline activations
        random_acts = {}
        for layer in layers:
            first_concept = next(iter(concept_acts))
            if first_concept in concept_acts and layer in concept_acts[first_concept]:
                d = concept_acts[first_concept][layer].shape[1]
            else:
                d = 128
            random_acts[layer] = [
                np.random.RandomState(seed + i).randn(n_clips_per_concept, d).astype(np.float32)
                for i in range(n_random_baselines)
            ]

        for system_id in tqdm(sorted(valid_systems.keys()), desc='Systems'):
            per_system_scores[system_id] = {}
            clips_data = valid_systems[system_id]

            sys_clips = load_system_clips(
                cfg['data']['tar_dir_dev'], clips_data,
                max_clips=n_clips_per_system, rng=rng
            )
            if sys_clips is None:
                print(f"  WARNING: No clips loaded for system {system_id}, skipping")
                continue
            sys_clips = sys_clips.to(device)

            for concept in tqdm(concept_names, desc=f'{system_id}', leave=False):
                if concept not in concept_acts or not concept_acts[concept]:
                    continue
                per_system_scores[system_id][concept] = {}
                for layer in layers:
                    if layer not in concept_acts.get(concept, {}):
                        continue

                    # Multiple CAV reruns for statistical power (Kim et al. 2018)
                    rerun_scores = []
                    for rerun_i in range(n_reruns):
                        cav_result = train_cav(
                            concept_acts[concept][layer],
                            random_acts[layer][rerun_i % n_random_baselines],
                            random_state=seed + rerun_i * 100
                        )
                        score_result = compute_tcav_score(
                            model, sys_clips, target_class=0,
                            layer_name=layer, cav=cav_result['cav'],
                            scaler=cav_result['scaler'],
                            device=device, batch_size=batch_size
                        )
                        rerun_scores.append(score_result['tcav_score'])

                    # Random baseline scores for significance test
                    random_baseline_scores = []
                    for baseline_i in range(n_random_baselines):
                        rand_acts = random_acts[layer][baseline_i % n_random_baselines]
                        rand_cav = train_cav(
                            rand_acts,
                            concept_acts[concept][layer],
                            random_state=seed + baseline_i * 1000
                        )
                        rand_score = compute_tcav_score(
                            model, sys_clips, target_class=0,
                            layer_name=layer, cav=rand_cav['cav'],
                            scaler=rand_cav['scaler'],
                            device=device, batch_size=batch_size
                        )
                        random_baseline_scores.append([rand_score['tcav_score']])

                    sig_result = test_significance(rerun_scores, random_baseline_scores, n_pairs=1)
                    mean_score = float(np.mean(rerun_scores))
                    score_distributions[(system_id, concept, layer)] = {
                        'rerun_scores': rerun_scores,
                        'mean_score': mean_score,
                        'pval': sig_result['pval'],
                    }
                    per_system_scores[system_id][concept][layer] = {
                        'mean_score': mean_score,
                        'pval': sig_result['pval'],
                        'rerun_scores': rerun_scores,
                    }

        # Step 6: BH FDR correction
        print(f"\nStep 5: Applying BH FDR correction...")
        all_keys = sorted(score_distributions.keys())
        if all_keys:
            all_pvals = np.array([score_distributions[k]['pval'] for k in all_keys])
            fdr_result = fdr_bh_correction(all_pvals, alpha=alpha)
            significance = {}
            for i, key in enumerate(all_keys):
                significance[key] = bool(fdr_result['rejected'][i])
            print(f"  Total tests: {len(all_pvals)}")
            print(f"  Significant after FDR (alpha={alpha}): {fdr_result['n_significant']}")
        else:
            significance = {}
            fdr_result = {'n_significant': 0}
            all_pvals = np.array([])
            print("  No test results available for FDR correction")

        # Step 7: Per-system concept signatures
        print("\nStep 6: Identifying per-system concept signatures...")
        system_list = sorted(valid_systems.keys())
        signatures = identify_concept_signatures(significance, system_list, concept_names, layers)
        for sys_id, sig in signatures.items():
            print(f"  {sys_id}: {sig['significant_concepts']}")

        # Step 8: Concept transferability
        print("\nStep 7: Classifying concept transferability...")
        transferability = classify_concept_transferability(
            significance, system_list, concept_names, layers
        )
        for concept, info in transferability.items():
            print(f"  {concept}: {info['classification']} ({info['n_significant_systems']}/{info['total_systems']} systems)")

        # Step 9: Save results
        print("\nStep 8: Saving results...")
        results_summary = {
            'config': config_path,
            'systems_analyzed': system_list,
            'systems_excluded': {k: v for k, v in excluded_systems.items()},
            'n_concepts': len(concept_names),
            'n_layers': len(layers),
            'n_tests_total': int(len(all_pvals)),
            'n_significant': int(fdr_result['n_significant']),
            'alpha': alpha,
            'fdr_method': cfg['per_system']['fdr_method'],
            'per_system_scores': {
                sys_id: {
                    concept: {
                        layer: {
                            'mean_score': per_system_scores[sys_id][concept][layer]['mean_score'],
                            'pval': per_system_scores[sys_id][concept][layer]['pval'],
                            'significant': significance.get((sys_id, concept, layer), False),
                        }
                        for layer in layers
                        if layer in per_system_scores.get(sys_id, {}).get(concept, {})
                    }
                    for concept in concept_names
                    if concept in per_system_scores.get(sys_id, {})
                }
                for sys_id in system_list
                if sys_id in per_system_scores
            },
            'concept_signatures': signatures,
            'transferability': transferability,
        }

        results_path = scores_dir / 'per_system_results.json'
        with open(str(results_path), 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"  Saved: {results_path}")

        runner.save_results(results_summary)

        # Step 10: Visualizations
        print("\nStep 9: Generating visualizations...")
        best_layer = 'post_gru'
        score_matrix = np.zeros((len(system_list), len(concept_names)))
        sig_mask = np.zeros((len(system_list), len(concept_names)), dtype=bool)
        for i, sys_id in enumerate(system_list):
            for j, concept in enumerate(concept_names):
                if (sys_id in per_system_scores and
                        concept in per_system_scores.get(sys_id, {}) and
                        best_layer in per_system_scores[sys_id].get(concept, {})):
                    score_matrix[i, j] = per_system_scores[sys_id][concept][best_layer]['mean_score']
                    sig_mask[i, j] = significance.get((sys_id, concept, best_layer), False)

        heatmap_path = str(figures_dir / 'per_system_heatmap.png')
        plot_per_system_heatmap(
            score_matrix, system_list, concept_names, sig_mask, heatmap_path,
            title=f'Per-System Concept Attribution (layer: {best_layer})'
        )
        print(f"  Saved: {heatmap_path}")

        transferability_path = str(figures_dir / 'transferability_clustermap.png')
        plot_transferability(score_matrix, system_list, concept_names, transferability_path)
        print(f"  Saved: {transferability_path}")

    print("\nPer-system TCAV analysis complete.")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Per-system TCAV analysis')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to per_system_config.yaml')
    args = parser.parse_args()
    sys.exit(run_per_system_analysis(args.config))
