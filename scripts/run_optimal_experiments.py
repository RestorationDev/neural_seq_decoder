#!/usr/bin/env python3
"""
Script to run optimal parameter experiments for Label Smoothing and Layer Normalization.
This script systematically tests different combinations to find optimal hyperparameters.
"""

import subprocess
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

BASE_DIR = "/oak/stanford/groups/henderj/stfan/logs/speech_logs/pt_neural_decoder_experiments"

# Experiment configurations
EXPERIMENTS = [
    # Baseline (no modifications)
    {
        "name": "baseline",
        "label_smoothing": 0.0,
        "use_layer_norm": False,
        "layer_norm_position": "after_gru",
    },
    
    # Label Smoothing experiments
    {
        "name": "label_smoothing_0.05",
        "label_smoothing": 0.05,
        "use_layer_norm": False,
        "layer_norm_position": "after_gru",
    },
    {
        "name": "label_smoothing_0.1",
        "label_smoothing": 0.1,
        "use_layer_norm": False,
        "layer_norm_position": "after_gru",
    },
    {
        "name": "label_smoothing_0.15",
        "label_smoothing": 0.15,
        "use_layer_norm": False,
        "layer_norm_position": "after_gru",
    },
    
    # Layer Normalization experiments
    {
        "name": "layer_norm_after_gru",
        "label_smoothing": 0.0,
        "use_layer_norm": True,
        "layer_norm_position": "after_gru",
    },
    {
        "name": "layer_norm_after_input",
        "label_smoothing": 0.0,
        "use_layer_norm": True,
        "layer_norm_position": "after_input",
    },
    {
        "name": "layer_norm_both",
        "label_smoothing": 0.0,
        "use_layer_norm": True,
        "layer_norm_position": "both",
    },
    
    # Combined experiments (optimal combinations)
    {
        "name": "combined_smooth_0.1_norm_gru",
        "label_smoothing": 0.1,
        "use_layer_norm": True,
        "layer_norm_position": "after_gru",
    },
    {
        "name": "combined_smooth_0.05_norm_both",
        "label_smoothing": 0.05,
        "use_layer_norm": True,
        "layer_norm_position": "both",
    },
    {
        "name": "combined_smooth_0.1_norm_both",
        "label_smoothing": 0.1,
        "use_layer_norm": True,
        "layer_norm_position": "both",
    },
]


def run_experiment(exp_config):
    """Run a single experiment with given configuration."""
    name = exp_config["name"]
    output_dir = os.path.join(BASE_DIR, name)
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {name}")
    print(f"  Label Smoothing: {exp_config['label_smoothing']}")
    print(f"  Layer Norm: {exp_config['use_layer_norm']}")
    print(f"  Layer Norm Position: {exp_config['layer_norm_position']}")
    print(f"  Output Directory: {output_dir}")
    print(f"{'='*60}\n")
    
    cmd = [
        sys.executable,
        "-m", "neural_decoder.neural_decoder_trainer",
        "--config-path", "src/neural_decoder/conf",
        "--config-name", "config",
        f"outputDir={output_dir}",
        f"label_smoothing={exp_config['label_smoothing']}",
        f"use_layer_norm={exp_config['use_layer_norm']}",
        f"layer_norm_position={exp_config['layer_norm_position']}",
    ]
    
    try:
        result = subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.dirname(__file__)))
        print(f"✓ Experiment {name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Experiment {name} failed with error: {e}")
        return False


def main():
    """Run all experiments."""
    print("="*60)
    print("Label Smoothing and Layer Normalization Experiments")
    print("="*60)
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"Base directory: {BASE_DIR}")
    print()
    
    # Create base directory
    os.makedirs(BASE_DIR, exist_ok=True)
    
    results = []
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}]")
        success = run_experiment(exp)
        results.append((exp["name"], success))
    
    # Print summary
    print("\n" + "="*60)
    print("Experiment Summary")
    print("="*60)
    for name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {name}")
    
    successful = sum(1 for _, success in results if success)
    print(f"\nTotal: {successful}/{len(EXPERIMENTS)} experiments completed successfully")
    print(f"Results saved in: {BASE_DIR}")


if __name__ == "__main__":
    main()

