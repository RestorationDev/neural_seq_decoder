#!/usr/bin/env python3
"""
Script to run optimal parameter experiments for Label Smoothing and Layer Normalization.
This script systematically tests different combinations to find optimal hyperparameters.
"""

import os
import sys

# Set PyTorch CUDA memory allocator to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Change to neural_decoder directory for config loading
neural_decoder_dir = os.path.join(project_root, 'src', 'neural_decoder')
os.chdir(neural_decoder_dir)

# Import after path setup
from omegaconf import OmegaConf
from neural_decoder.neural_decoder_trainer import trainModel
import torch
import gc

# Colab paths
BASE_DIR = "/content/drive/MyDrive/ECEC243A/FinalProject/outputs"
DATASET_PATH = "/content/drive/MyDrive/ECEC243A/FinalProject/data/ptDecoder_ctc"

print(f"[INFO] Using Colab paths.", flush=True)
print(f"[INFO] BASE_DIR: {BASE_DIR}", flush=True)
print(f"[INFO] DATASET_PATH: {DATASET_PATH}", flush=True)

# Verify dataset path exists
if not os.path.exists(DATASET_PATH):
    print(f"[WARNING] Dataset path does not exist: {DATASET_PATH}", flush=True)
    print(f"[WARNING] Please ensure the dataset file exists at this path.", flush=True)

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


def run_experiment(exp_config, base_config):
    """Run a single experiment with given configuration."""
    # Clear GPU memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Check available memory
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        free_memory_gb = free_memory / (1024**3)
        print(f"[INFO] Available GPU memory: {free_memory_gb:.2f} GB", flush=True)
        if free_memory_gb < 2.0:
            print(f"[WARNING] Low GPU memory available. Consider killing other processes or reducing batch size.", flush=True)
    gc.collect()
    
    name = exp_config["name"]
    output_dir = os.path.join(BASE_DIR, name)
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {name}")
    print(f"  Label Smoothing: {exp_config['label_smoothing']}")
    print(f"  Layer Norm: {exp_config['use_layer_norm']}")
    print(f"  Layer Norm Position: {exp_config['layer_norm_position']}")
    print(f"  Output Directory: {output_dir}")
    print(f"{'='*60}\n", flush=True)
    
    # Create a copy of base config and update with experiment-specific values
    # Remove Hydra-specific sections that cause interpolation errors
    config_dict = OmegaConf.to_container(base_config, resolve=False)
    if 'hydra' in config_dict:
        del config_dict['hydra']
    
    # CRITICAL: Reduce batch size BEFORE creating OmegaConf to ensure it's applied
    # With limited GPU memory, we need smaller batches
    config_dict['batchSize'] = 8  # Reduced from 64 to 8 (8x reduction in memory per batch)
    
    # Create new config from dict
    cfg = OmegaConf.create(config_dict)
    
    # Update with experiment-specific values (use both dict and OmegaConf syntax to ensure it works)
    cfg['outputDir'] = output_dir
    cfg['datasetPath'] = DATASET_PATH
    cfg['label_smoothing'] = exp_config['label_smoothing']
    cfg['use_layer_norm'] = exp_config['use_layer_norm']
    cfg['layer_norm_position'] = exp_config['layer_norm_position']
    # Skip test evaluation to save GPU memory (can evaluate separately later)
    cfg['skip_test_eval'] = True
    # Ensure batch size is set (in case dict update didn't work)
    cfg.batchSize = 8
    cfg['batchSize'] = 8
    
    # Debug: Verify batch size is actually set
    print(f"[DEBUG] Batch size set to: {cfg.get('batchSize', 'NOT SET')}", flush=True)
    print(f"[DEBUG] Config batchSize value: {cfg.batchSize}", flush=True)
    
    try:
        trainModel(cfg)
        print(f"✓ Experiment {name} completed successfully", flush=True)
        
        # Clear GPU memory after each experiment
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        return True
    except Exception as e:
        print(f"✗ Experiment {name} failed with error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        
        # Clear GPU memory even on failure
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        return False


def main():
    """Run all experiments."""
    print("="*60)
    print("Label Smoothing and Layer Normalization Experiments")
    print("="*60)
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"Base directory: {BASE_DIR}")
    print(f"Dataset path: {DATASET_PATH}")
    print()
    
    # Check GPU memory status
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
        reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)
        free_memory = total_memory - reserved_memory
        
        print(f"[INFO] GPU Memory Status:")
        print(f"  Total: {total_memory:.2f} GB")
        print(f"  Allocated: {allocated_memory:.2f} GB")
        print(f"  Reserved: {reserved_memory:.2f} GB")
        print(f"  Free: {free_memory:.2f} GB")
        print()
        
        if free_memory < 3.0:
            print("[WARNING] Low GPU memory available!")
            print("[WARNING] The model needs ~2 GB for backprop. Consider:")
            print("  1. Killing other GPU processes: !nvidia-smi")
            print("  2. Restarting Colab runtime: Runtime -> Restart runtime")
            print("  3. Reducing batch size in config.yaml")
            print()
    
    # Load base config
    config_path = os.path.join(neural_decoder_dir, 'conf', 'config.yaml')
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found at {config_path}")
        return
    
    base_config = OmegaConf.load(config_path)
    print(f"Loaded config from: {config_path}")
    print()
    
    # Create base directory
    os.makedirs(BASE_DIR, exist_ok=True)
    
    results = []
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}]")
        success = run_experiment(exp, base_config)
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

