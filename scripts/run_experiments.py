#!/usr/bin/env python3
"""Simple script to run experiments with different label smoothing and layer norm settings."""

import os
from neural_decoder.neural_decoder_trainer import trainModel

# Base paths (update for Colab)
BASE_DIR = "/content/drive/MyDrive/ECEC243A/FinalProject/outputs"
DATASET_PATH = "/content/drive/MyDrive/ECEC243A/FinalProject/data/ptDecoder_ctc"

# Experiments to run - Focused on architecture improvements
experiments = [
    # ===== NEW ARCHITECTURE EXPERIMENTS =====
    {
        "name": "post_gru_stack",
        "label_smoothing": 0.0,
        "use_layer_norm": False,
        "layer_norm_position": "none",
        "use_post_gru_stack": True,
        "post_gru_stack_layers": 2,
        "post_gru_stack_dropout": 0.1
    },
    {
        "name": "progressive_smoothing",
        "label_smoothing": "progressive",  # Decays from 0.15 to 0.05
        "use_layer_norm": False,
        "layer_norm_position": "none",
        "use_post_gru_stack": False
    },
    {
        "name": "layer_norm_pre",
        "label_smoothing": 0.0,
        "use_layer_norm": True,
        "layer_norm_position": "pre",
        "use_post_gru_stack": False
    },
    {
        "name": "layer_norm_dual",
        "label_smoothing": 0.0,
        "use_layer_norm": True,
        "layer_norm_position": "dual",
        "use_post_gru_stack": False
    },
    {
        "name": "progressive_post_gru_stack",
        "label_smoothing": "progressive",
        "use_layer_norm": False,
        "layer_norm_position": "none",
        "use_post_gru_stack": True,
        "post_gru_stack_layers": 2,
        "post_gru_stack_dropout": 0.1
    },
    {
        "name": "pre_norm_post_gru_stack",
        "label_smoothing": 0.0,
        "use_layer_norm": True,
        "layer_norm_position": "pre",
        "use_post_gru_stack": True,
        "post_gru_stack_layers": 2,
        "post_gru_stack_dropout": 0.1
    },
    {
        "name": "dual_norm_post_gru_stack",
        "label_smoothing": 0.0,
        "use_layer_norm": True,
        "layer_norm_position": "dual",
        "use_post_gru_stack": True,
        "post_gru_stack_layers": 2,
        "post_gru_stack_dropout": 0.1
    },
    {
        "name": "progressive_dual_norm_post_gru_stack",
        "label_smoothing": "progressive",
        "use_layer_norm": True,
        "layer_norm_position": "dual",
        "use_post_gru_stack": True,
        "post_gru_stack_layers": 2,
        "post_gru_stack_dropout": 0.1
    },
    
    # ===== PREVIOUS EXPERIMENTS (Commented out - run separately) =====
    # {
    #     "name": "baseline",
    #     "label_smoothing": 0.0,
    #     "use_layer_norm": False,
    #     "layer_norm_position": "none"
    # },
    # {
    #     "name": "label_smooth_0.05",
    #     "label_smoothing": 0.05,
    #     "use_layer_norm": False,
    #     "layer_norm_position": "none"
    # },
    # {
    #     "name": "label_smooth_0.1",
    #     "label_smoothing": 0.1,
    #     "use_layer_norm": False,
    #     "layer_norm_position": "none"
    # },
    # {
    #     "name": "layer_norm",
    #     "label_smoothing": 0.0,
    #     "use_layer_norm": True,
    #     "layer_norm_position": "post"  # This is the default post-norm
    # },
    # {
    #     "name": "combined_0.1_norm",
    #     "label_smoothing": 0.1,
    #     "use_layer_norm": True,
    #     "layer_norm_position": "post"  # 0.1 smoothing + post-norm
    # },
]

# Base config (from config.yaml defaults)
base_args = {
    "outputDir": "",
    "datasetPath": DATASET_PATH,
    "seqLen": 150,
    "maxTimeSeriesLen": 1200,
    "batchSize": 64,
    "lrStart": 0.02,
    "lrEnd": 0.02,
    "nUnits": 1024,
    "nBatch": 10000,
    "nLayers": 5,
    "seed": 0,
    "nClasses": 40,
    "nInputFeatures": 256,
    "dropout": 0.4,
    "whiteNoiseSD": 0.8,
    "constantOffsetSD": 0.2,
    "gaussianSmoothWidth": 2.0,
    "strideLen": 4,
    "kernelLen": 32,
    "bidirectional": True,
    "l2_decay": 1e-5,
}

# Run experiments
os.makedirs(BASE_DIR, exist_ok=True)
for exp in experiments:
    print(f"\n{'='*60}")
    print(f"Running: {exp['name']}")
    print(f"  Label Smoothing: {exp['label_smoothing']}")
    print(f"  Layer Norm: {exp['use_layer_norm']}")
    if exp.get('use_layer_norm', False):
        print(f"  Layer Norm Position: {exp.get('layer_norm_position', 'post')}")
    if exp.get('use_post_gru_stack', False):
        print(f"  Post-GRU Stack: {exp.get('post_gru_stack_layers', 2)} layers, dropout={exp.get('post_gru_stack_dropout', 0.1)}")
    print(f"{'='*60}\n")
    
    args = base_args.copy()
    args["outputDir"] = os.path.join(BASE_DIR, exp["name"])
    args["label_smoothing"] = exp["label_smoothing"]
    args["use_layer_norm"] = exp.get("use_layer_norm", False)
    args["layer_norm_position"] = exp.get("layer_norm_position", "post")
    args["use_post_gru_stack"] = exp.get("use_post_gru_stack", False)
    args["post_gru_stack_layers"] = exp.get("post_gru_stack_layers", 2)
    args["post_gru_stack_dropout"] = exp.get("post_gru_stack_dropout", 0.1)
    
    try:
        trainModel(args)
        print(f"✓ Completed: {exp['name']}\n")
    except Exception as e:
        print(f"✗ Failed: {exp['name']} - {e}\n")

print("All experiments completed!")

