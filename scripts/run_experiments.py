#!/usr/bin/env python3
"""Simple script to run experiments with different label smoothing and layer norm settings."""

import os
from neural_decoder.neural_decoder_trainer import trainModel

# Base paths (update for Colab)
BASE_DIR = "/content/drive/MyDrive/ECEC243A/FinalProject/outputs"
DATASET_PATH = "/content/drive/MyDrive/ECEC243A/FinalProject/data/ptDecoder_ctc"

# Experiments to run
experiments = [
    {"name": "baseline", "label_smoothing": 0.0, "use_layer_norm": False},
    {"name": "label_smooth_0.05", "label_smoothing": 0.05, "use_layer_norm": False},
    {"name": "label_smooth_0.1", "label_smoothing": 0.1, "use_layer_norm": False},
    {"name": "layer_norm", "label_smoothing": 0.0, "use_layer_norm": True},
    {"name": "combined_0.1_norm", "label_smoothing": 0.1, "use_layer_norm": True},
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
    print(f"{'='*60}\n")
    
    args = base_args.copy()
    args["outputDir"] = os.path.join(BASE_DIR, exp["name"])
    args["label_smoothing"] = exp["label_smoothing"]
    args["use_layer_norm"] = exp["use_layer_norm"]
    
    try:
        trainModel(args)
        print(f"✓ Completed: {exp['name']}\n")
    except Exception as e:
        print(f"✗ Failed: {exp['name']} - {e}\n")

print("All experiments completed!")

