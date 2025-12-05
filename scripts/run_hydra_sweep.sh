#!/bin/bash
# Script to run Hydra multirun sweep for comprehensive hyperparameter search
# This uses Hydra's built-in multirun feature for parallel execution

echo "Starting Hydra multirun sweep for Label Smoothing and Layer Normalization..."

# Run comprehensive sweep using Hydra multirun
python -m neural_decoder.neural_decoder_trainer \
    --config-path=src/neural_decoder/conf \
    --config-name=config \
    -m \
    label_smoothing=0.0,0.05,0.1 \
    use_layer_norm=False,True \
    layer_norm_position=after_gru,both \
    outputDir=/oak/stanford/groups/henderj/stfan/logs/speech_logs/pt_neural_decoder_sweep

echo "Hydra sweep completed!"
echo "Results saved in output directory with subdirectories for each configuration"

