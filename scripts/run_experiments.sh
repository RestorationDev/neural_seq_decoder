#!/bin/bash
# Script to run hyperparameter sweep experiments for Label Smoothing and Layer Normalization
# This script runs multiple experiments with different combinations of hyperparameters

# Base directory for experiments
BASE_DIR="/oak/stanford/groups/henderj/stfan/logs/speech_logs/pt_neural_decoder_experiments"

# Label smoothing values to test
LABEL_SMOOTHING_VALUES=(0.0 0.05 0.1)

# Layer normalization configurations
LAYER_NORM_CONFIGS=("False:after_gru" "True:after_gru" "True:after_input" "True:both")

echo "Starting hyperparameter sweep experiments..."
echo "Base directory: $BASE_DIR"

# Experiment 1: Label Smoothing only (baseline comparison)
echo ""
echo "=== Experiment 1: Label Smoothing Sweep ==="
for smoothing in "${LABEL_SMOOTHING_VALUES[@]}"; do
    echo "Running with label_smoothing=$smoothing, use_layer_norm=False"
    python -m neural_decoder.neural_decoder_trainer \
        --config-path=src/neural_decoder/conf \
        --config-name=config \
        outputDir="${BASE_DIR}/label_smoothing_${smoothing}" \
        label_smoothing=${smoothing} \
        use_layer_norm=False
done

# Experiment 2: Layer Normalization only (baseline comparison)
echo ""
echo "=== Experiment 2: Layer Normalization Sweep ==="
for config in "${LAYER_NORM_CONFIGS[@]}"; do
    IFS=':' read -r use_norm norm_pos <<< "$config"
    echo "Running with use_layer_norm=$use_norm, layer_norm_position=$norm_pos"
    python -m neural_decoder.neural_decoder_trainer \
        --config-path=src/neural_decoder/conf \
        --config-name=config \
        outputDir="${BASE_DIR}/layer_norm_${use_norm}_${norm_pos}" \
        label_smoothing=0.0 \
        use_layer_norm=${use_norm} \
        layer_norm_position=${norm_pos}
done

# Experiment 3: Combined (optimal combinations)
echo ""
echo "=== Experiment 3: Combined Label Smoothing + Layer Normalization ==="
OPTIMAL_SMOOTHING=0.1  # Adjust based on Experiment 1 results
OPTIMAL_NORM_CONFIG="True:after_gru"  # Adjust based on Experiment 2 results

IFS=':' read -r use_norm norm_pos <<< "$OPTIMAL_NORM_CONFIG"
echo "Running with label_smoothing=$OPTIMAL_SMOOTHING, use_layer_norm=$use_norm, layer_norm_position=$norm_pos"
python -m neural_decoder.neural_decoder_trainer \
    --config-path=src/neural_decoder/conf \
    --config-name=config \
    outputDir="${BASE_DIR}/combined_${OPTIMAL_SMOOTHING}_${use_norm}_${norm_pos}" \
    label_smoothing=${OPTIMAL_SMOOTHING} \
    use_layer_norm=${use_norm} \
    layer_norm_position=${norm_pos}

echo ""
echo "All experiments completed!"
echo "Results saved in: $BASE_DIR"

