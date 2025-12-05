# Label Smoothing and Layer Normalization Experiments

This document describes the experiments set up for finding optimal parameters for Label Smoothing and Layer Normalization.

## Features Implemented

### 1. Label Smoothing
- **Location**: `src/neural_decoder/neural_decoder_trainer.py`
- **Function**: `apply_label_smoothing()`
- **Description**: Applies label smoothing to logits before CTC loss computation. This helps prevent overconfidence and improves generalization.
- **Parameter**: `label_smoothing` (float, range: 0.0-1.0, typical: 0.0-0.1)
  - `0.0`: No smoothing (baseline)
  - `0.05`: Light smoothing
  - `0.1`: Moderate smoothing
  - `0.15+`: Strong smoothing (may hurt performance)

### 2. Layer Normalization
- **Location**: `src/neural_decoder/model.py`
- **Description**: Adds LayerNorm layers at strategic points in the architecture to stabilize training and improve convergence.
- **Parameters**:
  - `use_layer_norm` (bool): Enable/disable layer normalization
  - `layer_norm_position` (str): Where to apply layer normalization
    - `"after_input"`: After input transformation (day layer)
    - `"after_gru"`: After GRU layers (before output layer)
    - `"both"`: Both positions
    - `"none"`: No normalization (when `use_layer_norm=False`)

## Configuration

### Base Configuration
Edit `src/neural_decoder/conf/config.yaml` to set default values:
```yaml
label_smoothing: 0.0
use_layer_norm: False
layer_norm_position: after_gru
```

## Running Experiments

### Option 1: Python Script (Recommended)
Run all predefined experiments:
```bash
python scripts/run_optimal_experiments.py
```

This will run:
1. Baseline (no modifications)
2. Label smoothing sweep (0.0, 0.05, 0.1, 0.15)
3. Layer normalization sweep (after_gru, after_input, both)
4. Combined experiments (optimal combinations)

### Option 2: Hydra Multirun (Parallel Execution)
Run comprehensive sweep using Hydra's multirun:
```bash
bash scripts/run_hydra_sweep.sh
```

Or manually:
```bash
python -m neural_decoder.neural_decoder_trainer \
    --config-path=src/neural_decoder/conf \
    --config-name=config \
    -m \
    label_smoothing=0.0,0.05,0.1 \
    use_layer_norm=False,True \
    layer_norm_position=after_gru,both
```

### Option 3: Individual Experiments
Run a single experiment:
```bash
python -m neural_decoder.neural_decoder_trainer \
    --config-path=src/neural_decoder/conf \
    --config-name=config \
    label_smoothing=0.1 \
    use_layer_norm=True \
    layer_norm_position=after_gru \
    outputDir=/path/to/output
```

### Option 4: Shell Script
Run sequential experiments:
```bash
bash scripts/run_experiments.sh
```

## Experiment Design

### Phase 1: Individual Feature Testing
1. **Label Smoothing Only**
   - Test values: 0.0, 0.05, 0.1, 0.15
   - Keep `use_layer_norm=False`
   - Identify optimal smoothing value

2. **Layer Normalization Only**
   - Test positions: after_gru, after_input, both
   - Keep `label_smoothing=0.0`
   - Identify optimal normalization position

### Phase 2: Combined Testing
- Combine best label smoothing value with best layer norm configuration
- Test additional combinations for synergy effects

## Expected Results

Results will be saved in the output directory with:
- `modelWeights`: Best model checkpoint
- `trainingStats`: Training statistics (loss, CER over time)
- `args`: Configuration used for the experiment

### Metrics to Compare
1. **Test Loss**: Lower is better
2. **Character Error Rate (CER)**: Lower is better
3. **Training Stability**: Check loss curves for smooth convergence
4. **Convergence Speed**: Number of batches to reach best performance

## Analyzing Results

After running experiments, compare:
1. Final test CER for each configuration
2. Training curves (loss over batches)
3. Convergence speed
4. Training stability (smooth vs. noisy curves)

Example analysis script:
```python
import pickle
import numpy as np
import os

# Load results from each experiment
experiments = ["baseline", "label_smoothing_0.1", "layer_norm_after_gru", ...]
results = {}

for exp_name in experiments:
    stats_path = f"/path/to/{exp_name}/trainingStats"
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    results[exp_name] = {
        "final_cer": stats["testCER"][-1],
        "best_cer": np.min(stats["testCER"]),
        "final_loss": stats["testLoss"][-1],
    }

# Print comparison
for exp_name, metrics in results.items():
    print(f"{exp_name}: CER={metrics['best_cer']:.4f}, Loss={metrics['final_loss']:.4f}")
```

## Recommendations

Based on typical results:
- **Label Smoothing**: Start with 0.05-0.1 for CTC tasks
- **Layer Normalization**: `after_gru` is often most effective for RNNs
- **Combined**: Often provides best results when both features complement each other

## Notes

- Label smoothing is applied during both training and evaluation for consistency
- Layer normalization uses PyTorch's `nn.LayerNorm` with default parameters
- All experiments use the same random seed for reproducibility
- Adjust `nBatch` in config if you want shorter experiments for quick testing

