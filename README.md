# ToxMamba: Peptide Toxicity Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

The accurate identification of toxic peptides is of great significance for drug safety evaluation and biological defense. This study proposes ToxMamba, a toxic peptide prediction framework that integrates ISM pre-trained embeddings, state space model (State Space Model, SSM), multi-scale feature learning, and voting mechanisms. The model is implemented based on the Mamba architecture for efficient feature modeling and captures sequence patterns under different receptive fields through parallel branch structures. On an independent test dataset, ToxMamba consistently outperforms existing methods, improving the F1 score, MCC, and AUC metrics by 4.5%-31.7%, 11.2%-46.8%, and 2%-31%, respectively. Ablation experiments further indicate that the Mamba variant based on SSM is superior to CNN-LSTM-Attention, Transformer, and KAN in all evaluation metrics. Feature visualization results show that ISM embeddings can more clearly distinguish the feature space distribution of toxic and non-toxic samples; intermediate layer feature analysis further reveals key residue decision sites consistent with known toxic functional motifs, indicating that the model has good biological interpretability. Overall, this study provides an effective and interpretable computational framework for the high-throughput screening of toxic peptides.
<img width="1073" height="650" alt="image" src="https://github.com/user-attachments/assets/da3dc121-6f23-4fd1-a17a-a48eaa53e85f" />


## Overview

ToxMamba implements a state-of-the-art peptide toxicity prediction model based on the Mamba architecture. The model takes peptide sequences as input and predicts their toxicity (binary classification: toxic vs. non-toxic).

### Key Features

- **Mamba-based Architecture**: Multi-scale selective state space models for sequence modeling
- **Pseudo-Siamese Convolution**: Position-specific convolutions with three kernel sizes (3, 5, 7)
- **Multi-Representation Fusion**: Adaptive fusion of features from different Mamba branches
- **Multiple Embedding Modes**: Supports one-hot encoding, adaptive tokenization, and pretrained ESM-2 embeddings
- **5-Fold Cross-Validation**: Rigorous evaluation with independent test set
- **Comprehensive Metrics**: AUC, AUPR, MCC, F1, Sensitivity, Specificity, Accuracy, and more
- **Reproducible**: Fixed random seeds and deterministic settings

## Repository Structure

```
ToxMamba/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore
├── LICENSE
├── configs/
│   └── config.toml                    # Configuration file (hyperparameters, paths)
├── data/
│   └── PTP_SMGCA_data/               # Dataset directory
│       ├── train.csv                  # Training data (name, sequence, label)
│       ├── test.csv                   # Test data (for CV evaluation)
│       └── independent_test.csv       # Independent test set
├── src/
│   ├── main.py                        # Main entry point
│   ├── args.py                        # Configuration and argument parsing
│   ├── trainer.py                     # Training, validation, and testing loops
│   ├── data/
│   │   ├── dataset.py                 # Dataset classes for data loading
│   │   └── collate.py                 # Collate functions for batch processing
│   ├── models/
│   │   └── mamba_classifier.py        # Mamba-based classifier architecture
│   └── utils/
│       ├── loss.py                    # Loss functions (FocalLoss, PolyLoss, etc.)
│       ├── metrics.py                 # Evaluation metrics and visualization
│       └── helpers.py                 # Utility functions (early stopping, seeding)
├── scripts/
│   └── run.sh                         # Shell script for training & evaluation
└── results/                           # Output directory (created at runtime)
```

## Installation

### Prerequisites

- Python 3.9 or later
- CUDA-capable GPU (recommended) or CPU-only mode

### Setup

```bash
# Clone the repository
git clone https://github.com/Marscolono/ToxMamba.git
cd ToxMamba

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >= 2.0.0 | Deep learning framework |
| numpy | >= 1.24.0 | Numerical computing |
| pandas | >= 2.0.0 | Data handling |
| scikit-learn | >= 1.3.0 | Evaluation metrics |
| tqdm | >= 4.65.0 | Progress bars |
| einops | >= 0.7.0 | Tensor operations |
| prettytable | >= 3.9.0 | Formatted output |
| tomli | >= 2.0.0 | TOML config parsing (Python < 3.11) |
| matplotlib | >= 3.7.0 | Loss curve plotting |
| scipy | >= 1.11.0 | Statistical tests |
| prefetch-generator | >= 1.0.3 | Efficient data loading |

## Data Format

### Input Data Format

Data files are tab-separated or comma-separated text files with header:

| Column | Description |
|--------|-------------|
| `name` | Sample identifier |
| `seq` | Peptide sequence (amino acid letters) |
| `label` | Binary label (0 = non-toxic, 1 = toxic) |

**Example (`train.csv`):**
```
name	seq	label
train_1	CGDINAPCQSDCDCCGYSVTCDCYWSKDCKCRESNFVIGMALRKAFCKNK	1
train_2	ALCCYGYRFCCPNFR	1
train_3	FKGRMITHKEIGAKVLAEFAEKTQDI	0
```

### Data Organization Modes

Controlled by `file_sytle` in `config.toml`:
- **`train_test_2file`** (default): `train.csv` + `test.csv` — internal 5-fold CV
- **`train_valid_test_3file`**: Pre-split `train_0.txt`, `valid_0.txt`, ..., `test.txt`
- **`train_valid_1file_only5fold`**: Single `train.csv`, 5-fold CV only

## Configuration

All hyperparameters are controlled via `configs/config.toml`:

```toml
# Dataset
species = 'Toxi_data'

# Model Architecture
d_model = 1280
n_layer = 6
d_state = 16
expand = 4

# Training
epochs = 80
batch_size = 256
Learning_rate = 1e-4
K_fold = 5
Patience = 20

# Loss Function
loss_func_name = 'FocalLoss'  # Options: FocalLoss, CrossEntropyLoss, PolyLoss_drugban
```

## Usage

### Quick Start

```bash
bash scripts/run.sh
```

### Manual Execution

```bash
# Training + Testing
python src/main.py \
    --model-name S_Pretrain_emb_Mamba_ISM \
    --embed-type pretrain_ISM \
    --tomlpath configs/config.toml \
    --seed 42

# Testing only (requires trained weights)
# Set train_test_mode = ['test'] in config.toml first
python src/main.py \
    --model-name S_Pretrain_emb_Mamba_ISM \
    --embed-type pretrain_ISM \
    --tomlpath configs/config.toml \
    --seed 42
```

### Embedding Types

| Type | Description | Requires |
|------|-------------|----------|
| `adapt` | Integer token encoding | Nothing extra |
| `onehot` | One-hot amino acid encoding | Nothing extra |
| `pretrain` / `pretrain_ISM` | ESM-2 embeddings | `.pt` files in `data/{dataset}/ESM2_embeddings/` |

### Generating ESM-2 Embeddings

```python
import torch
from transformers import AutoTokenizer, AutoModel

model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

for name, seq in sequences:
    inputs = tokenizer(seq, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state  # (1, seq_len, 1280)
    torch.save(embedding, f"data/{dataset}/ESM2_embeddings/{name}.pt")
```

## Model Architecture

```
Input: Pretrained Embedding (batch, seq_len, d_model)
    │
    ├── Branch 1 (kernel=3) ──► Mamba Block x6 ──► RMSNorm ──► Y3
    ├── Branch 2 (kernel=5) ──► Mamba Block x6 ──► RMSNorm ──► Y5
    └── Branch 3 (kernel=7) ──► Mamba Block x6 ──► RMSNorm ──► Y7
                                      │
                              Multi-Representation Fusion
                                      │
                              Flatten → MLP → Output (2 classes)
```

- **Parameters**: ~330M
- **PSConv**: Position-specific convolutions with learnable per-position weights
- **SSM**: Mamba's selective scan for efficient long-range modeling
- **RMSNorm**: Root Mean Square normalization

## Output and Results

Results are saved in `results/{dataset}/{model}_{loss}_{num_class}/`:

```
├── log_file/
│   ├── log_train_valid_{fold}.txt      # Per-fold training logs
│   └── log_test_ensemble.txt           # Test evaluation logs
├── loss_pictures/
│   └── loss_curve_fold_{fold}.png      # Loss curves
├── label_pred_screen/
│   └── test_result_fold_{fold}.txt     # Predictions (name, label, score)
├── metrics/
│   └── metrics_test_{fold}.csv         # Per-fold metrics
├── model_metrics_5fold.csv             # Aggregate results
└── {model_name}_{fold}.pth             # Model checkpoints
```

### Metrics Reported

| Metric | Description |
|--------|-------------|
| AUC | Area Under ROC Curve |
| AUPR | Area Under Precision-Recall Curve |
| ACC / BACC | Accuracy / Balanced Accuracy |
| SN / SP | Sensitivity / Specificity |
| Precision / F1 | Precision / F1 Score |
| MCC | Matthews Correlation Coefficient |

### Prediction File Format

```
sample_name    true_label    predicted_score
train_1        1             0.9823
train_2        0             0.1204
```

## Reproducibility

- Fixed random seeds (default: 42)
- Deterministic cuDNN settings
- Config and code snapshot saved in output directory

## Extending

### Adding a New Dataset

1. Create `data/YOUR_DATASET/` with `train.csv` and `test.csv`
2. Add config in `src/args.py` → `_get_dataset_config()`
3. Set `species = 'YOUR_DATASET'` in `config.toml`

### Adding a New Model

1. Define model class in `src/models/` with `forward()` and `train_model()` methods
2. Register in `src/trainer.py` → `_instantiate_model()`

## Citation

```bibtex
@article{toxmamba,
  title={ToxMamba: Peptide Toxicity Prediction using Mamba-based Multi-Scale Architecture},
  author={},
  journal={},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE)
