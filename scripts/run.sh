#!/bin/bash
# =============================================================================
# ToxMamba - Training & Evaluation Script
# =============================================================================

set -e

SCRIPT="src/main.py"
TOML_PATH="configs/config.toml"
SEED=42
MODEL="S_Pretrain_emb_Mamba_ISM"
EMBED_TYPE="pretrain_ISM"

echo "============================================"
echo "  ToxMamba - Peptide Toxicity Prediction"
echo "  Model:     ${MODEL}"
echo "  Embedding: ${EMBED_TYPE}"
echo "  Config:    ${TOML_PATH}"
echo "  Seed:      ${SEED}"
echo "============================================"
echo ""

python -W ignore ${SCRIPT} \
    --model-name ${MODEL} \
    --embed-type ${EMBED_TYPE} \
    --tomlpath ${TOML_PATH} \
    --seed ${SEED}

echo ""
echo "============================================"
echo "  Pipeline completed successfully!"
echo "============================================"
