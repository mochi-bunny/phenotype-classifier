#!/usr/bin/env bash
# setup_nebius.sh — One-shot Nebius GPU VM setup for phenotype model training.
#
# Usage:
#   ssh user@<nebius-vm-ip> 'bash -s' < setup_nebius.sh
#   OR
#   scp setup_nebius.sh user@<nebius-vm-ip>:~ && ssh user@<nebius-vm-ip> ./setup_nebius.sh
#
# Prerequisites:
#   - Nebius VM with GPU (RTX PRO 6000 or similar)
#   - Ubuntu 24.04 + CUDA pre-installed (Nebius default images)

set -euo pipefail

PROJECT_DIR="$HOME/phenotype_training"
DATA_DIR="$PROJECT_DIR/data"
MODEL_DIR="$PROJECT_DIR/models"

echo "========================================="
echo " Nebius Phenotype Model Training Setup"
echo "========================================="

# 1. System deps
echo "[1/6] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv unzip wget git

# 2. Python venv
echo "[2/6] Setting up Python environment..."
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
    onnx onnxruntime-gpu \
    Pillow \
    transformers \
    kagglehub \
    fastapi uvicorn pydantic httpx

# 3. Download PlantVillage dataset (tomato subset)
echo "[3/6] Downloading PlantVillage tomato dataset..."
mkdir -p "$DATA_DIR/plantvillage_raw"

python3 -c "
import kagglehub
path = kagglehub.dataset_download('abdallahalidev/plantvillage-dataset')
print(f'Downloaded to: {path}')

import shutil, os
src = os.path.join(path, 'plantvillage dataset', 'color')
# Copy only tomato classes
for d in os.listdir(src):
    if 'Tomato' in d or 'tomato' in d:
        shutil.copytree(os.path.join(src, d), os.path.join('$DATA_DIR/plantvillage_raw', d), dirs_exist_ok=True)
        print(f'  Copied: {d}')
"

echo "[INFO] Dataset contents:"
ls -la "$DATA_DIR/plantvillage_raw/"
find "$DATA_DIR/plantvillage_raw" -name "*.jpg" -o -name "*.JPG" -o -name "*.png" | wc -l
echo " total images"

# 4. Copy training scripts
echo "[4/6] Setting up training scripts..."
# If running via SSH pipe, scripts should be in the same directory.
# Otherwise clone from your repo:
# git clone https://github.com/<your-repo>/nebius.git "$PROJECT_DIR/repo"
# cp "$PROJECT_DIR/repo/training/"*.py "$PROJECT_DIR/"

# 5. Train the model
echo "[5/6] Training phenotype detection model..."
mkdir -p "$MODEL_DIR"

python3 train.py \
    --data_dir "$DATA_DIR" \
    --remap_from "$DATA_DIR/plantvillage_raw" \
    --output_dir "$MODEL_DIR" \
    --epochs 15 \
    --batch_size 32 \
    --lr 1e-4 \
    --arch resnet50

echo "[INFO] Training complete. Model artifacts:"
ls -la "$MODEL_DIR/"

# 6. (Optional) Generate Cosmos synthetic data and retrain
# Uncomment if Cosmos Transfer is available on this VM:
#
# echo "[6/6] Generating Cosmos synthetic data..."
# pip install cosmos-transfer2
# python3 cosmos_datagen.py \
#     --seed_dir "$DATA_DIR/plantvillage_raw" \
#     --output_dir "$DATA_DIR/cosmos_synthetic" \
#     --variants_per_image 10
#
# echo "Retraining with Cosmos-augmented dataset..."
# # Merge original + synthetic
# mkdir -p "$DATA_DIR/combined"
# for cls in Healthy Stressed Diseased; do
#     mkdir -p "$DATA_DIR/combined/$cls"
#     cp "$DATA_DIR/remapped/$cls/"* "$DATA_DIR/combined/$cls/" 2>/dev/null || true
#     cp "$DATA_DIR/cosmos_synthetic/$cls/"* "$DATA_DIR/combined/$cls/" 2>/dev/null || true
# done
#
# python3 train.py \
#     --data_dir "$DATA_DIR/combined" \
#     --output_dir "$MODEL_DIR" \
#     --epochs 20 \
#     --batch_size 32

echo ""
echo "========================================="
echo " DONE. Model ready at:"
echo "   ONNX:  $MODEL_DIR/phenotype_model.onnx"
echo "   Meta:  $MODEL_DIR/training_meta.json"
echo ""
echo " Next steps:"
echo "   1. scp $MODEL_DIR/phenotype_model.onnx local:inference/"
echo "   2. scp $MODEL_DIR/training_meta.json local:inference/"
echo "   3. Run the FastAPI server locally"
echo "========================================="
