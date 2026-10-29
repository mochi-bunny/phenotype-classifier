# Dockerfile — Nebius Serverless Job for phenotype model training
#
# Build:
#   docker build -t phenotype-training:latest -f training/Dockerfile .
#
# Run locally (if GPU available):
#   docker run --gpus all -v $(pwd)/models:/workspace/models phenotype-training:latest
#
# Push to Nebius Container Registry and submit as Serverless Job

FROM nvcr.io/nvidia/pytorch:24.07-py3

WORKDIR /workspace

# Install additional deps
RUN pip install --no-cache-dir \
    onnx \
    onnxruntime-gpu \
    kagglehub \
    transformers \
    Pillow

# Copy training scripts
COPY training/train.py .
COPY training/cosmos_datagen.py .

# Download PlantVillage at build time (cached in image)
RUN python3 -c "\
import kagglehub, shutil, os; \
path = kagglehub.dataset_download('abdallahalidev/plantvillage-dataset'); \
src = os.path.join(path, 'plantvillage dataset', 'color'); \
os.makedirs('data/plantvillage_raw', exist_ok=True); \
[shutil.copytree(os.path.join(src, d), os.path.join('data/plantvillage_raw', d), dirs_exist_ok=True) \
 for d in os.listdir(src) if 'Tomato' in d or 'tomato' in d]"

# Default: train the model
CMD ["python3", "train.py", \
     "--data_dir", "data", \
     "--remap_from", "data/plantvillage_raw", \
     "--output_dir", "models", \
     "--epochs", "15", \
     "--batch_size", "32", \
     "--arch", "resnet50"]
