"""
Phenotype detection model training on Nebius GPU.

Trains a ResNet-50 (or ViT-small) classifier on PlantVillage tomato images
(optionally augmented with Cosmos-generated synthetic data) for 3-class
phenotype classification: Healthy / Stressed / Diseased.

Run on Nebius GPU:
    python train.py --data_dir data/training --epochs 15 --batch_size 32

Output:
    models/phenotype_model.onnx   — ONNX model for inference
    models/phenotype_model.pth    — PyTorch checkpoint
    models/class_names.json       — Class name mapping
    models/training_meta.json     — Training metadata (for Learner's Permit model_hash)
"""

import os
import json
import hashlib
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models


# 3-class mapping: PlantVillage tomato classes → phenotype classes
PLANTVILLAGE_TO_PHENOTYPE = {
    "Tomato___healthy": "Healthy",
    "Tomato_healthy": "Healthy",
    "Healthy": "Healthy",
    "Tomato___Early_blight": "Stressed",
    "Tomato___Leaf_Mold": "Stressed",
    "Tomato___Spider_mites_Two_spotted_spider_mite": "Stressed",
    "Tomato_Early_blight": "Stressed",
    "Tomato_Leaf_Mold": "Stressed",
    "Stressed": "Stressed",
    "Tomato___Bacterial_spot": "Diseased",
    "Tomato___Late_blight": "Diseased",
    "Tomato___Septoria_leaf_spot": "Diseased",
    "Tomato___Target_Spot": "Diseased",
    "Tomato___Tomato_mosaic_virus": "Diseased",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Diseased",
    "Tomato_Bacterial_spot": "Diseased",
    "Tomato_Late_blight": "Diseased",
    "Diseased": "Diseased",
}

PHENOTYPE_CLASSES = ["Diseased", "Healthy", "Stressed"]


def remap_dataset(source_dir: str, target_dir: str):
    """Remap PlantVillage class folders into 3 phenotype classes."""
    source = Path(source_dir)
    target = Path(target_dir)

    if target.exists() and any(target.iterdir()):
        print(f"[INFO] Remapped dataset already exists at {target}")
        return

    for cls in PHENOTYPE_CLASSES:
        (target / cls).mkdir(parents=True, exist_ok=True)

    count = {c: 0 for c in PHENOTYPE_CLASSES}
    for class_dir in sorted(source.iterdir()):
        if not class_dir.is_dir():
            continue
        phenotype_class = PLANTVILLAGE_TO_PHENOTYPE.get(class_dir.name)
        if phenotype_class is None:
            print(f"[SKIP] Unknown class: {class_dir.name}")
            continue

        for img in class_dir.glob("*"):
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                dest = target / phenotype_class / f"{class_dir.name}_{img.name}"
                if not dest.exists():
                    os.symlink(img.resolve(), dest)
                count[phenotype_class] += 1

    print(f"[INFO] Remapped dataset: {count}")


def get_transforms(train: bool = True):
    """Image transforms for training and validation."""
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_model(num_classes: int = 3, arch: str = "resnet50") -> nn.Module:
    """Build a pretrained model with a new classification head."""
    if arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "vit_small":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model


def compute_model_hash(model_path: str) -> str:
    """Compute SHA-256 hash of the model file for Learner's Permit scope tracking."""
    sha256 = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def export_onnx(model, output_path: str, device: str = "cpu"):
    """Export trained model to ONNX format."""
    model = model.to(device).eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
    )
    print(f"[INFO] ONNX model exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train phenotype detection model")
    parser.add_argument("--data_dir", required=True, help="Path to PlantVillage or Cosmos-augmented dataset")
    parser.add_argument("--output_dir", default="models", help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--arch", default="resnet50", choices=["resnet50", "vit_small"])
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--remap_from", default=None, help="If set, remap PlantVillage classes from this dir first")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Remap PlantVillage classes to 3 phenotype classes if needed
    training_dir = args.data_dir
    if args.remap_from:
        training_dir = str(Path(args.data_dir) / "remapped")
        remap_dataset(args.remap_from, training_dir)

    # Load dataset
    print(f"[INFO] Loading dataset from {training_dir}")
    full_dataset = datasets.ImageFolder(training_dir, transform=get_transforms(train=True))

    # Verify classes match expected
    print(f"[INFO] Classes found: {full_dataset.classes}")
    assert set(full_dataset.classes) == set(PHENOTYPE_CLASSES), (
        f"Expected {PHENOTYPE_CLASSES}, got {full_dataset.classes}"
    )

    # Train/val split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    # Use val transforms for val set
    val_dataset = datasets.ImageFolder(training_dir, transform=get_transforms(train=False))
    val_indices = val_set.indices
    val_set_proper = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set_proper, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"[INFO] Train: {train_size}, Val: {val_size}")

    # Build model
    model = build_model(num_classes=len(PHENOTYPE_CLASSES), arch=args.arch).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_acc = 0.0
    training_start = time.time()

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"  Epoch {epoch + 1:2d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path / "phenotype_model.pth")
            print(f"    [SAVE] New best val accuracy: {val_acc:.4f}")

    training_time = time.time() - training_start
    print(f"\n[INFO] Training complete in {training_time:.1f}s. Best val acc: {best_val_acc:.4f}")

    # Load best model and export
    model.load_state_dict(torch.load(output_path / "phenotype_model.pth", weights_only=True))

    onnx_path = str(output_path / "phenotype_model.onnx")
    export_onnx(model, onnx_path)

    model_hash = compute_model_hash(onnx_path)

    # Save class names and metadata
    class_names_path = output_path / "class_names.json"
    with open(class_names_path, "w") as f:
        json.dump(PHENOTYPE_CLASSES, f)

    meta = {
        "model_hash": model_hash,
        "architecture": args.arch,
        "num_classes": len(PHENOTYPE_CLASSES),
        "class_names": PHENOTYPE_CLASSES,
        "best_val_accuracy": best_val_acc,
        "epochs": args.epochs,
        "training_time_seconds": training_time,
        "dataset_dir": args.data_dir,
        "input_size": [3, 224, 224],
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    }
    with open(output_path / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] Model hash: {model_hash}")
    print(f"  ONNX:   {onnx_path}")
    print(f"  Meta:   {output_path / 'training_meta.json'}")
    print(f"  Classes: {PHENOTYPE_CLASSES}")


if __name__ == "__main__":
    main()
