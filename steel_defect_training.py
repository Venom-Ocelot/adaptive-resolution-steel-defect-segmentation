#!/usr/bin/env python3
"""Steel defect training, converted from the supplied notebook.
Run: python -u steel_defect_training.py --data-dir ~/finalproject --output-dir runs/python_run
Dependencies: torch torchvision segmentation-models-pytorch pandas numpy pillow scikit-learn matplotlib
"""
import argparse
import os
import sys
import traceback
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams:
            stream.flush()

def display(value):
    print(value.to_string() if hasattr(value, "to_string") else value, flush=True)

plot_number = 0
def save_plot():
    import matplotlib.pyplot as plt
    global plot_number
    plot_number += 1
    name = f"plot_{plot_number:02d}.png"
    plt.savefig(name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {name}", flush=True)

def main(args, data_dir):
    import pandas as pd #
    import torch #
    from torch.utils.data import DataLoader, Dataset, Subset
    from torchvision import transforms
    import torch.nn.functional as F
    import segmentation_models_pytorch as smp #
    from PIL import Image #
    import numpy as np #
    from pathlib import Path #
    from sklearn.model_selection import train_test_split #

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA GPU unavailable. Submit this script to a GPU node, or pass --allow-cpu.")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else DEVICE, flush=True)


    df = pd.read_csv(data_dir / "train.csv")

    display(df.head())
    print("Defect classes:", df["ClassId"].unique())


    def decode_rle(rle, height=256, width=1600):
        mask = np.zeros(height * width, dtype=np.uint8)

        if pd.isna(rle) or not str(rle).strip():
            return mask.reshape((height, width), order="F")

        values = np.asarray(str(rle).split(), dtype=int)
        starts = values[::2] - 1
        lengths = values[1::2]

        for start, length in zip(starts, lengths):
            mask[start:start + length] = 1

        return mask.reshape((height, width), order="F")

    class SteelDataset(Dataset):
        def __init__(self, image_dir, df, resolution="low"):
            self.image_paths = sorted(Path(image_dir).glob("*.jpg"))
            self.annotations = {
                name: group
                for name, group in df.groupby("ImageId")
            }

            if resolution not in ("low", "high"):
                raise ValueError("resolution must be 'low' or 'high'")

            self.size = (128, 800) if resolution == "low" else (256, 1600)

            self.image_transform = transforms.Compose([
                transforms.Resize(self.size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            self.mask_transform = transforms.Resize(
                self.size,
                interpolation=transforms.InterpolationMode.NEAREST
            )

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            path = self.image_paths[idx]

            with Image.open(path) as img:
                image = self.image_transform(img.convert("RGB"))

            mask = np.zeros((256, 1600), dtype=np.uint8)

            if path.name in self.annotations:
                for _, row in self.annotations[path.name].iterrows():
                    mask = np.maximum(
                        mask,
                        decode_rle(row["EncodedPixels"])
                    )

            mask = torch.from_numpy(mask.copy()).unsqueeze(0).float()
            mask = self.mask_transform(mask)

            return image, mask

    IMAGE_DIR = data_dir / "train_images"

    lowres_dataset = SteelDataset(
        IMAGE_DIR, df, resolution="low"
    )

    highres_dataset = SteelDataset(
        IMAGE_DIR, df, resolution="high"
    )

    lowres_loader = DataLoader(
        lowres_dataset,
        batch_size=8,
        shuffle=True
    )

    highres_loader = DataLoader(
        highres_dataset,
        batch_size=2,
        shuffle=True
    )

    # Test one batch
    images, masks = next(iter(lowres_loader))

    print("Images:", images.shape)
    print("Masks:", masks.shape)

    lowres_model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    ).to(DEVICE)

    highres_model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    ).to(DEVICE)

    import matplotlib.pyplot as plt
    import numpy as np

    # Find an image containing a defect
    idx = next(
        i for i, path in enumerate(highres_dataset.image_paths)
        if path.name in highres_dataset.annotations
        and highres_dataset.annotations[path.name]["EncodedPixels"].notna().any()
    )

    image, mask = highres_dataset[idx]

    # Undo ImageNet normalization for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = (image.cpu() * std + mean).clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    mask = mask.squeeze(0).cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(16, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Steel Image")

    axes[1].imshow(image)
    axes[1].imshow(
        np.ma.masked_where(mask == 0, mask),
        cmap="Reds",
        alpha=0.5,
        vmin=0,
        vmax=1
    )
    axes[1].set_title("Ground-Truth Defect Overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    save_plot()

    SEED = 42

    indices = np.arange(len(lowres_dataset))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=SEED,
        shuffle=True
    )

    # Low-resolution datasets
    lowres_train = Subset(lowres_dataset, train_idx)
    lowres_val = Subset(lowres_dataset, val_idx)

    # High-resolution datasets
    highres_train = Subset(highres_dataset, train_idx)
    highres_val = Subset(highres_dataset, val_idx)

    # DataLoaders
    lowres_train_loader = DataLoader(
        lowres_train, batch_size=8, shuffle=True
    )

    lowres_val_loader = DataLoader(
        lowres_val, batch_size=8, shuffle=False
    )

    highres_train_loader = DataLoader(
        highres_train, batch_size=2, shuffle=True
    )

    highres_val_loader = DataLoader(
        highres_val, batch_size=2, shuffle=False
    )

    print(f"Training images: {len(train_idx)}")
    print(f"Validation images: {len(val_idx)}")

    def dice_loss(logits, targets, smooth=1.0):
        probs = torch.sigmoid(logits)

        intersection = (probs * targets).sum(dim=(1, 2, 3))
        denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

        dice = (2 * intersection + smooth) / (denominator + smooth)

        return 1 - dice.mean()


    def segmentation_loss(logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = dice_loss(logits, targets)

        return bce + dice


    def segmentation_metrics(logits, targets, threshold=0.5):
        predictions = (torch.sigmoid(logits) >= threshold).bool()
        targets = targets.bool()

        tp = (predictions & targets).sum().item()
        fp = (predictions & ~targets).sum().item()
        fn = (~predictions & targets).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

        return {
            "precision": precision,
            "recall": recall,
            "dice": dice
        }

    lowres_model.eval()

    images, masks = next(iter(lowres_val_loader))

    images = images.to(DEVICE)
    masks = masks.to(DEVICE)

    with torch.no_grad():
        logits = lowres_model(images)

    loss = segmentation_loss(logits, masks)
    metrics = segmentation_metrics(logits, masks)

    print("Loss:", loss.item())
    print("Metrics:", metrics)

    from torch.optim import AdamW

    optimizer = AdamW(
        lowres_model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    def train_epoch(model, loader, optimizer):
        model.train()
        total_loss = 0.0

        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)
            loss = segmentation_loss(logits, masks)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def validate_epoch(model, loader):
        model.eval()

        total_loss = 0.0
        tp = fp = fn = 0

        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)
            loss = segmentation_loss(logits, masks)

            total_loss += loss.item() * images.size(0)

            predictions = torch.sigmoid(logits) >= 0.5
            targets = masks.bool()

            tp += (predictions & targets).sum().item()
            fp += (predictions & ~targets).sum().item()
            fn += (~predictions & targets).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

        return {
            "loss": total_loss / len(loader.dataset),
            "precision": precision,
            "recall": recall,
            "dice": dice
        }

    EPOCHS = args.epochs

    history = []

    best_dice = -1.0

    for epoch in range(EPOCHS):
        train_loss = train_epoch(
            lowres_model,
            lowres_train_loader,
            optimizer
        )

        metrics = validate_epoch(
            lowres_model,
            lowres_val_loader
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **metrics
        })

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {metrics['loss']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"Dice: {metrics['dice']:.4f}"
        )

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]

            torch.save(
                lowres_model.state_dict(),
                "lowres_best.pth"
            )

    history_df = pd.DataFrame(history)
    history_df.to_csv("lowres_history.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Training")
    axes[0].plot(history_df["epoch"], history_df["loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["precision"], label="Precision")
    axes[1].plot(history_df["epoch"], history_df["recall"], label="Recall")
    axes[1].plot(history_df["epoch"], history_df["dice"], label="Dice")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    save_plot()

    lowres_model.load_state_dict(
        torch.load(
            "lowres_best.pth",
            map_location=DEVICE,
            weights_only=True
        )
    )

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    results = []

    for threshold in thresholds:
        tp = fp = fn = 0

        lowres_model.eval()

        with torch.no_grad():
            for images, masks in lowres_val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                logits = lowres_model(images)
                predictions = torch.sigmoid(logits) >= threshold
                targets = masks.bool()

                tp += (predictions & targets).sum().item()
                fp += (predictions & ~targets).sum().item()
                fn += (~predictions & targets).sum().item()

        results.append({
            "threshold": threshold,
            "precision": tp / (tp + fp + 1e-8),
            "recall": tp / (tp + fn + 1e-8),
            "dice": 2 * tp / (2 * tp + fp + fn + 1e-8)
        })

    threshold_df = pd.DataFrame(results)
    threshold_df.to_csv("threshold_metrics.csv", index=False)
    display(threshold_df)

    lowres_model.eval()

    images, masks = next(iter(lowres_val_loader))

    with torch.no_grad():
        logits = lowres_model(images.to(DEVICE))
        probabilities = torch.sigmoid(logits).cpu()

    fig, axes = plt.subplots(4, 1, figsize=(16, 9))

    image = images[0].cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = (image * std + mean).clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()

    axes[0].imshow(image)
    axes[0].set_title("Original")

    axes[1].imshow(masks[0, 0].numpy(), cmap="gray")
    axes[1].set_title("Ground Truth")

    for ax, threshold in zip(axes[2:], [0.1, 0.5]):
        prediction = probabilities[0, 0] >= threshold
        ax.imshow(prediction.numpy(), cmap="gray")
        ax.set_title(f"Prediction — Threshold {threshold}")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    save_plot()

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    threshold = 0.1
    tile_size = 128

    probability = probabilities[0, 0].numpy()
    selected_tiles = []

    height, width = probability.shape

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.imshow(image)

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = probability[y:y+tile_size, x:x+tile_size]

            if (tile >= threshold).any():
                selected_tiles.append((x, y))

                rect = patches.Rectangle(
                    (x, y),
                    min(tile_size, width - x),
                    min(tile_size, height - y),
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none"
                )
                ax.add_patch(rect)

    ax.set_title(f"Selected Candidate Tiles: {len(selected_tiles)}")
    ax.axis("off")
    save_plot()

    total_tiles = (
        ((height + tile_size - 1) // tile_size)
        * ((width + tile_size - 1) // tile_size)
    )

    print(f"Selected tiles: {len(selected_tiles)}/{total_tiles}")
    print(f"Tile selection rate: {len(selected_tiles)/total_tiles:.2%}")

    lowres_model.eval()

    images, masks = next(iter(lowres_val_loader))

    images = images.to(DEVICE)
    masks = masks.to(DEVICE)

    with torch.no_grad():
        logits = lowres_model(images)

    loss = segmentation_loss(logits, masks)
    metrics = segmentation_metrics(logits, masks)

    print("Loss:", loss.item())
    print("Metrics:", metrics)

    highres_optimizer = AdamW(
        highres_model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    EPOCHS = args.epochs

    highres_history = []

    highres_best_dice = -1.0

    for epoch in range(EPOCHS):
        train_loss = train_epoch(
            highres_model,
            highres_train_loader,
            highres_optimizer
        )

        metrics = validate_epoch(
            highres_model,
            highres_val_loader
        )

        highres_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **metrics
        })

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {metrics['loss']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"Dice: {metrics['dice']:.4f}"
        )

        if metrics["dice"] > highres_best_dice:
            highres_best_dice = metrics["dice"]

            torch.save(
                highres_model.state_dict(),
                "highres_best.pth"
            )

    highres_history_df = pd.DataFrame(highres_history)
    highres_history_df.to_csv("highres_history.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(highres_history_df["epoch"], highres_history_df["train_loss"], label="Training")
    axes[0].plot(highres_history_df["epoch"], highres_history_df["loss"], label="Validation")
    axes[0].set_title("High-Resolution Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(highres_history_df["epoch"], highres_history_df["precision"], label="Precision")
    axes[1].plot(highres_history_df["epoch"], highres_history_df["recall"], label="Recall")
    axes[1].plot(highres_history_df["epoch"], highres_history_df["dice"], label="Dice")
    axes[1].set_title("High-Resolution Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    save_plot()

    highres_model.load_state_dict(
        torch.load(
            "highres_best.pth",
            map_location=DEVICE,
            weights_only=True
        )
    )

    highres_metrics = validate_epoch(
        highres_model,
        highres_val_loader
    )

    print("High-resolution best model validation metrics:", highres_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train low- and high-resolution steel defect models.")
    parser.add_argument("--data-dir", type=Path, default=Path.cwd(), help="Folder containing train.csv and train_images/")
    parser.add_argument("--output-dir", type=Path, default=Path("steel_outputs"))
    parser.add_argument("--epochs", type=int, default=10, help="Epochs for each model")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)
    with open("training.log", "a", buffering=1) as log, open("errors.log", "a", buffering=1) as errors:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(original_stdout, log)
        sys.stderr = Tee(original_stderr, errors)
        try:
            if not (data_dir / "train.csv").is_file():
                raise FileNotFoundError(data_dir / "train.csv")
            if not any((data_dir / "train_images").glob("*.jpg")):
                raise FileNotFoundError(f"No JPG images found in {data_dir / 'train_images'}")
            print("Data:", data_dir)
            print("Outputs:", output_dir)
            main(args, data_dir)
            print("Training completed successfully.")
        except Exception:
            traceback.print_exc()
            sys.exit(1)
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
