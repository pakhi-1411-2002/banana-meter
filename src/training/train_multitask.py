from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

from src.data.dataloaders import get_dataloaders
from src.models.multitask_resnet import BananaMultiTaskResNet


# =========================
# Config
# =========================
CSV_PATH = "data/metadata/labels.csv"
BATCH_SIZE = 8
EPOCHS_STAGE_1 = 8   # frozen backbone
EPOCHS_STAGE_2 = 6   # unfrozen fine-tuning
LEARNING_RATE_HEADS = 1e-3
LEARNING_RATE_FULL = 1e-4
NUM_CLASSES = 5
ALPHA = 2.0   # classification loss weight
BETA = 1.0    # regression loss weight

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = Path("models")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = CHECKPOINT_DIR / "multitask_resnet18_best.pth"


# =========================
# Loss and eval
# =========================
def multitask_loss(
    stage_logits: torch.Tensor,
    stage_targets: torch.Tensor,
    days_pred: torch.Tensor,
    days_targets: torch.Tensor,
    ce_loss_fn: nn.Module,
    reg_loss_fn: nn.Module,
    alpha: float = 1.0,
    beta: float = 0.5,
):
    cls_loss = ce_loss_fn(stage_logits, stage_targets)
    reg_loss = reg_loss_fn(days_pred, days_targets)
    total = alpha * cls_loss + beta * reg_loss
    return total, cls_loss, reg_loss


def evaluate(model, loader, ce_loss_fn, reg_loss_fn):
    model.eval()

    total_losses = []
    cls_losses = []
    reg_losses = []

    all_stage_preds = []
    all_stage_targets = []
    all_days_preds = []
    all_days_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(DEVICE)
            stage_targets = batch["stage"].to(DEVICE)
            days_targets = batch["days_left"].to(DEVICE)

            stage_logits, days_pred = model(images)

            total_loss, cls_loss, reg_loss = multitask_loss(
                stage_logits=stage_logits,
                stage_targets=stage_targets,
                days_pred=days_pred,
                days_targets=days_targets,
                ce_loss_fn=ce_loss_fn,
                reg_loss_fn=reg_loss_fn,
                alpha=ALPHA,
                beta=BETA,
            )

            total_losses.append(total_loss.item())
            cls_losses.append(cls_loss.item())
            reg_losses.append(reg_loss.item())

            stage_pred_classes = stage_logits.argmax(dim=1).cpu().numpy()
            all_stage_preds.extend(stage_pred_classes.tolist())
            all_stage_targets.extend(stage_targets.cpu().numpy().tolist())

            # clamp to valid range for reporting only
            days_pred_np = torch.clamp(days_pred, min=0.0, max=7.0).cpu().numpy()
            all_days_preds.extend(days_pred_np.tolist())
            all_days_targets.extend(days_targets.cpu().numpy().tolist())

    acc = accuracy_score(all_stage_targets, all_stage_preds)
    f1 = f1_score(all_stage_targets, all_stage_preds, average="macro")
    mae = mean_absolute_error(all_days_targets, all_days_preds)
    rmse = np.sqrt(mean_squared_error(all_days_targets, all_days_preds))

    return {
        "total_loss": float(np.mean(total_losses)),
        "cls_loss": float(np.mean(cls_losses)),
        "reg_loss": float(np.mean(reg_losses)),
        "acc": float(acc),
        "f1": float(f1),
        "mae": float(mae),
        "rmse": float(rmse),
    }


def train_one_epoch(model, loader, optimizer, ce_loss_fn, reg_loss_fn):
    model.train()

    total_losses = []
    cls_losses = []
    reg_losses = []

    for batch in loader:
        images = batch["image"].to(DEVICE)
        stage_targets = batch["stage"].to(DEVICE)
        days_targets = batch["days_left"].to(DEVICE)

        optimizer.zero_grad()

        stage_logits, days_pred = model(images)

        total_loss, cls_loss, reg_loss = multitask_loss(
            stage_logits=stage_logits,
            stage_targets=stage_targets,
            days_pred=days_pred,
            days_targets=days_targets,
            ce_loss_fn=ce_loss_fn,
            reg_loss_fn=reg_loss_fn,
            alpha=ALPHA,
            beta=BETA,
        )

        total_loss.backward()
        optimizer.step()

        total_losses.append(total_loss.item())
        cls_losses.append(cls_loss.item())
        reg_losses.append(reg_loss.item())

    return {
        "total_loss": float(np.mean(total_losses)),
        "cls_loss": float(np.mean(cls_losses)),
        "reg_loss": float(np.mean(reg_losses)),
    }


def save_checkpoint(model, path: Path):
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path: Path):
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model


def print_metrics(prefix: str, metrics: dict):
    print(
        f"{prefix} | "
        f"Total Loss: {metrics['total_loss']:.4f} | "
        f"Cls Loss: {metrics['cls_loss']:.4f} | "
        f"Reg Loss: {metrics['reg_loss']:.4f} | "
        f"Acc: {metrics['acc']:.4f} | "
        f"F1: {metrics['f1']:.4f} | "
        f"MAE: {metrics['mae']:.4f} | "
        f"RMSE: {metrics['rmse']:.4f}"
    )


def main():
    print("Using device:", DEVICE)

    train_loader, val_loader, test_loader = get_dataloaders(
        csv_path=CSV_PATH,
        batch_size=BATCH_SIZE,
    )

    model = BananaMultiTaskResNet(
        num_classes=NUM_CLASSES,
        pretrained=True,
        freeze_backbone=True,
        dropout=0.3,
    ).to(DEVICE)

    ce_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.SmoothL1Loss()

    # -------------------------
    # Stage 1: train heads only
    # -------------------------
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE_HEADS,
    )

    best_val_mae = float("inf")

    print("\n=== Stage 1: training heads with frozen backbone ===")
    for epoch in range(1, EPOCHS_STAGE_1 + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, ce_loss_fn, reg_loss_fn)
        val_metrics = evaluate(model, val_loader, ce_loss_fn, reg_loss_fn)

        print(
            f"Epoch {epoch}/{EPOCHS_STAGE_1} | "
            f"Train Loss: {train_metrics['total_loss']:.4f} | "
            f"Val Loss: {val_metrics['total_loss']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f}"
        )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            save_checkpoint(model, BEST_MODEL_PATH)
            print(f"Saved best checkpoint to {BEST_MODEL_PATH}")

    # -------------------------
    # Stage 2: unfreeze and fine-tune
    # -------------------------
    model.unfreeze_backbone()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_FULL)

    print("\n=== Stage 2: fine-tuning full model ===")
    for epoch in range(1, EPOCHS_STAGE_2 + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, ce_loss_fn, reg_loss_fn)
        val_metrics = evaluate(model, val_loader, ce_loss_fn, reg_loss_fn)

        print(
            f"Fine-tune Epoch {epoch}/{EPOCHS_STAGE_2} | "
            f"Train Loss: {train_metrics['total_loss']:.4f} | "
            f"Val Loss: {val_metrics['total_loss']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f}"
        )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            save_checkpoint(model, BEST_MODEL_PATH)
            print(f"Saved best checkpoint to {BEST_MODEL_PATH}")

    # -------------------------
    # Final test evaluation
    # -------------------------
    print("\nLoading best model for final test...")
    best_model = BananaMultiTaskResNet(
        num_classes=NUM_CLASSES,
        pretrained=False,
        freeze_backbone=False,
        dropout=0.3,
    ).to(DEVICE)
    best_model = load_checkpoint(best_model, BEST_MODEL_PATH)

    test_metrics = evaluate(best_model, test_loader, ce_loss_fn, reg_loss_fn)
    print_metrics("TEST", test_metrics)


if __name__ == "__main__":
    main()