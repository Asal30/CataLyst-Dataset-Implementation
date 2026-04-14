import os
import glob
import time
import random
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.models as models

from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

"""# `02. Config`"""

CONFIG = {
    "csv_path": "data/processed/final_training_dataset.csv",
    "base_dir": "data",
    "checkpoint_dir": "data/saved_models/checkpoints/",
    "best_model_dir": "data/saved_models/best_models/",
    "batch_size": 32,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "epochs": 30,
    "img_size": 224,
    "num_workers": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "presence_loss_weight": 1.0,
    "concept_loss_weight": 0.5,
    "severity_loss_weight": 0.8,
    "val_size": 0.2,
    "random_state": 42,
    "use_amp": torch.cuda.is_available(),
    "early_stopping_patience": 10,
    "path_cols": ["relative_path", "file_path", "image_path", "path"],
    "concept_cols": {
        "NO": ["NO", "NO_pseudo"],
        "NC": ["NC", "NC_pseudo"],
        "CO": ["CO", "CO_pseudo", "C", "C_pseudo"],
        "PSC": ["PSC", "PSC_pseudo", "P", "P_pseudo"],
    },
    "presence_cols": ["binary_cataract_label", "presence_label", "presence", "is_cataract"],
    "severity_cols": ["severity", "severity_label", "path_case_label", "case_label"],
    "severity_classes": ["normal", "immature", "mild", "moderate", "mature", "cataract"],
}

CONCEPT_NAMES = ["NO", "NC", "CO", "PSC"]
SEVERITY_CLASSES = CONFIG["severity_classes"]
SEVERITY_TO_INDEX = {name: idx for idx, name in enumerate(SEVERITY_CLASSES)}

"""# 03. Utils"""

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dirs():
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["best_model_dir"], exist_ok=True)

def find_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None

def safe_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default

def normalize_text_label(value) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {"non_cataract": "normal", "healthy": "normal", "clear": "normal", "severe": "mature"}
    return aliases.get(text, text)

def score_0_5_to_0_1(x: float) -> float:
    x = safe_float(x, default=np.nan)
    if np.isnan(x):
        return np.nan
    return np.clip(x / 5.0, 0.0, 1.0)

def infer_presence_from_concepts(concept_values_0_1: np.ndarray) -> float:
    if np.all(np.isnan(concept_values_0_1)):
        return 0.0
    valid = concept_values_0_1[~np.isnan(concept_values_0_1)]
    if len(valid) == 0:
        return 0.0
    return 1.0 if np.max(valid) >= 0.20 else 0.0

def parse_presence_value(value, concept_values_0_1: np.ndarray) -> float:
    if pd.isna(value):
        return infer_presence_from_concepts(concept_values_0_1)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(1.0 if float(value) >= 0.5 else 0.0)
    text = normalize_text_label(value)
    if text in {"cataract", "yes", "positive", "present", "true"}:
        return 1.0
    if text in {"non_cataract", "normal", "immature", "no", "negative", "absent", "false"}:
        return 0.0
    return infer_presence_from_concepts(concept_values_0_1)

def parse_severity_value(value) -> int:
    if pd.isna(value):
        return -100
    text = normalize_text_label(value)
    if text is None:
        return -100
    if text in SEVERITY_TO_INDEX:
        return SEVERITY_TO_INDEX[text]
    return -100

"""# 04. Dataset"""

class CataractDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, base_dir: str = ""):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform
        self.base_dir = base_dir
        self.path_col = find_first_existing_column(self.df, CONFIG["path_cols"])
        if self.path_col is None:
            raise ValueError(f"No valid image path column found. Expected one of: {CONFIG['path_cols']}")
        self.concept_cols = {
            concept_name: find_first_existing_column(self.df, candidates)
            for concept_name, candidates in CONFIG["concept_cols"].items()
        }
        missing_concepts = [k for k, v in self.concept_cols.items() if v is None]
        if missing_concepts:
            raise ValueError(f"Missing required concept columns for: {missing_concepts}")
        self.presence_col = find_first_existing_column(self.df, CONFIG["presence_cols"])
        self.severity_col = find_first_existing_column(self.df, CONFIG["severity_cols"])
        if self.severity_col is None:
            raise ValueError(f"No severity column found. Expected one of: {CONFIG['severity_cols']}")

    def __len__(self):
        return len(self.df)

    def _resolve_image_path(self, raw_path: str) -> str:
        raw_path = str(raw_path).replace("\\", "/")
        if os.path.isabs(raw_path):
            return raw_path
        return os.path.join(self.base_dir,"data", raw_path)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(row[self.path_col])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        concept_values = []
        concept_mask = []
        for concept_name in CONCEPT_NAMES:
            col = self.concept_cols[concept_name]
            raw_val = safe_float(row[col], default=np.nan)
            if np.isnan(raw_val):
                concept_values.append(0.0)
                concept_mask.append(1.0 * 0.0)
            else:
                concept_values.append(score_0_5_to_0_1(raw_val))
                concept_mask.append(1.0)
        concept_values = np.array(concept_values, dtype=np.float32)
        concept_mask = np.array(concept_mask, dtype=np.float32)

        presence_raw = row[self.presence_col] if self.presence_col is not None else np.nan
        presence_target = parse_presence_value(presence_raw, concept_values)
        severity_raw = row[self.severity_col]
        severity_target = parse_severity_value(severity_raw)

        labels = {
            "concepts": torch.tensor(concept_values, dtype=torch.float32),
            "concept_mask": torch.tensor(concept_mask, dtype=torch.float32),
            "presence": torch.tensor([presence_target], dtype=torch.float32),
            "severity": torch.tensor(severity_target, dtype=torch.long),
        }
        return image, labels

"""# 05. Transforms"""

train_tf = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(8),
    transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
])

"""# 06. Model"""

class CBMModel(nn.Module):
    def __init__(self, num_severity_classes: int):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.shared = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.concept_head = nn.Linear(128, 4)
        self.presence_head = nn.Linear(128, 1)
        self.severity_head = nn.Linear(128, num_severity_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        shared = self.shared(x)
        concepts = torch.sigmoid(self.concept_head(shared))
        presence_logits = self.presence_head(shared)
        severity_logits = self.severity_head(shared)
        return concepts, presence_logits, severity_logits

"""# 07. Loss + Metrics"""

def masked_smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss_per_item = F.smooth_l1_loss(pred, target, reduction="none")
    masked_loss = loss_per_item * mask
    denom = mask.sum().clamp(min=1.0)
    return masked_loss.sum() / denom

def compute_mae_masked(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> float:
    abs_err = torch.abs(pred - true) * mask
    denom = mask.sum().clamp(min=1.0)
    return (abs_err.sum() / denom).item()

def compute_spearman_masked(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> float:
    pred_np = pred.detach().cpu().numpy()
    true_np = true.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    corrs = []
    for i in range(pred_np.shape[1]):
        valid = mask_np[:, i] > 0.5
        if valid.sum() < 2:
            continue
        corr, _ = spearmanr(pred_np[valid, i], true_np[valid, i])
        if not np.isnan(corr):
            corrs.append(corr)
    return float(np.mean(corrs)) if corrs else 0.0

def compute_presence_accuracy(pred_presence_logits: torch.Tensor, true_presence: torch.Tensor) -> float:
    pred_prob = torch.sigmoid(pred_presence_logits)
    pred_bin = (pred_prob >= 0.5).float()
    return pred_bin.eq(true_presence).float().mean().item()

def compute_severity_metrics(pred_severity_logits: torch.Tensor, true_severity: torch.Tensor) -> Tuple[float, float]:
    valid = true_severity != -100
    if valid.sum().item() == 0:
        return 0.0, 0.0
    pred_cls = torch.argmax(pred_severity_logits[valid], dim=1).detach().cpu().numpy()
    true_cls = true_severity[valid].detach().cpu().numpy()
    acc = float((pred_cls == true_cls).mean())
    f1 = float(f1_score(true_cls, pred_cls, average="macro"))
    return acc, f1

def compute_total_loss(pred_concepts, true_concepts, concept_mask, pred_presence_logits, true_presence, pred_severity_logits, true_severity):
    concept_loss = masked_smooth_l1_loss(pred_concepts, true_concepts, concept_mask)
    presence_loss = F.binary_cross_entropy_with_logits(pred_presence_logits, true_presence)
    severity_loss = F.cross_entropy(pred_severity_logits, true_severity, ignore_index=-100)
    total_loss = (
        CONFIG["concept_loss_weight"] * concept_loss
        + CONFIG["presence_loss_weight"] * presence_loss
        + CONFIG["severity_loss_weight"] * severity_loss
    )
    return total_loss, {
        "concept_loss": float(concept_loss.item()),
        "presence_loss": float(presence_loss.item()),
        "severity_loss": float(severity_loss.item()),
        "total_loss": float(total_loss.item()),
    }

"""# 08. Train & Validate"""

def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    running_total = running_concept = running_presence = running_severity = 0.0
    for images, labels in loader:
        images = images.to(device)
        true_concepts = labels["concepts"].to(device)
        concept_mask = labels["concept_mask"].to(device)
        true_presence = labels["presence"].to(device)
        true_severity = labels["severity"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=CONFIG["use_amp"]):
            pred_concepts, pred_presence_logits, pred_severity_logits = model(images)
            total_loss, loss_items = compute_total_loss(
                pred_concepts, true_concepts, concept_mask,
                pred_presence_logits, true_presence,
                pred_severity_logits, true_severity
            )
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_total += loss_items["total_loss"]
        running_concept += loss_items["concept_loss"]
        running_presence += loss_items["presence_loss"]
        running_severity += loss_items["severity_loss"]
    n = max(len(loader), 1)
    return {
        "loss": running_total / n,
        "concept_loss": running_concept / n,
        "presence_loss": running_presence / n,
        "severity_loss": running_severity / n,
    }

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    running_total = running_concept = running_presence = running_severity = 0.0
    all_pred_concepts = []; all_true_concepts = []; all_masks = []
    all_pred_presence_logits = []; all_true_presence = []
    all_pred_severity_logits = []; all_true_severity = []

    for images, labels in loader:
        images = images.to(device)
        true_concepts = labels["concepts"].to(device)
        concept_mask = labels["concept_mask"].to(device)
        true_presence = labels["presence"].to(device)
        true_severity = labels["severity"].to(device)
        pred_concepts, pred_presence_logits, pred_severity_logits = model(images)
        total_loss, loss_items = compute_total_loss(
            pred_concepts, true_concepts, concept_mask,
            pred_presence_logits, true_presence,
            pred_severity_logits, true_severity
        )
        running_total += loss_items["total_loss"]
        running_concept += loss_items["concept_loss"]
        running_presence += loss_items["presence_loss"]
        running_severity += loss_items["severity_loss"]
        all_pred_concepts.append(pred_concepts)
        all_true_concepts.append(true_concepts)
        all_masks.append(concept_mask)
        all_pred_presence_logits.append(pred_presence_logits)
        all_true_presence.append(true_presence)
        all_pred_severity_logits.append(pred_severity_logits)
        all_true_severity.append(true_severity)

    n = max(len(loader), 1)
    all_pred_concepts = torch.cat(all_pred_concepts, dim=0)
    all_true_concepts = torch.cat(all_true_concepts, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    all_pred_presence_logits = torch.cat(all_pred_presence_logits, dim=0)
    all_true_presence = torch.cat(all_true_presence, dim=0)
    all_pred_severity_logits = torch.cat(all_pred_severity_logits, dim=0)
    all_true_severity = torch.cat(all_true_severity, dim=0)

    return {
        "loss": running_total / n,
        "concept_loss": running_concept / n,
        "presence_loss": running_presence / n,
        "severity_loss": running_severity / n,
        "mae": compute_mae_masked(all_pred_concepts, all_true_concepts, all_masks),
        "spearman": compute_spearman_masked(all_pred_concepts, all_true_concepts, all_masks),
        "presence_acc": compute_presence_accuracy(all_pred_presence_logits, all_true_presence),
        "severity_acc": compute_severity_metrics(all_pred_severity_logits, all_true_severity)[0],
        "severity_f1": compute_severity_metrics(all_pred_severity_logits, all_true_severity)[1],
    }

"""# 09. Chekpoint Helpers"""

def save_checkpoint(path: str, payload: dict):
    torch.save(payload, path)

def load_latest_checkpoint(model, optimizer, device):
    checkpoint_files = sorted(glob.glob(os.path.join(CONFIG["checkpoint_dir"], "checkpoint_epoch_*.pth")))
    if not checkpoint_files:
        print("No checkpoints found. Starting fresh.")
        return model, optimizer, 0, float("inf")
    latest_ckpt_path = checkpoint_files[-1]
    print(f"Found latest checkpoint: {latest_ckpt_path}")
    ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = ckpt["epoch"]
    best_loss = ckpt.get("best_loss", ckpt.get("val_loss", float("inf")))
    print(f"Resumed from epoch {start_epoch} | best_loss={best_loss:.4f}")
    return model, optimizer, start_epoch, best_loss

"""# 10. Main Method"""

def main():
    set_seed(CONFIG["random_state"])
    ensure_dirs()
    print("Loading dataset...")
    df = pd.read_csv(CONFIG["csv_path"])
    severity_col = find_first_existing_column(df, CONFIG["severity_cols"])
    if severity_col is None:
        raise ValueError(f"No severity column found in CSV. Expected one of: {CONFIG['severity_cols']}")

    concept_candidates = {name: find_first_existing_column(df, CONFIG["concept_cols"][name]) for name in CONCEPT_NAMES}
    presence_col = find_first_existing_column(df, CONFIG["presence_cols"])

    def row_presence_target(row):
        vals = []
        for k in CONCEPT_NAMES:
            v = score_0_5_to_0_1(row[concept_candidates[k]])
            vals.append(v)
        vals = np.array(vals, dtype=np.float32)
        raw_presence = row[presence_col] if presence_col is not None else np.nan
        return parse_presence_value(raw_presence, vals)

    df["_presence_target"] = df.apply(row_presence_target, axis=1)
    df["_severity_target"] = df[severity_col].map(parse_severity_value)
    df["_stratify_label"] = df.apply(lambda r: f"{int(r['_presence_target'])}_{int(r['_severity_target'])}" if int(r["_severity_target"]) != -100 else f"{int(r['_presence_target'])}_missing", axis=1)

    try:
        train_df, val_df = train_test_split(df, test_size=CONFIG["val_size"], random_state=CONFIG["random_state"], shuffle=True, stratify=df["_stratify_label"])
    except Exception:
        train_df, val_df = train_test_split(df, test_size=CONFIG["val_size"], random_state=CONFIG["random_state"], shuffle=True)

    print(f"Train size: {len(train_df)}")
    print(f"Val size:   {len(val_df)}")
    print(f"Severity column used: {severity_col}")

    train_ds = CataractDataset(train_df, transform=train_tf, base_dir=CONFIG["base_dir"])
    val_ds = CataractDataset(val_df, transform=val_tf, base_dir=CONFIG["base_dir"])

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=torch.cuda.is_available())

    model = CBMModel(num_severity_classes=len(SEVERITY_CLASSES)).to(CONFIG["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=CONFIG["use_amp"])
    stable_best_path = os.path.join(CONFIG["best_model_dir"], "best_model.pth")

    model, optimizer, start_epoch, best_loss = load_latest_checkpoint(model, optimizer, CONFIG["device"])
    if os.path.exists(stable_best_path):
        try:
            best_ckpt = torch.load(stable_best_path, map_location=CONFIG["device"], weights_only=False)
            best_loss = min(best_loss, best_ckpt.get("val_loss", float("inf")))
            print(f"Loaded stable best reference: best_loss={best_loss:.4f}")
        except Exception as e:
            print(f"Could not load stable best model pointer: {e}")

    end_epoch = start_epoch + CONFIG["epochs"]
    print(f"Starting training from epoch {start_epoch + 1} to {end_epoch}...")
    total_start = time.time()
    no_improve_count = 0

    for epoch in range(start_epoch, end_epoch):
        epoch_start = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, CONFIG["device"])
        val_metrics = validate(model, val_loader, CONFIG["device"])
        scheduler.step(val_metrics["loss"])
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch + 1}/{end_epoch}")
        print(f"LR: {current_lr:.6f}")
        print(f"Train | loss={train_metrics['loss']:.4f} | concept={train_metrics['concept_loss']:.4f} | presence={train_metrics['presence_loss']:.4f} | severity={train_metrics['severity_loss']:.4f}")
        print(f"Val   | loss={val_metrics['loss']:.4f} | concept={val_metrics['concept_loss']:.4f} | presence={val_metrics['presence_loss']:.4f} | severity={val_metrics['severity_loss']:.4f} | mae={val_metrics['mae']:.4f} | spearman={val_metrics['spearman']:.4f} | presence_acc={val_metrics['presence_acc']:.4f} | severity_acc={val_metrics['severity_acc']:.4f} | severity_f1={val_metrics['severity_f1']:.4f}")
        print(f"Epoch time: {epoch_time:.2f}s")

        checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], f"checkpoint_epoch_{epoch + 1:03d}.pth")
        checkpoint_payload = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_spearman": val_metrics["spearman"],
            "val_presence_acc": val_metrics["presence_acc"],
            "val_severity_acc": val_metrics["severity_acc"],
            "val_severity_f1": val_metrics["severity_f1"],
            "best_loss": best_loss,
            "config": CONFIG,
            "severity_classes": SEVERITY_CLASSES,
            "notes": "Weakly supervised concept-inspired cataract model with presence + severity learning.",
        }
        save_checkpoint(checkpoint_path, checkpoint_payload)
        print(f"Checkpoint saved -> {checkpoint_path}")

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            no_improve_count = 0
            best_payload = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_spearman": val_metrics["spearman"],
                "val_presence_acc": val_metrics["presence_acc"],
                "val_severity_acc": val_metrics["severity_acc"],
                "val_severity_f1": val_metrics["severity_f1"],
                "config": CONFIG,
                "severity_classes": SEVERITY_CLASSES,
                "notes": "Best weakly supervised concept-inspired cataract model with severity learning.",
            }
            save_checkpoint(stable_best_path, best_payload)
            print(f"Best model updated -> {stable_best_path} | val_loss={best_loss:.4f}")
        else:
            no_improve_count += 1
            print(f"No improvement count: {no_improve_count}/{CONFIG['early_stopping_patience']}")
        if no_improve_count >= CONFIG["early_stopping_patience"]:
            print("Early stopping triggered.")
            break

    total_time = time.time() - total_start
    print(f"\nTraining complete.")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Total training time: {total_time:.2f}s")

if __name__ == "__main__":
    main()