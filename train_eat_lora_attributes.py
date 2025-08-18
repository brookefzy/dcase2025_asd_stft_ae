import argparse
import os
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

import librosa

try:
    from peft import LoraConfig, get_peft_model
except ImportError as e:  # pragma: no cover - optional dependency
    LoraConfig = None
    get_peft_model = None


def load_eat_model(num_attributes: int) -> nn.Module:
    """Load the pre-trained EAT model and replace the head for attribute prediction."""
    # Download and load model from the official repository via torch.hub
    model = torch.hub.load("cwx-worst-one/EAT", "eat_small", pretrained=True)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_attributes)
    return model


class AttributeDataset(Dataset):
    """Dataset returning log-mel spectrograms and their attribute labels."""

    def __init__(self, root: Path, csv_path: Path, sr: int = 16000, n_mels: int = 128):
        self.root = Path(root)
        self.df = pd.read_csv(csv_path)
        if "filename" not in self.df.columns:
            self.df.rename(columns={self.df.columns[0]: "filename"}, inplace=True)
        self.attr_cols = [c for c in self.df.columns if c != "filename"]
        self.sr = sr
        self.n_mels = n_mels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = self.root / row["filename"]
        wav, _ = librosa.load(audio_path, sr=self.sr)
        mel = librosa.feature.melspectrogram(wav, sr=self.sr, n_mels=self.n_mels)
        mel = librosa.power_to_db(mel, ref=np.max)
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(row[self.attr_cols].values.astype(np.float32))
        return mel_tensor, label, row["filename"]


def apply_lora(model: nn.Module, r: int = 4, alpha: int = 16, dropout: float = 0.1) -> nn.Module:
    """Wrap the model with LoRA layers using peft."""
    if LoraConfig is None or get_peft_model is None:
        raise ImportError("peft is required for LoRA fine-tuning. Please install it via pip install peft")
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["qkv", "proj"],
        task_type="FEATURE_EXTRACTION",
    )
    return get_peft_model(model, lora_cfg)


def split_dataset(dataset: Dataset, train_ratio: float = 0.8):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split = int(len(indices) * train_ratio)
    return Subset(dataset, indices[:split]), Subset(dataset, indices[split:])


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    preds = {}
    with torch.no_grad():
        for x, _, fnames in loader:
            x = x.to(device)
            out = model(x)
            preds.update({fn: out[i].cpu().numpy() for i, fn in enumerate(fnames)})
    return preds


def main():
    parser = argparse.ArgumentParser(description="Fine-tune EAT with LoRA for attribute prediction")
    parser.add_argument("--data-root", type=str, default="data/dcase2025t2/dev_data/raw")
    parser.add_argument("--machine-type", type=str, required=True, help="Machine type folder name")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="results/eat_lora")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    machine_dir = Path(args.data_root) / args.machine_type
    attr_csv = machine_dir / "attributes_00.csv"

    dataset = AttributeDataset(machine_dir, attr_csv)
    num_attrs = len(dataset.attr_cols)
    train_ds, val_ds = split_dataset(dataset)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = load_eat_model(num_attrs)
    model = apply_lora(model)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch}: train loss={loss:.4f}")

    preds = evaluate(model, DataLoader(dataset, batch_size=args.batch_size), device)
    df_out = pd.DataFrame.from_dict(preds, orient="index", columns=dataset.attr_cols)
    df_out.index.name = "filename"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.machine_type}_pred.csv"
    df_out.to_csv(out_csv)
    print(f"Predictions saved to {out_csv}")


if __name__ == "__main__":
    main()
