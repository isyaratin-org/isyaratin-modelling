"""
Evaluasi Ensemble Multi-Seed untuk Model BISINDO Manual.

Memuat beberapa checkpoint LSTM (seed berbeda), menjalankan inferensi video consensus
pada masing-masing, lalu menggabungkan probabilitas (rata-rata) untuk keputusan akhir.
Mendukung Test-Time Augmentation (flip horizontal) opsional.

Cara pakai:
  uv run python evaluate_ensemble.py --models models/manual_lstm_best.pt models/manual_lstm_seed43.pt models/manual_lstm_seed44.pt
  uv run python evaluate_ensemble.py --models ... --tta
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from holistic_extractor import EXTENDED_FEAT_DIM, FEAT_DIM, compute_extended_features, flip_horizontal_landmarks

CACHE_FILE = "data/manual_holistic_lstm_cache.npz"
LABELS_FILE = "labels/manual_labels.json"


class BisindoLSTMClassifier(nn.Module):
    """Arsitektur BiLSTM 2-layer identik dengan train-new.py (untuk load checkpoint)."""

    def __init__(self, feat_dim, num_classes, hidden1=128, hidden2=96, head_dim=192, dropout=0.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.input_norm = nn.LayerNorm(feat_dim)
        self.lstm1 = nn.LSTM(feat_dim, hidden1, num_layers=1, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1 * 2, hidden2, num_layers=1, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(dropout)
        pooled_dim = hidden2 * 4
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, num_classes),
        )

    def forward(self, x):
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        x = self.input_norm(x)
        h1, _ = self.lstm1(x)
        h1 = self.drop1(h1)
        h2, _ = self.lstm2(h1)
        h2 = self.drop2(h2)
        mean_pool = h2.mean(dim=1)
        max_pool = h2.amax(dim=1)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        return self.head(pooled)


def load_model(model_path, device):
    """Muat checkpoint LSTM dan kembalikan (model, meta)."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    meta = ckpt.get("meta", {})
    state_dict = ckpt.get("ema_state_dict") or ckpt.get("model_state_dict")
    arch = meta.get("arch", "mgn")
    feat_dim = int(meta.get("feat_dim", EXTENDED_FEAT_DIM if arch == "mgn" else FEAT_DIM))
    num_classes = int(meta.get("num_classes", 20))
    hidden1 = int(meta.get("hidden1", 128))
    hidden2 = int(meta.get("hidden2", 96))
    head_dim = int(meta.get("head_dim", 192))

    if arch == "mgn":
        raise NotImplementedError("Ensemble saat ini hanya mendukung arsitektur LSTM.")

    model = BisindoLSTMClassifier(
        feat_dim=feat_dim,
        num_classes=num_classes,
        hidden1=hidden1,
        hidden2=hidden2,
        head_dim=head_dim,
        dropout=0.0,
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, meta


def predict_video(model, raw_v, device, seq_len, feat_dim, tta):
    """Prediksi probabilitas konsensus untuk satu video."""
    t = raw_v.shape[0]
    if t < seq_len:
        sub_idx = np.linspace(0, t - 1, seq_len).round().astype(np.int64)
        windows = [raw_v[sub_idx]]
    else:
        stride = 4
        starts = list(range(0, t - seq_len + 1, stride))
        if (t - seq_len) not in starts:
            starts.append(t - seq_len)
        windows = [raw_v[s : s + seq_len] for s in starts]
        global_sub = np.linspace(0, t - 1, seq_len).round().astype(np.int64)
        windows.append(raw_v[global_sub])

    windows_arr = np.stack(windows)
    if tta:
        windows_arr = np.concatenate([windows_arr, flip_horizontal_landmarks(windows_arr)], axis=0)
    if feat_dim == EXTENDED_FEAT_DIM:
        windows_arr = compute_extended_features(windows_arr)

    with torch.no_grad():
        x = torch.from_numpy(windows_arr).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).mean(dim=0).cpu().numpy()
    return probs


def main():
    parser = argparse.ArgumentParser(description="Evaluasi Ensemble Multi-Seed BISINDO Manual")
    parser.add_argument("--models", nargs="+", required=True, help="Daftar path checkpoint .pt")
    parser.add_argument("--cache", type=str, default=CACHE_FILE)
    parser.add_argument("--tta", action="store_true", help="Aktifkan Test-Time Augmentation")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")

    # Muat cache
    cached = np.load(args.cache, allow_pickle=True)
    seqs = cached["sequences"]
    lengths = cached["lengths"]
    labels = cached["labels"]
    filenames = cached["filenames"].tolist() if "filenames" in cached else [f"v{i}" for i in range(len(labels))]

    class_names = []
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            class_names = [str(x) for x in json.load(f)]
    else:
        class_names = cached["class_names"].tolist()

    # Muat semua model
    models = []
    for mp in args.models:
        print(f"Memuat model: {mp}")
        model, meta = load_model(mp, device)
        models.append((model, meta))

    # Inferensi per model
    all_probs = []  # (n_models, n_videos, n_classes)
    for model, meta in models:
        seq_len = int(meta.get("seq_len", 32))
        feat_dim = int(meta.get("feat_dim", FEAT_DIM))
        probs_list = []
        for i in range(len(labels)):
            raw_v = seqs[i, : lengths[i]]
            p = predict_video(model, raw_v, device, seq_len, feat_dim, args.tta)
            probs_list.append(p)
        all_probs.append(np.stack(probs_list))
        print(f"  Model selesai ({len(labels)} video)")

    all_probs = np.stack(all_probs)  # (n_models, n_videos, n_classes)
    ensemble_probs = all_probs.mean(axis=0)  # rata-rata antar model

    y_pred = ensemble_probs.argmax(axis=-1)
    y_true = labels

    acc = float((y_pred == y_true).mean())
    print("\n" + "=" * 60)
    print(f"ENSEMBLE ({len(models)} model, TTA={'ON' if args.tta else 'OFF'})")
    print(f"  Top-1 Accuracy : {acc * 100:.2f}%")
    print(f"  Total Error    : {int((y_pred != y_true).sum())} / {len(y_true)}")
    print("=" * 60)

    # Per-model accuracy untuk perbandingan
    print("\nAkurasi per model:")
    for idx, (model, meta) in enumerate(models):
        m_pred = all_probs[idx].argmax(axis=-1)
        m_acc = float((m_pred == y_true).mean())
        print(f"  Model {idx} ({os.path.basename(args.models[idx])}): {m_acc * 100:.2f}%")

    # Confusion pairs ensemble
    print("\nPasangan tertukar (ensemble):")
    for i in range(len(y_true)):
        if y_pred[i] != y_true[i]:
            print(f"  {class_names[y_true[i]]:20s} -> {class_names[y_pred[i]]:20s} ({filenames[i]})")


if __name__ == "__main__":
    main()
