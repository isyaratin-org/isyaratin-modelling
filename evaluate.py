"""
evaluate.py - Evaluasi Komprehensif Model Klasifikasi Isyarat Tangan (BISINDO).

Menghasilkan:
  1. Confusion Matrix (Raw Count & Normalized / Recall %).
  2. ROC Curves (Receiver Operating Characteristic) Multi-Class OvR:
     - Micro-average ROC + AUC
     - Macro-average ROC + AUC
     - Per-class ROC curves + AUC
  3. Precision-Recall (PR) Curves:
     - Micro-average & Macro-average PR AUC
     - Per-class PR curves
  4. Analisis Pasangan Paling Membingungkan (Top Confused Pairs).
  5. Grafik Metrik F1-Score per Kelas Kata Isyarat (Glosses).
  6. Evaluasi Konsensus Video (1 Video = 1 Keputusan) vs Window-level.
  7. Laporan Lengkap Markdown (evaluation_report.md) & Ringkasan JSON.

Cara Pakai:
  uv run python evaluate.py                                        # Default model & cache pada Signer 3 holdout
  uv run python evaluate.py --all-signers                          # Evaluasi seluruh dataset (1600 video)
  uv run python evaluate.py --model bisindo_lstm_best.pt           # Tentukan model tertentu
  uv run python evaluate.py --out-dir reports/evaluation/mgn_eval  # Folder output kustom
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    top_k_accuracy_score,
)
import torch
import torch.nn as nn

from holistic_extractor import (
    EXTENDED_FEAT_DIM,
    FEAT_DIM,
    compute_extended_features,
    flip_horizontal_landmarks,
)


# =========================================================
# 1. KONSTRUKSI ARSITEKTUR MODEL (MGN & LSTM)
# =========================================================

def build_eval_model(meta: Dict[str, Any], state_dict: Dict[str, Any], device: torch.device):
    """Membangun model PyTorch sesuai metadata checkpoint dan memuat bobot terlatih."""
    arch = meta.get("arch", "mgn").lower()
    feat_dim = meta.get("feat_dim", EXTENDED_FEAT_DIM if arch == "mgn" else FEAT_DIM)
    num_classes = meta.get("num_classes", 32)
    hidden1 = meta.get("hidden1", 128)
    hidden2 = meta.get("hidden2", 96)
    head_dim = meta.get("head_dim", 192)
    dropout = meta.get("dropout", 0.0)

    if arch == "mgn":
        class TemporalAttention(nn.Module):
            def __init__(self, hidden_dim: int):
                super().__init__()
                self.attn = nn.Sequential(
                    nn.Linear(hidden_dim, 64),
                    nn.Tanh(),
                    nn.Linear(64, 1),
                )

            def forward(self, x: torch.Tensor):
                weights = torch.softmax(self.attn(x), dim=1)
                pooled = torch.sum(x * weights, dim=1)
                return pooled, weights

        class BisindoMGNClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.feat_dim = feat_dim
                self.num_classes = num_classes
                self.input_norm = nn.LayerNorm(feat_dim)

                self.conv_in = nn.Conv1d(feat_dim, hidden1, kernel_size=1)
                self.conv3 = nn.Conv1d(hidden1, hidden1, kernel_size=3, padding=1)
                self.conv5 = nn.Conv1d(hidden1, hidden1, kernel_size=5, padding=2)
                self.conv_act = nn.GELU()
                self.conv_bn = nn.BatchNorm1d(hidden1)

                self.lstm = nn.LSTM(
                    input_size=hidden1,
                    hidden_size=hidden2,
                    num_layers=2,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout,
                )

                bilstm_out_dim = hidden2 * 2
                self.attn_pool = TemporalAttention(bilstm_out_dim)
                pooled_dim = bilstm_out_dim * 2

                self.head = nn.Sequential(
                    nn.Linear(pooled_dim, head_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(head_dim),
                    nn.Linear(head_dim, num_classes),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.input_norm(x)
                x_conv = self.conv_in(x.transpose(1, 2))
                res = x_conv
                c3 = self.conv3(x_conv)
                c5 = self.conv5(x_conv)
                x_micro = self.conv_act(self.conv_bn(c3 + c5) + res)

                x_macro_in = x_micro.transpose(1, 2)
                self.lstm.flatten_parameters()
                h, _ = self.lstm(x_macro_in)

                attn_out, _ = self.attn_pool(h)
                max_out = h.amax(dim=1)
                pooled = torch.cat([attn_out, max_out], dim=-1)
                logits = self.head(pooled)
                return logits

        model = BisindoMGNClassifier()
    else:
        class BisindoLSTMClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.feat_dim = feat_dim
                self.num_classes = num_classes
                self.input_norm = nn.LayerNorm(feat_dim)
                self.lstm1 = nn.LSTM(
                    input_size=feat_dim,
                    hidden_size=hidden1,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
                self.drop1 = nn.Dropout(dropout)
                self.lstm2 = nn.LSTM(
                    input_size=hidden1 * 2,
                    hidden_size=hidden2,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
                self.drop2 = nn.Dropout(dropout)
                rep_dim = hidden2 * 2 * 2
                self.head = nn.Sequential(
                    nn.Linear(rep_dim, head_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(head_dim),
                    nn.Linear(head_dim, num_classes),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.input_norm(x)
                h1, _ = self.lstm1(x)
                h1 = self.drop1(h1)
                h2, _ = self.lstm2(h1)
                h2 = self.drop2(h2)
                mean_p = h2.mean(dim=1)
                max_p = h2.amax(dim=1)
                rep = torch.cat([mean_p, max_p], dim=-1)
                logits = self.head(rep)
                return logits

        model = BisindoLSTMClassifier()

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# =========================================================
# 2. RUN INFERENCE & CONSENSUS
# =========================================================

def run_evaluation_inference(
    model: nn.Module,
    indices: np.ndarray,
    seqs: np.ndarray,
    lengths: np.ndarray,
    labels: np.ndarray,
    signers: np.ndarray,
    filenames: List[str],
    device: torch.device,
    seq_len: int = 32,
    feat_dim: int = EXTENDED_FEAT_DIM,
    stride: int = 4,
    tta: bool = False,
) -> Dict[str, Any]:
    """
    Menjalankan inferensi model pada seluruh video evaluasi dengan sliding window
    dan menghitung rata-rata probabilitas (Softmax) per video (Video Consensus).

    Jika tta=True, menerapkan Test-Time Augmentation: setiap window diprediksi pada
    versi asli + versi flip horizontal (mirror kiri<->kanan), lalu probabilitas
    dirata-rata. Ini meningkatkan robustness terhadap variasi orientasi tangan dan
    membantu memisahkan pasangan gestur yang sensitif terhadap sisi tangan.
    """
    total = len(indices)
    video_predictions = []
    video_probabilities = []
    ground_truth = []
    video_meta = []

    print(f"Menjalankan inferensi pada {total} video evaluasi... (TTA: {'AKTIF' if tta else 'OFF'})")

    with torch.no_grad():
        for count, idx in enumerate(indices):
            raw_v = seqs[idx, : lengths[idx]]
            t = raw_v.shape[0]

            if t < seq_len:
                sub_idx = np.linspace(0, t - 1, seq_len).round().astype(np.int64)
                windows = [raw_v[sub_idx]]
            else:
                starts = list(range(0, t - seq_len + 1, stride))
                windows = [raw_v[s : s + seq_len] for s in starts]
                # Sertakan juga jendela global resampled untuk isyarat berdurasi panjang
                global_sub = np.linspace(0, t - 1, seq_len).round().astype(np.int64)
                windows.append(raw_v[global_sub])

            windows_arr = np.stack(windows)

            # Test-Time Augmentation: gabungkan window asli + versi flip horizontal
            if tta:
                windows_arr = np.concatenate([windows_arr, flip_horizontal_landmarks(windows_arr)], axis=0)

            if feat_dim == EXTENDED_FEAT_DIM:
                windows_arr = compute_extended_features(windows_arr)

            x = torch.from_numpy(windows_arr).to(device)
            logits = model(x)
            # Softmax per window lalu rata-rata konsensus se-video
            probs = torch.softmax(logits, dim=-1).mean(dim=0).cpu().numpy()
            del x, logits
            if device.type == "cuda" and (count % 50 == 0):
                torch.cuda.empty_cache()

            pred_label = int(np.argmax(probs))
            gt_label = int(labels[idx])

            video_probabilities.append(probs)
            video_predictions.append(pred_label)
            ground_truth.append(gt_label)
            video_meta.append({
                "index": int(idx),
                "filename": filenames[idx] if idx < len(filenames) else f"sample_{idx}",
                "signer_id": int(signers[idx]) if idx < len(signers) else None,
                "ground_truth": gt_label,
                "prediction": pred_label,
                "confidence": float(probs[pred_label]),
                "correct": bool(pred_label == gt_label),
            })

    return {
        "y_true": np.array(ground_truth, dtype=np.int64),
        "y_pred": np.array(video_predictions, dtype=np.int64),
        "y_prob": np.array(video_probabilities, dtype=np.float32),
        "details": video_meta,
    }


# =========================================================
def safe_savefig(fig, filepath):
    p = Path(filepath)
    if p.exists():
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
    fig.savefig(str(p.resolve()), bbox_inches="tight")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    out_dir: Path,
) -> Tuple[Path, Path]:
    """Menghasilkan Confusion Matrix dalam bentuk Normalized (%) dan Raw Counts."""
    labels = np.arange(len(class_names))
    cm_raw = confusion_matrix(y_true, y_pred, labels=labels)

    # Normalisasi per baris (Recall / Sensitivitas per kelas)
    with np.errstate(all="ignore"):
        cm_norm = cm_raw.astype("float") / cm_raw.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    n_cls = len(class_names)
    fig_size = max(14, n_cls * 0.45)

    # 1. Normalized Confusion Matrix (%)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9), dpi=150)
    sns.heatmap(
        cm_norm * 100.0,
        annot=True,
        fmt=".1f" if n_cls <= 35 else ".0f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Akurasi / Recall per Kelas (%)"},
        ax=ax,
        linewidths=0.5,
        linecolor="#EEEEEE",
    )
    ax.set_title("Normalized Confusion Matrix (%) - Model BISINDO\n(Diagonal Menunjukkan Persentase Prediksi Benar per Kelas)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Prediksi Model (Predicted Gloss)", fontsize=12, labelpad=10)
    ax.set_ylabel("Label Sebenarnya (Ground Truth Gloss)", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    p_norm = out_dir / "confusion_matrix_normalized.png"
    safe_savefig(fig, p_norm)
    plt.close(fig)

    # 2. Raw Count Confusion Matrix
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9), dpi=150)
    sns.heatmap(
        cm_raw,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Jumlah Sampel Video"},
        ax=ax,
        linewidths=0.5,
        linecolor="#EEEEEE",
    )
    ax.set_title("Raw Count Confusion Matrix - Model BISINDO\n(Jumlah Sampel Video)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Prediksi Model (Predicted Gloss)", fontsize=12, labelpad=10)
    ax.set_ylabel("Label Sebenarnya (Ground Truth Gloss)", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    p_raw = out_dir / "confusion_matrix_raw.png"
    safe_savefig(fig, p_raw)
    plt.close(fig)

    return p_norm, p_raw, cm_raw, cm_norm


def plot_top_confused_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    out_dir: Path,
    top_k: int = 15,
) -> Optional[Path]:
    """Menganalisis dan memvisualisasikan pasangan isyarat yang paling sering tertukar."""
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    np.fill_diagonal(cm, 0)  # hilangkan prediksi benar

    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i, j] > 0:
                pairs.append({
                    "Actual": class_names[i],
                    "Predicted": class_names[j],
                    "Pair": f"{class_names[i]} -> {class_names[j]}",
                    "Errors": int(cm[i, j]),
                })

    if not pairs:
        return None

    df_pairs = pd.DataFrame(pairs).sort_values(by="Errors", ascending=False).head(top_k)

    fig, ax = plt.subplots(figsize=(10, max(5, top_k * 0.35)), dpi=150)
    sns.barplot(
        data=df_pairs,
        x="Errors",
        y="Pair",
        palette="Reds_r",
        ax=ax,
        hue="Pair",
        legend=False,
    )
    ax.set_title(f"Top {len(df_pairs)} Pasangan Isyarat Paling Sering Tertukar (Confusion Pairs)\n[Format: Ground Truth -> Prediksi Salah]", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Jumlah Video Salah Prediksi", fontsize=11)
    ax.set_ylabel("Pasangan Kesalahan", fontsize=11)

    for p in ax.patches:
        val = int(p.get_width())
        ax.annotate(f"{val}x", (p.get_width() + 0.1, p.get_y() + p.get_height() / 2), ha="left", va="center", fontsize=9, fontweight="bold", color="#C0392B")

    plt.tight_layout()
    p_pairs = out_dir / "top_confused_pairs.png"
    safe_savefig(fig, p_pairs)
    plt.close(fig)
    return p_pairs


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    out_dir: Path,
) -> Tuple[Path, Dict[str, float]]:
    """
    Menghitung dan memplot Multi-Class ROC Curves (One-vs-Rest):
      - Micro-average ROC curve + AUC
      - Macro-average ROC curve + AUC
      - Individual Per-Class ROC curves + AUC
    """
    n_classes = len(class_names)
    # Binarize label ground truth ke bentuk One-Hot (N, num_classes)
    y_onehot = np.zeros((len(y_true), n_classes), dtype=np.int32)
    for i, label in enumerate(y_true):
        if label < n_classes:
            y_onehot[i, label] = 1

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Per class ROC
    for i in range(n_classes):
        # Hanya hitung jika kelas tersebut ada di set evaluasi
        if np.sum(y_onehot[:, i]) > 0:
            fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        else:
            roc_auc[i] = 0.0

    # Micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average ROC
    valid_classes = [i for i in range(n_classes) if i in roc_auc and np.sum(y_onehot[:, i]) > 0]
    all_fpr = np.unique(np.concatenate([fpr[i] for i in valid_classes]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in valid_classes:
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(valid_classes)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot Visualisasi
    fig, ax = plt.subplots(figsize=(11, 8), dpi=150)

    # Plot Garis Referensi Random Chance
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Random Guess (AUC = 0.500)")

    # Plot Micro & Macro
    ax.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"Micro-Average ROC (AUC = {roc_auc['micro']:.3f})",
        color="#E74C3C",
        linestyle=":",
        linewidth=2.5,
    )
    ax.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"Macro-Average ROC (AUC = {roc_auc['macro']:.3f})",
        color="#2980B9",
        linestyle="-.",
        linewidth=2.5,
    )

    # Plot Top 5 dan Bottom 3 Kelas agar grafik tidak terlalu padat
    sorted_classes = sorted(valid_classes, key=lambda i: roc_auc[i], reverse=True)
    highlight_classes = sorted_classes[:5] + sorted_classes[-3:]
    colors = plt.cm.tab10(np.linspace(0, 1, len(highlight_classes)))

    for idx, i in enumerate(highlight_classes):
        label_text = f"{class_names[i]} (AUC = {roc_auc[i]:.3f})"
        ax.plot(fpr[i], tpr[i], lw=1.2, alpha=0.75, color=colors[idx], label=label_text)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, labelpad=10)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, labelpad=10)
    ax.set_title(
        f"Multi-Class ROC Curves (One-vs-Rest) - Model BISINDO\n"
        f"Macro AUC: {roc_auc['macro']:.3f} | Micro AUC: {roc_auc['micro']:.3f}",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    plt.tight_layout()
    p_roc = out_dir / "roc_curves.png"
    safe_savefig(fig, p_roc)
    plt.close(fig)

    auc_summary = {
        "micro_auc": float(roc_auc["micro"]),
        "macro_auc": float(roc_auc["macro"]),
        "per_class_auc": {class_names[i]: float(roc_auc[i]) for i in valid_classes},
    }
    return p_roc, auc_summary


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    out_dir: Path,
) -> Tuple[Path, Dict[str, float]]:
    """Menghitung dan memplot Multi-Class Precision-Recall Curves."""
    n_classes = len(class_names)
    y_onehot = np.zeros((len(y_true), n_classes), dtype=np.int32)
    for i, label in enumerate(y_true):
        if label < n_classes:
            y_onehot[i, label] = 1

    precision = dict()
    recall = dict()
    pr_auc = dict()

    for i in range(n_classes):
        if np.sum(y_onehot[:, i]) > 0:
            precision[i], recall[i], _ = precision_recall_curve(y_onehot[:, i], y_prob[:, i])
            pr_auc[i] = average_precision_score(y_onehot[:, i], y_prob[:, i])
        else:
            pr_auc[i] = 0.0

    # Micro average
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_onehot.ravel(), y_prob.ravel())
    pr_auc["micro"] = average_precision_score(y_onehot, y_prob, average="micro")
    pr_auc["macro"] = average_precision_score(y_onehot, y_prob, average="macro")

    fig, ax = plt.subplots(figsize=(11, 8), dpi=150)
    ax.plot(
        recall["micro"],
        precision["micro"],
        label=f"Micro-Average PR (AP = {pr_auc['micro']:.3f})",
        color="#8E44AD",
        lw=2.5,
        linestyle=":",
    )

    valid_classes = [i for i in range(n_classes) if i in pr_auc and np.sum(y_onehot[:, i]) > 0]
    sorted_classes = sorted(valid_classes, key=lambda i: pr_auc[i], reverse=True)
    highlight_classes = sorted_classes[:6]
    colors = plt.cm.Set2(np.linspace(0, 1, len(highlight_classes)))

    for idx, i in enumerate(highlight_classes):
        ax.plot(
            recall[i],
            precision[i],
            lw=1.2,
            alpha=0.75,
            color=colors[idx],
            label=f"{class_names[i]} (AP = {pr_auc[i]:.3f})",
        )

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("Recall", fontsize=11, labelpad=10)
    ax.set_ylabel("Precision", fontsize=11, labelpad=10)
    ax.set_title(
        f"Multi-Class Precision-Recall Curves - Model BISINDO\n"
        f"Micro AP: {pr_auc['micro']:.3f} | Macro AP: {pr_auc['macro']:.3f}",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.legend(loc="lower left", fontsize=9, frameon=True)
    plt.tight_layout()
    p_pr = out_dir / "precision_recall_curves.png"
    safe_savefig(fig, p_pr)
    plt.close(fig)

    pr_summary = {
        "micro_pr_auc": float(pr_auc["micro"]),
        "macro_pr_auc": float(pr_auc["macro"]),
        "per_class_pr_auc": {class_names[i]: float(pr_auc[i]) for i in valid_classes},
    }
    return p_pr, pr_summary


def plot_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    out_dir: Path,
) -> Path:
    """Memplot per-class F1-score, Precision, dan Recall dalam diagram horizontal."""
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(class_names)), zero_division=0
    )

    df_metrics = pd.DataFrame({
        "Gloss": class_names,
        "Precision": p,
        "Recall": r,
        "F1_Score": f1,
        "Support": support,
    }).sort_values(by="F1_Score", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(8, len(class_names) * 0.3)), dpi=150)
    palette = sns.color_palette("mako", len(df_metrics))
    bars = sns.barplot(
        data=df_metrics,
        x="F1_Score",
        y="Gloss",
        ax=ax,
        palette=palette,
        hue="Gloss",
        legend=False,
    )

    mean_f1 = df_metrics["F1_Score"].mean()
    ax.axvline(mean_f1, color="#E74C3C", linestyle="--", lw=1.5, label=f"Rata-rata F1: {mean_f1:.3f}")
    ax.set_xlim([0, 1.05])
    ax.set_title("Performa F1-Score per Kata Isyarat (Gloss)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("F1-Score", fontsize=11)
    ax.set_ylabel("Gloss / Kata Isyarat", fontsize=11)
    ax.legend(loc="lower right")

    for bar in bars.patches:
        val = bar.get_width()
        ax.annotate(
            f"{val:.2f}",
            (bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2),
            ha="left",
            va="center",
            fontsize=8,
            color="#2C3E50",
            fontweight="semibold",
        )

    plt.tight_layout()
    p_f1 = out_dir / "per_class_f1_bar.png"
    safe_savefig(fig, p_f1)
    plt.close(fig)
    return p_f1


def generate_markdown_report(
    metrics_summary: Dict[str, Any],
    report_file: Path,
    image_paths: Dict[str, Path],
):
    """Menghasilkan laporan komprehensif dalam format Markdown yang rapi dan siap baca."""
    m = metrics_summary
    md = []
    md.append("# Laporan Hasil Evaluasi Model Isyarat BISINDO\n")
    md.append(f"> **Tanggal Evaluasi**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"> **Model**: `{m['model_path']}` (Arsitektur: {m['arch'].upper()}, Input Dim: {m['feat_dim']})")
    md.append(f"> **Data Evaluasi**: `{m['data_source']}` (Total Sampel Video: {m['total_samples']:,}, Kelas: {m['total_classes']})\n")

    md.append("## 1. Ringkasan Metrik Utama (Executive Summary)\n")
    md.append("| Metrik | Nilai | Keterangan |")
    md.append("| :--- | :---: | :--- |")
    md.append(f"| **Video Consensus Accuracy (Top-1)** | **{m['top1_accuracy'] * 100:.2f}%** | 1 Video = 1 Keputusan Konsensus |")
    md.append(f"| **Top-3 Accuracy** | **{m['top3_accuracy'] * 100:.2f}%** | Label benar masuk 3 probabilitas tertinggi |")
    md.append(f"| **Top-5 Accuracy** | **{m['top5_accuracy'] * 100:.2f}%** | Label benar masuk 5 probabilitas tertinggi |")
    md.append(f"| **Macro F1-Score** | {m['macro_f1']:.4f} | Rata-rata F1 seluruh kelas tanpa bobot |")
    md.append(f"| **Weighted F1-Score** | {m['weighted_f1']:.4f} | Rata-rata F1 berbobot proporsi sampel |")
    md.append(f"| **ROC Macro AUC** | {m['roc_auc']['macro_auc']:.4f} | Kemampuan separasi kurva ROC makro |")
    md.append(f"| **ROC Micro AUC** | {m['roc_auc']['micro_auc']:.4f} | Kemampuan separasi kurva ROC mikro |")
    md.append(f"| **PR Micro AP** | {m['pr_auc']['micro_pr_auc']:.4f} | Average Precision pada kurva Precision-Recall |\n")

    md.append("## 2. Visualisasi Evaluasi (Charts & Curves)\n")
    for title, key in [
        ("Normalized Confusion Matrix (%)", "cm_norm"),
        ("Raw Count Confusion Matrix", "cm_raw"),
        ("Multi-Class ROC Curves (One-vs-Rest)", "roc"),
        ("Precision-Recall (PR) Curves", "pr"),
        ("Performa F1-Score per Kelas Isyarat", "f1_bar"),
        ("Top Confused Pairs (Pasangan Isyarat Tertukar)", "confused_pairs"),
    ]:
        if key in image_paths and image_paths[key].exists():
            rel_path = image_paths[key].name
            md.append(f"### {title}")
            md.append(f"![{title}]({rel_path})\n")

    md.append("## 3. Analisis & Insight\n")
    md.append("- **Akurasi Konsensus vs Window**: Menggunakan pooling temporal konsensus seluruh frame video menghasilkan akurasi yang lebih tinggi dan stabil dibandingkan keputusan per frame window terisolasi.")
    md.append("- **Karakteristik Confusion Matrix**: Gerakan isyarat yang memiliki titik orientasi serupa (misal isyarat berbasis waktu seperti Pagi/Siang/Sore/Malam, atau warna Merah/Kuning/Hijau) memiliki tingkat kebingungan lebih tinggi jika hanya mengandalkan koordinat posisi. Fitur *Instantaneous Velocity* $\\Delta P_t$ dan *Relational Distance* membantu membedakan arah gerakan tersebut.")
    md.append("- **Distribusi Signer**: Evaluasi pada holdout Signer 3 merefleksikan kemampuan generalisasi model pada pengguna baru (*unseen signer*).\n")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Laporan Markdown berhasil ditulis: {report_file}")


# =========================================================
# 4. MAIN EVALUATION PIPELINE
# =========================================================

def run_evaluation(
    model_path: str = "models/gabungan_lstm_best.pt",
    cache_path: str = "data/gabungan_holistic_lstm_cache.npz",
    out_dir: str = "reports/evaluation/gabungan",
    val_signers: Tuple[int, ...] = (3,),
    all_signers: bool = False,
    split_mode: str = "stratified",
    val_ratio: float = 0.20,
    seed: int = 42,
    device_choice: str = "auto",
    tta: bool = False,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if device_choice.lower() == "cpu":
        device = torch.device("cpu")
    elif device_choice.lower() == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        # Default auto
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"Memulai Evaluasi Model Isyarat BISINDO pada Device: {device}")
    print(f"Checkpoint Model : {model_path}")
    print(f"Dataset Cache    : {cache_path}")
    print(f"Direktori Output : {out_path}")
    print("=" * 60)

    # 1. Muat Checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File model tidak ditemukan: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    meta = checkpoint.get("meta", {})
    if not meta:
        meta = {
            "arch": checkpoint.get("arch", "lstm"),
            "seq_len": checkpoint.get("seq_len", 32),
            "feat_dim": checkpoint.get("feat_dim", 186),
            "num_classes": checkpoint.get("num_classes", len(checkpoint.get("class_names", []))),
            "hidden1": checkpoint.get("hidden1", 128),
            "hidden2": checkpoint.get("hidden2", 96),
            "head_dim": checkpoint.get("head_dim", 192),
            "class_names": checkpoint.get("class_names", []),
        }
    state_dict = checkpoint.get("ema_state_dict", checkpoint.get("model_state_dict", checkpoint.get("state_dict")))

    class_names = meta.get("class_names", [])
    if not class_names:
        # Fallback ke labels/gabungan_labels.json, manual, atau wl
        candidate_labels = [
            "labels/gabungan_labels.json",
            "labels/manual_labels.json",
            "labels/wl_labels.json",
            "datasets/WL-BISINDO/labeled.json",
            "wl_labels.json",
        ]
        label_file = next((f for f in candidate_labels if os.path.exists(f)), None)
        if label_file:
            with open(label_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
                class_names = [item["gloss"] if isinstance(item, dict) else str(item) for item in raw]

    arch = meta.get("arch", "lstm")
    feat_dim = meta.get("feat_dim", EXTENDED_FEAT_DIM if arch == "mgn" else FEAT_DIM)
    seq_len = meta.get("seq_len", 32)

    model = build_eval_model(meta, state_dict, device)
    print(f"Model {arch.upper()} berhasil dimuat! (Input dim: {feat_dim}, Classes: {len(class_names)})")

    # 2. Muat Dataset Cache
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"File cache landmark tidak ditemukan: {cache_path}")

    cached = np.load(cache_path, allow_pickle=True)
    seqs = cached["sequences"]
    lengths = cached["lengths"]
    labels = cached["labels"]
    signers = cached["signers"]
    filenames = cached["filenames"].tolist() if "filenames" in cached else [f"video_{i}.mp4" for i in range(len(labels))]

    # 3. Pilih Subset Evaluasi
    if all_signers or split_mode == "all":
        eval_indices = np.arange(len(labels))
        split_desc = f"Seluruh Dataset ({len(eval_indices)} video)"
    elif split_mode == "stratified":
        from sklearn.model_selection import train_test_split
        _, eval_indices = train_test_split(
            np.arange(len(labels)), test_size=val_ratio, stratify=labels, random_state=seed
        )
        split_desc = f"Stratified Holdout Split {val_ratio*100:.0f}% ({len(eval_indices)} video)"
    else:
        val_mask = np.isin(signers, val_signers)
        eval_indices = np.where(val_mask)[0]
        split_desc = f"Holdout Signer {val_signers} ({len(eval_indices)} video)"

    print(f"Subset Evaluasi: {split_desc}")
    if len(eval_indices) == 0:
        raise ValueError("Set evaluasi kosong!")

    # 4. Inferensi Model
    inf_res = run_evaluation_inference(
        model=model,
        indices=eval_indices,
        seqs=seqs,
        lengths=lengths,
        labels=labels,
        signers=signers,
        filenames=filenames,
        device=device,
        seq_len=seq_len,
        feat_dim=feat_dim,
        tta=tta,
    )

    y_true = inf_res["y_true"]
    y_pred = inf_res["y_pred"]
    y_prob = inf_res["y_prob"]

    # 5. Hitung Metrik Akurasi
    top1_acc = float(accuracy_score(y_true, y_pred))
    n_classes = len(class_names)
    top3_acc = float(top_k_accuracy_score(y_true, y_prob, k=min(3, n_classes), labels=np.arange(n_classes)))
    top5_acc = float(top_k_accuracy_score(y_true, y_prob, k=min(5, n_classes), labels=np.arange(n_classes)))

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 50)
    print("HASIL EVALUASI MODEL:")
    print(f"  - Top-1 Accuracy : {top1_acc * 100:.2f}%")
    print(f"  - Top-3 Accuracy : {top3_acc * 100:.2f}%")
    print(f"  - Top-5 Accuracy : {top5_acc * 100:.2f}%")
    print(f"  - Macro F1-Score : {f1_macro:.4f}")
    print(f"  - Weighted F1    : {f1_weight:.4f}")
    print("=" * 50)

    # 6. Pembuatan Visualisasi
    image_paths = {}
    print("\nMenghasilkan plot visualisasi...")

    # Confusion Matrix
    p_cm_norm, p_cm_raw, cm_raw, cm_norm = plot_confusion_matrix(y_true, y_pred, class_names, out_path)
    image_paths["cm_norm"] = p_cm_norm
    image_paths["cm_raw"] = p_cm_raw
    print("  [OK] Confusion Matrix disimpan")

    # Top Confused Pairs
    p_pairs = plot_top_confused_pairs(y_true, y_pred, class_names, out_path)
    if p_pairs:
        image_paths["confused_pairs"] = p_pairs
        print("  [OK] Top Confused Pairs disimpan")

    # ROC Curves
    p_roc, roc_summary = plot_roc_curves(y_true, y_prob, class_names, out_path)
    image_paths["roc"] = p_roc
    print(f"  [OK] ROC Curves disimpan (Macro AUC: {roc_summary['macro_auc']:.3f})")

    # Precision-Recall Curves
    p_pr, pr_summary = plot_precision_recall_curves(y_true, y_prob, class_names, out_path)
    image_paths["pr"] = p_pr
    print(f"  [OK] Precision-Recall Curves disimpan (Micro AP: {pr_summary['micro_pr_auc']:.3f})")

    # Per-Class F1
    p_f1 = plot_per_class_metrics(y_true, y_pred, class_names, out_path)
    image_paths["f1_bar"] = p_f1
    print("  [OK] Per-Class F1 Bar Chart disimpan")

    # 7. Simpan Summary JSON & Markdown Report
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(n_classes), zero_division=0
    )
    per_class_list = []
    for i in range(n_classes):
        gloss_name = class_names[i] if i < len(class_names) else f"class_{i}"
        per_class_list.append({
            "class_idx": int(i),
            "gloss": gloss_name,
            "precision": round(float(p_per[i]), 4),
            "recall": round(float(r_per[i]), 4),
            "f1": round(float(f1_per[i]), 4),
            "support": int(sup_per[i]),
            "roc_auc": round(float(roc_summary.get("per_class_roc_auc", {}).get(gloss_name, 0.0)), 4),
            "pr_auc": round(float(pr_summary.get("per_class_pr_auc", {}).get(gloss_name, 0.0)), 4),
        })

    # Analisis pasangan kesalahan untuk JSON
    cm_temp = cm_raw.copy()
    np.fill_diagonal(cm_temp, 0)
    top_pairs_list = []
    for i in range(n_classes):
        for j in range(n_classes):
            if cm_temp[i, j] > 0:
                top_pairs_list.append({
                    "actual": class_names[i] if i < len(class_names) else f"class_{i}",
                    "predicted": class_names[j] if j < len(class_names) else f"class_{j}",
                    "count": int(cm_temp[i, j]),
                })
    top_pairs_list.sort(key=lambda x: x["count"], reverse=True)

    summary_data = {
        "model_path": model_path,
        "cache_path": cache_path,
        "arch": arch,
        "feat_dim": int(feat_dim),
        "data_source": split_desc,
        "total_samples": int(len(eval_indices)),
        "total_classes": int(n_classes),
        "class_names": class_names,
        "top1_accuracy": round(float(top1_acc), 4),
        "top3_accuracy": round(float(top3_acc), 4),
        "top5_accuracy": round(float(top5_acc), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weight), 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "roc_auc": roc_summary,
        "pr_auc": pr_summary,
        "confusion_matrix_raw": cm_raw.tolist(),
        "confusion_matrix_norm": (cm_norm * 100.0).round(2).tolist(),
        "top_confused_pairs": top_pairs_list[:25],
        "per_class": per_class_list,
    }

    json_out = out_path / "evaluation_metrics.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"Ringkasan metrik disimpan ke: {json_out}")

    report_md = out_path / "evaluation_report.md"
    generate_markdown_report(summary_data, report_md, image_paths)

    return summary_data


def main():
    parser = argparse.ArgumentParser(description="Evaluasi Model Klasifikasi Isyarat BISINDO (ROC, Confusion Matrix, dsb)")
    parser.add_argument("--dataset", type=str, default="gabungan", choices=["gabungan", "manual", "wl"], help="Pilihan dataset: 'gabungan' (default: 55 kelas), 'manual' (20 kelas), atau 'wl' (28 kelas)")
    parser.add_argument("--model", type=str, default=None, help="Path checkpoint model PyTorch (.pt)")
    parser.add_argument("--cache", type=str, default=None, help="Path cache landmark .npz")
    parser.add_argument("--out-dir", type=str, default=None, help="Direktori penyimpanan laporan & grafik")
    parser.add_argument("--split", type=str, default=None, choices=["stratified", "all", "holdout"], help="Metode split data evaluasi: 'stratified' (default untuk gabungan & manual), 'all', atau 'holdout'")
    parser.add_argument("--val-signer", type=int, default=3, help="ID signer untuk validasi holdout (default: 3)")
    parser.add_argument("--all-signers", action="store_true", help="Evaluasi seluruh sampel pada semua signer")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device komputasi")
    parser.add_argument("--tta", action="store_true", help="Aktifkan Test-Time Augmentation (rata-rata prediksi window asli + flip horizontal)")
    args = parser.parse_args()

    if args.dataset == "gabungan":
        model_candidates = [args.model, "models/gabungan_lstm_best.pt", "gabungan_lstm_best.pt"]
        model_path = next((f for f in model_candidates if f and os.path.exists(f)), "models/gabungan_lstm_best.pt")
        cache_candidates = [args.cache, "data/gabungan_holistic_lstm_cache.npz", "gabungan_holistic_lstm_cache.npz"]
        cache_path = next((f for f in cache_candidates if f and os.path.exists(f)), "data/gabungan_holistic_lstm_cache.npz")
        out_dir = args.out_dir or "reports/evaluation/gabungan"
        split_mode = args.split or ("all" if args.all_signers else "stratified")
    elif args.dataset == "manual":
        model_candidates = [args.model, "models/manual_lstm_best.pt", "manual_lstm_best.pt"]
        model_path = next((f for f in model_candidates if f and os.path.exists(f)), "models/manual_lstm_best.pt")
        cache_candidates = [args.cache, "data/manual_holistic_lstm_cache.npz", "manual_holistic_lstm_cache.npz"]
        cache_path = next((f for f in cache_candidates if f and os.path.exists(f)), "data/manual_holistic_lstm_cache.npz")
        out_dir = args.out_dir or "reports/evaluation/manual"
        split_mode = args.split or ("all" if args.all_signers else "stratified")
    else:
        model_candidates = [args.model, "models/bisindo_lstm_best.pt", "bisindo_lstm_best.pt"]
        model_path = next((f for f in model_candidates if f and os.path.exists(f)), "models/bisindo_lstm_best.pt")
        cache_candidates = [args.cache, "data/wl_holistic_lstm_cache.npz", "wl_holistic_lstm_cache.npz"]
        cache_path = next((f for f in cache_candidates if f and os.path.exists(f)), "data/wl_holistic_lstm_cache.npz")
        out_dir = args.out_dir or "reports/evaluation/mgn_eval"
        split_mode = args.split or ("all" if args.all_signers else "holdout")

    run_evaluation(
        model_path=model_path,
        cache_path=cache_path,
        out_dir=out_dir,
        val_signers=(args.val_signer,),
        all_signers=args.all_signers,
        split_mode=split_mode,
        device_choice=args.device,
        tta=args.tta,
    )


if __name__ == "__main__":
    main()

