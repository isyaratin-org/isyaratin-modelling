# %%
"""
Training Classifier SEQUENCE (Video) BISINDO dengan Multi-Landmark + BiLSTM (PyTorch + CUDA).

Fitur Multi-Landmark (186 dimensi per frame):
  - Left Hand  : 21 landmark x 3 (x, y, z) = 63 fitur
  - Right Hand : 21 landmark x 3 (x, y, z) = 63 fitur
  - Arm & Bahu : 6 landmark x 3 (bahu, siku, pergelangan) = 18 fitur
  - Head/Face  : 11 landmark x 3 (hidung, mata, telinga, sudut mulut) = 33 fitur
  - Torso/Pinggul: 2 landmark x 3 (pinggul kiri & kanan untuk orientasi torso) = 6 fitur
  - Presence Flags: [has_left_hand, has_right_hand, has_pose] = 3 fitur
  Total: 63 + 63 + 18 + 33 + 6 + 3 = 186 dimensi per frame.

Normalisasi Anatomis Tubuh (Body-Centric Normalization):
  - Origin = Titik tengah antara bahu kiri dan bahu kanan (torso_center).
  - Skala  = Jarak Euclidean antara kedua bahu (torso_scale).
  - Seluruh koordinat tangan, lengan, dan kepala ditranslasikan ke origin dan dibagi skala tubuh.
  - Hasil: Posisi tangan terhadap wajah (misal: isyarat 'Air' dekat mulut, 'Ingat' dekat dahi,
    'Saya' dekat dada) tetap konsisten terlepas dari resolusi kamera dan jarak signer.

Arsitektur Model:
  - Input: Sequence (Batch, SEQ_LEN=32, 186)
  - LayerNorm(186)
  - Stacked Bidirectional LSTM (Layer 1: 128 hidden, Layer 2: 96 hidden)
  - Dual Temporal Pooling (Mean Pooling + Max Pooling) -> 384 fitur representasi
  - Classifier Head: Linear(384, 192) -> GELU -> Dropout -> LayerNorm -> Linear(192, NumClasses)

Cara Menjalankan:
  1. Uji cepat (Smoke test ~32 video):
     uv run python train-new.py --smoke
  2. Training penuh dengan GPU:
     uv run python train-new.py --epochs 60 --batch-size 32
  3. Paksa ekstraksi ulang cache landmark:
     uv run python train-new.py --re-extract
  4. Interaktif di VS Code / Jupyter:
     Jalankan per sel (# %%) langsung di editor.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import copy
import json
import math
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm

# %%
# =========================================================
# 1. KONFIGURASI UMUM & HYPERPARAMETER
# =========================================================

DATA_DIR = "datasets/WL-BISINDO"
LABELS_FILE = "labels/wl_labels.json" if os.path.exists("labels/wl_labels.json") else "wl_labels.json"
CACHE_FILE = "data/wl_holistic_lstm_cache.npz" if os.path.exists("data/wl_holistic_lstm_cache.npz") else "wl_holistic_lstm_cache.npz"
MODEL_SAVE_PT = "models/bisindo_lstm_best.pt"
MODEL_SAVE_ONNX = "models/bisindo_lstm.onnx"
MODEL_SAVE_TFLITE = "models/sign_model.tflite"
LABELS_SAVE_TXT = "models/labels.txt"
FLUTTER_ASSETS_DIR = r"d:\Project\4mlai\apapun\isyaratin\assets\models"

# Pola nama file video: signer{ID}_label{ID}_sample{ID}.mp4
FILE_PATTERN = re.compile(r"signer(\d+)_label(\d+)_sample(\d+)", re.IGNORECASE)
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

# Landmark & Temporal Settings
SEQ_LEN = 32              # Jumlah frame per sequence window (~2.67 detik pada 12 fps)
TARGET_FPS = 12.0         # Sampling frame rate standar
MAX_FRAMES_PER_VIDEO = 120 # Batas maksimal frame yang diekstrak per video
MAX_SIDE = 512            # Maksimal resolusi sisi gambar saat ekstraksi MediaPipe

# Landmark Indices dari MediaPipe Pose (33 total)
# 11 Head points: 0=nose, 1=eye_inner_l, 2=eye_l, 3=eye_outer_l, 4=eye_inner_r, 5=eye_r,
#                 6=eye_outer_r, 7=ear_l, 8=ear_r, 9=mouth_l, 10=mouth_r
HEAD_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# 6 Arm points: 11=shoulder_l, 12=shoulder_r, 13=elbow_l, 14=elbow_r, 15=wrist_l, 16=wrist_r
ARM_INDICES = [11, 12, 13, 14, 15, 16]
# 2 Torso points: 23=hip_l, 24=hip_r
TORSO_INDICES = [23, 24]
SELECTED_POSE_INDICES = HEAD_INDICES + ARM_INDICES + TORSO_INDICES  # 19 landmarks

# Dimensi Fitur
NUM_HAND_LANDMARKS = 21
NUM_POSE_LANDMARKS = len(SELECTED_POSE_INDICES) # 19
FEAT_DIM = (NUM_HAND_LANDMARKS * 3 * 2) + (NUM_POSE_LANDMARKS * 3) + 3 # 63 + 63 + 57 + 3 = 186

# Signer Holdout untuk Validasi (Unseen Signer Validation)
# Signer 0 hanya punya 12 label. Signer 1, 2, 3, 4 punya 32 label lengkap.
VAL_SIGNERS = (3,)

# Hyperparameter Training
DEFAULT_EPOCHS = 60
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_HIDDEN1 = 128
DEFAULT_HIDDEN2 = 96
DEFAULT_HEAD_DIM = 192
DEFAULT_DROPOUT = 0.3
LABEL_SMOOTHING = 0.05
SEED = 42


def print_config():
    print("=== KONFIGURASI MULTI-LANDMARK BISINDO LSTM ===")
    print(f"Python      : {sys.executable}")
    try:
        import torch
        print(f"PyTorch     : {torch.__version__} | CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Device  : {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass
    print(f"Seq Len     : {SEQ_LEN} frames @ {TARGET_FPS} fps")
    print(f"Fitur Dim   : {FEAT_DIM} (2x21 Hands + 19 Pose Arm/Head/Torso + 3 Flags)")
    print(f"Dataset Dir : {DATA_DIR}")
    print(f"Cache File  : {CACHE_FILE}")


from holistic_extractor import (
    EXTRACTOR_VERSION,
    FEAT_DIM,
    EXTENDED_FEAT_DIM,
    SELECTED_POSE_INDICES,
    HEAD_INDICES,
    ARM_INDICES,
    TORSO_INDICES,
    extract_holistic_frame,
    extract_video_features,
    init_worker,
    process_video_task,
    compute_extended_features,
    flip_horizontal_landmarks,
)


def build_or_load_dataset(
    data_dir: str = DATA_DIR,
    cache_file: str = CACHE_FILE,
    labels_file: str = LABELS_FILE,
    re_extract: bool = False,
    smoke: bool = False,
    num_workers: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Ekstrak atau muat dataset dari cache npz.
    Return:
        padded_seqs : (N, MAX_FRAMES_PER_VIDEO, 186)
        lengths     : (N,) durasi frame tiap video
        labels      : (N,) label kelas int
        signers     : (N,) ID signer int
        filenames   : list of video filenames
        class_names : list of nama kelas
    """
    # 1. Cek Struktur Dataset (Subfolder-based seperti datasets/manual vs Flat Regex seperti WL-BISINDO)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Direktori dataset tidak ditemukan: {data_dir}")

    subdirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")
    ])
    is_folder_based = len(subdirs) > 0 and any(
        any(f.lower().endswith(VIDEO_EXTS) for f in os.listdir(os.path.join(data_dir, sd)))
        for sd in subdirs
    )

    folder_path_map = {}
    if os.path.exists(labels_file):
        with open(labels_file, "r", encoding="utf-8") as f:
            raw_labels = json.load(f)
        if isinstance(raw_labels, list):
            class_names = []
            for item in raw_labels:
                if isinstance(item, dict):
                    name = item.get("gloss") or item.get("name") or str(item.get("id"))
                    class_names.append(name)
                    # Jika ada field path atau folder di JSON, gunakan langsung path tersebut
                    if "path" in item:
                        folder_path_map[name] = item["path"]
                    elif "folder" in item:
                        folder_path_map[name] = os.path.join(data_dir, item["folder"])
                else:
                    class_names.append(str(item))
            print(f"[Label Dimuat dari {labels_file}] {len(class_names)} kelas: {class_names}")
        else:
            raise ValueError(f"Format {labels_file} harus berupa list/array JSON.")
    elif is_folder_based:
        class_names = subdirs
        with open(labels_file, "w", encoding="utf-8") as f:
            json.dump(class_names, f, indent=2, ensure_ascii=False)
        print(f"[Dataset Subfolder Terdeteksi] Ditemukan {len(class_names)} kelas: {class_names}")
    else:
        raise FileNotFoundError(f"File label tidak ditemukan: {labels_file}")

    # 2. Cek Cache
    if not re_extract and os.path.exists(cache_file):
        print(f"Memuat dataset dari cache: {cache_file} ...")
        cached = np.load(cache_file, allow_pickle=True)
        cached_dim = cached["sequences"].shape[-1]
        cached_ver = str(cached.get("extractor_version", "1.0.0"))
        if cached_dim != FEAT_DIM:
            print(f"PERINGATAN: Dimensi cache ({cached_dim}) beda dengan konfigurasi ({FEAT_DIM}). Re-ekstraksi...")
        elif cached_ver != EXTRACTOR_VERSION:
            print(f"PERINGATAN: Versi extractor di cache ({cached_ver}) beda dengan sistem ({EXTRACTOR_VERSION}). Re-ekstraksi...")
        else:
            seqs = cached["sequences"]
            lengths = cached["lengths"]
            labels = cached["labels"]
            signers = cached["signers"]
            filenames = cached["filenames"].tolist()
            print(f"Berhasil memuat {len(seqs)} video dari cache! (Feat dim: {cached_dim})")
            if smoke:
                smoke_idx = []
                for cls in range(len(class_names)):
                    matches = np.where(labels == cls)[0][:2]
                    smoke_idx.extend(matches.tolist())
                smoke_idx = np.array(smoke_idx)
                print(f"[SMOKE MODE] Menggunakan subset {len(smoke_idx)} video.")
                return seqs[smoke_idx], lengths[smoke_idx], labels[smoke_idx], signers[smoke_idx], [filenames[i] for i in smoke_idx], class_names
            return seqs, lengths, labels, signers, filenames, class_names

    # 3. Pindai File Video
    tasks = []
    if is_folder_based:
        for cls_id, cls_name in enumerate(class_names):
            cls_folder = folder_path_map.get(cls_name) or os.path.join(data_dir, cls_name)
            if not os.path.isdir(cls_folder):
                print(f"[Peringatan] Folder kelas '{cls_name}' tidak ditemukan di {cls_folder}, dilewati.")
                continue
            video_files = sorted([f for f in os.listdir(cls_folder) if f.lower().endswith(VIDEO_EXTS)])
            for sample_id, fn in enumerate(video_files):
                full_path = os.path.join(cls_folder, fn)
                rel_fn = f"{cls_name}/{fn}"
                signer_id = 1
                tasks.append((full_path, cls_id, signer_id, sample_id, rel_fn))
    else:
        all_files = sorted(os.listdir(data_dir))
        for fn in all_files:
            if not fn.lower().endswith(VIDEO_EXTS):
                continue
            m = FILE_PATTERN.search(fn)
            if not m:
                continue
            signer_id = int(m.group(1))
            label_id = int(m.group(2))
            sample_id = int(m.group(3))
            full_path = os.path.join(data_dir, fn)
            tasks.append((full_path, label_id, signer_id, sample_id, fn))

    print(f"Total video terdeteksi di {data_dir}: {len(tasks)}")
    if smoke:
        tasks_by_label = {}
        for t in tasks:
            tasks_by_label.setdefault(t[1], []).append(t)
        limited_tasks = []
        for l_id, l_list in tasks_by_label.items():
            limited_tasks.extend(l_list[:2])
        tasks = limited_tasks
        print(f"[SMOKE MODE] Dibatasi mengekstrak {len(tasks)} video.")

    # 4. Ekstraksi Video (Multi-Core Paralel)
    if num_workers is None:
        cpu_cnt = os.cpu_count() or 4
        num_workers = min(4, max(2, cpu_cnt // 2))

    extracted_records = []
    failed = 0

    if num_workers <= 1:
        init_worker()
        for t in tqdm(tasks, desc="Ekstraksi Video"):
            res = process_video_task(t)
            if res["ok"]:
                extracted_records.append(res)
            else:
                failed += 1
    else:
        from concurrent.futures import ProcessPoolExecutor
        print(f"Mengekstrak landmark secara paralel dengan {num_workers} worker CPU...")
        chunk = max(1, min(4, len(tasks) // (num_workers * 2)))
        with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
            for res in tqdm(executor.map(process_video_task, tasks, chunksize=chunk), total=len(tasks), desc=f"Ekstraksi Video ({num_workers} Worker)"):
                if res["ok"]:
                    extracted_records.append(res)
                else:
                    failed += 1

    print(f"\nEkstraksi selesai: {len(extracted_records)} berhasil, {failed} gagal.")
    if not extracted_records:
        raise RuntimeError("Tidak ada video yang berhasil diekstrak!")

    # 5. Pack ke Padded Array
    n_samples = len(extracted_records)
    padded_seqs = np.zeros((n_samples, MAX_FRAMES_PER_VIDEO, FEAT_DIM), dtype=np.float32)
    lengths = np.zeros(n_samples, dtype=np.int64)
    labels = np.zeros(n_samples, dtype=np.int64)
    signers = np.zeros(n_samples, dtype=np.int64)
    filenames = []

    for i, rec in enumerate(extracted_records):
        feats = rec["feats"]
        t = min(len(feats), MAX_FRAMES_PER_VIDEO)
        padded_seqs[i, :t] = feats[:t]
        lengths[i] = t
        labels[i] = rec["label"]
        signers[i] = rec["signer"]
        filenames.append(rec["filename"])

    # 6. Simpan Cache (hanya jika bukan smoke test)
    if not smoke:
        os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
        np.savez_compressed(
            cache_file,
            sequences=padded_seqs,
            lengths=lengths,
            labels=labels,
            signers=signers,
            filenames=np.array(filenames),
            class_names=np.array(class_names),
            extractor_version=EXTRACTOR_VERSION,
        )
        print(f"Dataset berhasil disimpan ke cache: {cache_file}")

    return padded_seqs, lengths, labels, signers, filenames, class_names


# %%
# =========================================================
# 4. TEMPORAL SAMPLING & DATA AUGMENTASI
# =========================================================

def find_motion_trimmed_bounds(feats: np.ndarray, min_frames: int = 16) -> Tuple[int, int]:
    """
    Strategi 2: Motion-Trimmed Sequence Cropping (Diperbaiki)
    Mendeteksi batas awal dan akhir gerakan aktif isyarat dari runtun frame (T, 186).
    - Menghitung energi gerak hanya pada frame di mana landmark aktif di kedua frame t dan t-1,
      mencegah spike palsu akibat transisi tangan dari absen menjadi terdeteksi.
    - Mengutamakan pergerakan pergelangan tangan (wrist) dan landmark tangan yang stabil.
    - Menjaga isyarat statis/hold agar tidak terpotong secara agresif.
    """
    t = feats.shape[0]
    if t <= min_frames:
        return 0, t - 1

    flags = feats[:, -3:]
    has_lh = flags[:, 0] > 0.5
    has_rh = flags[:, 1] > 0.5
    has_pose = flags[:, 2] > 0.5
    has_hand = has_lh | has_rh

    hand_idx = np.where(has_hand)[0]
    if len(hand_idx) < min_frames:
        return 0, t - 1

    coords = feats[:, :-3].reshape(t, 61, 3)

    # Frame-to-frame continuous motion energy (tanpa spike saat tangan baru muncul)
    lh_cont = has_lh[1:] & has_lh[:-1]
    rh_cont = has_rh[1:] & has_rh[:-1]
    pose_cont = has_pose[1:] & has_pose[:-1]

    diffs = np.zeros(t, dtype=np.float32)
    for i in range(1, t):
        m_diff = 0.0
        cnt = 0
        if lh_cont[i - 1]:
            # LH Wrist (point 0) + mean 21 joints
            m_diff += float(np.linalg.norm(coords[i, 0] - coords[i - 1, 0]))
            m_diff += float(np.mean(np.linalg.norm(coords[i, :21] - coords[i - 1, :21], axis=-1)))
            cnt += 2
        if rh_cont[i - 1]:
            # RH Wrist (point 21) + mean 21 joints
            m_diff += float(np.linalg.norm(coords[i, 21] - coords[i - 1, 21]))
            m_diff += float(np.mean(np.linalg.norm(coords[i, 21:42] - coords[i - 1, 21:42], axis=-1)))
            cnt += 2
        if pose_cont[i - 1]:
            # Pose wrists (ARM_INDICES: 15, 16 -> index 57, 58)
            m_diff += float(np.linalg.norm(coords[i, 57] - coords[i - 1, 57]))
            m_diff += float(np.linalg.norm(coords[i, 58] - coords[i - 1, 58]))
            cnt += 2
        diffs[i] = m_diff / max(1, cnt)

    # Deteksi ambang batas gerak aktif
    hand_diffs = diffs[hand_idx]
    if len(hand_diffs) > 0 and np.max(hand_diffs) > 0.05:
        motion_thresh = max(0.025, float(np.percentile(hand_diffs, 15)))
        moving_idx = np.where(has_hand & (diffs >= motion_thresh))[0]
        if len(moving_idx) >= min_frames // 2:
            s_idx = max(0, int(moving_idx[0]) - 3)
            e_idx = min(t - 1, int(moving_idx[-1]) + 3)
            if e_idx - s_idx >= min_frames:
                return s_idx, e_idx

    # Fallback konservatif jika isyarat statis/hold (tidak memotong gerakan berkecepatan rendah)
    s_idx = max(0, int(hand_idx[0]) - 2)
    e_idx = min(t - 1, int(hand_idx[-1]) + 2)
    if e_idx - s_idx < min_frames:
        return 0, t - 1
    return s_idx, e_idx


def sample_sequence_window(
    feats: np.ndarray,
    seq_len: int = SEQ_LEN,
    is_train: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Mengambil window sepanjang seq_len dari video berdurasi T frame
    dengan Motion-Trimmed Sequence Cropping (Strategi 2).
    """
    t = feats.shape[0]
    if t < seq_len:
        idx = np.linspace(0, t - 1, seq_len).round().astype(np.int64)
        return feats[idx].copy()

    # Potong fase diam di awal dan akhir
    s_bound, e_bound = find_motion_trimmed_bounds(feats, min_frames=16)
    trimmed_len = e_bound - s_bound + 1

    if is_train and rng is not None:
        # Augmentasi Jitter Temporal pada batas isyarat aktif (+/- 8%)
        margin_s = int(round(trimmed_len * 0.08))
        margin_e = int(round(trimmed_len * 0.08))
        s_idx = max(0, s_bound + int(rng.integers(-margin_s, margin_s + 1)))
        e_idx = min(t - 1, e_bound + int(rng.integers(-margin_e, margin_e + 1)))
        if e_idx - s_idx < 16:
            s_idx, e_idx = s_bound, e_bound

        # 75% probabilitas: Global Resampling dari fase aktif isyarat penuh (Awal -> Puncak -> Akhir)
        if rng.uniform(0.0, 1.0) < 0.75:
            idx = np.linspace(s_idx, e_idx, seq_len).round().astype(np.int64)
            return feats[idx].copy()

        # 25% probabilitas: Temporal time-warp / speed jitter di dalam segmen aktif
        scale = float(rng.uniform(0.80, 1.20))
        crop_len = int(round(seq_len * scale))
        crop_len = max(16, min(e_idx - s_idx + 1, crop_len))
        start = int(rng.integers(s_idx, max(s_idx + 1, e_idx - crop_len + 1)))
        sub = feats[start : start + crop_len]
        idx = np.linspace(0, len(sub) - 1, seq_len).round().astype(np.int64)
        return sub[idx].copy()
    else:
        # Evaluasi deterministik: Resample seluruh fase aktif isyarat ke seq_len
        idx = np.linspace(s_bound, e_bound, seq_len).round().astype(np.int64)
        return feats[idx].copy()


def augment_sequence_window(
    window: np.ndarray,
    rng: np.random.Generator,
    flip_prob: float = 0.5,
    noise_std: float = 0.012,
    scale_range: Tuple[float, float] = (0.88, 1.12),
    rot_range_deg: float = 8.0,
    temporal_flicker_prob: float = 0.25,
    full_dropout_prob: float = 0.03,
) -> np.ndarray:
    """
    Augmentasi spasial & temporal tingkat lanjut pada satu window (seq_len, 186):
      0. Horizontal Mirroring / Bilateral Flip (Left <-> Right Hands & Pose swap)
      1. Scale jitter & Spatial translation (jarak signer ke kamera)
      2. 2D Rotation jitter (rotasi tubuh pada sumbu XY)
      3. Gaussian noise mikro pada landmark
      4. Temporal Hand Flicker (simulasi oklusi sementara 1-4 frame kamera/blur)
      5. Wrist-Centric Hand Jitter (variasi posisi lokal tangan terhadap torso)
      6. Full Hand Dropout langka
    """
    out = window.copy()

    # 0. Random Horizontal Mirroring (Bilateral Flip)
    if rng.uniform(0.0, 1.0) < flip_prob:
        out = flip_horizontal_landmarks(out)

    seq_len = out.shape[0]
    coords = out[:, :-3].reshape(seq_len, -1, 3) # (seq_len, 61, 3)
    flags = out[:, -3:]

    # 1. Scale Jitter & Translation (Jarak signer)
    s = rng.uniform(*scale_range)
    coords = coords * s
    shift = rng.normal(0.0, 0.02, size=(1, 1, 3)).astype(np.float32)
    coords = coords + shift

    # 2. 2D Rotation Jitter
    deg = rng.uniform(-rot_range_deg, rot_range_deg)
    rad = math.radians(deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    xy = coords[:, :, :2]
    xy_rot = np.einsum("ij,tkj->tki", rot_matrix, xy)
    coords[:, :, :2] = xy_rot

    # 3. Gaussian Noise Mikro pada koordinat aktif
    noise = rng.normal(0.0, noise_std, size=coords.shape).astype(np.float32)
    coords = coords + noise

    # 4. Temporal Hand Flicker (MediaPipe drop 1-4 frame sementara karena gerakan cepat/blur)
    if rng.uniform(0.0, 1.0) < temporal_flicker_prob:
        target_slice = slice(0, 21) if rng.uniform(0.0, 1.0) < 0.5 else slice(21, 42)
        f_idx = 0 if target_slice.start == 0 else 1
        f_len = int(rng.integers(1, 5))
        f_start = int(rng.integers(0, max(1, seq_len - f_len)))
        coords[f_start : f_start + f_len, target_slice, :] = 0.0
        flags[f_start : f_start + f_len, f_idx] = 0.0

    # 5. Wrist-Centric Hand Jitter (variasi lokal tangan)
    if rng.uniform(0.0, 1.0) < 0.35:
        j_lh = rng.normal(0.0, 0.012, size=(1, 1, 3)).astype(np.float32)
        coords[:, :21, :] += j_lh
    if rng.uniform(0.0, 1.0) < 0.35:
        j_rh = rng.normal(0.0, 0.012, size=(1, 1, 3)).astype(np.float32)
        coords[:, 21:42, :] += j_rh

    # 6. Full Hand Dropout (simulasi signer hanya menggunakan 1 tangan aktif)
    if rng.uniform(0.0, 1.0) < full_dropout_prob:
        coords[:, :21, :] = 0.0
        flags[:, 0] = 0.0
    if rng.uniform(0.0, 1.0) < full_dropout_prob:
        coords[:, 21:42, :] = 0.0
        flags[:, 1] = 0.0

    # Re-masking ketat: koordinat landmark dari komponen yang tidak terdeteksi (flags=0) HARUS tetap 0.0
    lh_mask = (flags[:, 0:1] > 0.5).astype(np.float32)[:, :, None]  # (seq_len, 1, 1)
    rh_mask = (flags[:, 1:2] > 0.5).astype(np.float32)[:, :, None]
    pose_mask = (flags[:, 2:3] > 0.5).astype(np.float32)[:, :, None]

    coords[:, :21, :] *= lh_mask
    coords[:, 21:42, :] *= rh_mask
    coords[:, 42:61, :] *= pose_mask

    out = np.concatenate([coords.reshape(seq_len, -1), flags], axis=-1)
    return out.astype(np.float32)


# %%
# =========================================================
# 5. ARSITEKTUR MODEL: MGN (MULTI-GRANULARITY NETWORK) & LSTM
# =========================================================

def build_mgn_model(
    feat_dim: int = EXTENDED_FEAT_DIM,
    num_classes: int = 32,
    conv_dim: int = DEFAULT_HIDDEN1,
    lstm_hidden: int = DEFAULT_HIDDEN2,
    head_dim: int = DEFAULT_HEAD_DIM,
    dropout: float = DEFAULT_DROPOUT,
):
    """
    Membangun arsitektur Multi-Granularity Network (MGN):
      - Micro-Granularity: Temporal Conv1D (k=3, k=5) + Residual Connection
      - Macro-Granularity: 2-Layer Bidirectional LSTM
      - Stroke Attention Pooling + Peak MaxPooling
      - Classification MLP Head
    """
    import torch
    import torch.nn as nn

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

            # Micro-granularity: 1D Temporal Convolutions (k=3, k=5)
            self.conv_in = nn.Conv1d(feat_dim, conv_dim, kernel_size=1)
            self.conv3 = nn.Conv1d(conv_dim, conv_dim, kernel_size=3, padding=1)
            self.conv5 = nn.Conv1d(conv_dim, conv_dim, kernel_size=5, padding=2)
            self.conv_act = nn.GELU()
            self.conv_bn = nn.BatchNorm1d(conv_dim)

            # Macro-granularity: 2-Layer Bidirectional LSTM
            self.lstm = nn.LSTM(
                input_size=conv_dim,
                hidden_size=lstm_hidden,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=dropout,
            )

            bilstm_out_dim = lstm_hidden * 2
            self.attn_pool = TemporalAttention(bilstm_out_dim)

            pooled_dim = bilstm_out_dim * 2  # Attention Context + Peak Max Pooling
            self.head = nn.Sequential(
                nn.Linear(pooled_dim, head_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(head_dim),
                nn.Linear(head_dim, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.input_norm(x)

            # 1. Micro-granularity local conv
            x_conv = self.conv_in(x.transpose(1, 2))
            res = x_conv
            c3 = self.conv3(x_conv)
            c5 = self.conv5(x_conv)
            x_micro = self.conv_act(self.conv_bn(c3 + c5) + res)

            # 2. Macro-granularity BiLSTM
            x_macro_in = x_micro.transpose(1, 2)
            self.lstm.flatten_parameters()
            h, _ = self.lstm(x_macro_in)

            # 3. Dual Pooling: Temporal Attention + Peak Max
            attn_out, _ = self.attn_pool(h)
            max_out = h.amax(dim=1)
            pooled = torch.cat([attn_out, max_out], dim=-1)

            logits = self.head(pooled)
            return logits

    return BisindoMGNClassifier()


def build_lstm_model(
    feat_dim: int = FEAT_DIM,
    num_classes: int = 32,
    hidden1: int = DEFAULT_HIDDEN1,
    hidden2: int = DEFAULT_HIDDEN2,
    head_dim: int = DEFAULT_HEAD_DIM,
    dropout: float = DEFAULT_DROPOUT,
):
    """
    Membangun arsitektur PyTorch Bidirectional LSTM 2 Layer.
    """
    import torch
    import torch.nn as nn

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

            pooled_dim = hidden2 * 4

            self.head = nn.Sequential(
                nn.Linear(pooled_dim, head_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(head_dim),
                nn.Linear(head_dim, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
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

            logits = self.head(pooled)
            return logits

    return BisindoLSTMClassifier()


class ModelEMA:
    """
    Exponential Moving Average (EMA) untuk stabilisasi bobot model.
    Menyinkronkan parameter bobot dengan decay dan buffer BatchNorm model utama.
    """

    def __init__(self, model, decay: float = 0.995):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    def update(self, model):
        import torch
        with torch.no_grad():
            for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
                s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)
            for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
                s_buf.copy_(m_buf)


def get_focal_loss_class():
    """
    Factory yang mengembalikan class FocalLoss (lazy load PyTorch).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class FocalLoss(nn.Module):
        """
        Focal Loss multi-class dengan Label Smoothing dan Class Weighting:
          FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        Secara drastis mengurangi gradien dari sampel mudah (yang sudah memiliki p_t > 0.9)
        dan memfokuskan kapasitas model pada kelas-kelas sulit (hard confused pairs).
        """

        def __init__(
            self,
            gamma: float = 2.0,
            weight: Optional[torch.Tensor] = None,
            label_smoothing: float = 0.05,
        ):
            super().__init__()
            self.gamma = gamma
            self.weight = weight
            self.label_smoothing = label_smoothing

        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            num_classes = inputs.size(-1)
            log_p = F.log_softmax(inputs, dim=-1)
            p = torch.exp(log_p)

            with torch.no_grad():
                targets_one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1.0)
                if self.label_smoothing > 0.0:
                    smooth = self.label_smoothing / num_classes
                    targets_one_hot = targets_one_hot * (1.0 - self.label_smoothing) + smooth

            focal_weight = torch.pow(1.0 - p, self.gamma)
            loss = -targets_one_hot * focal_weight * log_p

            if self.weight is not None:
                loss = loss * self.weight.unsqueeze(0)

            return loss.sum(dim=-1).mean()

    return FocalLoss


# %%
# =========================================================
# 6. DATA GENERATOR / BATCH ITERATOR
# =========================================================

def iterate_batches(
    indices: np.ndarray,
    seqs: np.ndarray,
    lengths: np.ndarray,
    labels: np.ndarray,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seq_len: int = SEQ_LEN,
    feat_dim: int = EXTENDED_FEAT_DIM,
    is_train: bool = True,
    augment: bool = True,
    rng: Optional[np.random.Generator] = None,
):
    """
    Generator batch tensor untuk training / evaluasi.
    Otomatis menghitung dynamic velocity & relational features (386 dim)
    jika feat_dim == EXTENDED_FEAT_DIM.
    """
    import torch

    rng = rng or np.random.default_rng()
    idx_list = indices.copy()
    if is_train:
        rng.shuffle(idx_list)

    n_samples = len(idx_list)
    for start in range(0, n_samples, batch_size):
        chunk = idx_list[start : start + batch_size]
        B = len(chunk)
        X_raw = np.zeros((B, seq_len, FEAT_DIM), dtype=np.float32)
        Y = np.zeros(B, dtype=np.int64)

        for b, i in enumerate(chunk):
            raw_v = seqs[i, : lengths[i]]
            win = sample_sequence_window(raw_v, seq_len=seq_len, is_train=is_train, rng=rng)
            if is_train and augment:
                win = augment_sequence_window(win, rng=rng)
            X_raw[b] = win
            Y[b] = labels[i]

        if feat_dim == EXTENDED_FEAT_DIM:
            X = compute_extended_features(X_raw)
        else:
            X = X_raw

        yield torch.from_numpy(X), torch.from_numpy(Y)


# %%
# =========================================================
# 7. TRAINING & EVALUASI
# =========================================================

def evaluate_model(
    model,
    indices: np.ndarray,
    seqs: np.ndarray,
    lengths: np.ndarray,
    labels: np.ndarray,
    device,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seq_len: int = SEQ_LEN,
    feat_dim: int = EXTENDED_FEAT_DIM,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluasi model pada window tengah deterministik per video.
    Return: (val_loss, val_acc, all_preds, all_targets)
    """
    import torch
    import torch.nn as nn

    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x_b, y_b in iterate_batches(
            indices, seqs, lengths, labels,
            batch_size=batch_size, seq_len=seq_len,
            feat_dim=feat_dim, is_train=False, augment=False,
        ):
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            logits = model(x_b)
            loss = criterion(logits, y_b)

            total_loss += float(loss.item()) * len(y_b)
            preds = logits.argmax(dim=-1)
            correct += int((preds == y_b).sum().item())
            total += len(y_b)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y_b.cpu().numpy().tolist())

    val_loss = total_loss / max(1, total)
    val_acc = correct / max(1, total)
    return val_loss, val_acc, np.array(all_preds), np.array(all_targets)


def evaluate_video_consensus(
    model,
    indices: np.ndarray,
    seqs: np.ndarray,
    lengths: np.ndarray,
    labels: np.ndarray,
    device,
    seq_len: int = SEQ_LEN,
    feat_dim: int = EXTENDED_FEAT_DIM,
    stride: int = 4,
) -> float:
    """
    Evaluasi konsensus seluruh video: sliding window di sepanjang durasi video,
    kemudian dirata-rata probabilitasnya. Menghitung akurasi 1 video = 1 keputusan.
    """
    import torch

    model.eval()
    correct = 0
    total = len(indices)

    with torch.no_grad():
        for i in indices:
            raw_v = seqs[i, : lengths[i]]
            t = raw_v.shape[0]

            if t < seq_len:
                idx = np.linspace(0, t - 1, seq_len).round().astype(np.int64)
                windows = [raw_v[idx]]
            else:
                starts = list(range(0, t - seq_len + 1, stride))
                if (t - seq_len) not in starts:
                    starts.append(t - seq_len)
                windows = [raw_v[s : s + seq_len] for s in starts]

            windows_arr = np.stack(windows)
            if feat_dim == EXTENDED_FEAT_DIM:
                windows_arr = compute_extended_features(windows_arr)

            x = torch.from_numpy(windows_arr).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1).mean(dim=0)
            pred = int(probs.argmax().item())
            if pred == labels[i]:
                correct += 1

    return correct / max(1, total)


# %%
# =========================================================
# 8. EXPORT MODEL (ONNX & TFLITE)
# =========================================================

def export_model_onnx(
    model,
    out_path: str = MODEL_SAVE_ONNX,
    seq_len: int = SEQ_LEN,
    feat_dim: int = FEAT_DIM,
) -> str:
    """
    Ekspor model PyTorch LSTM ke format ONNX standar dengan output Softmax.
    Menggunakan dynamo=False untuk stabilitas RNN di PyTorch.
    """
    import torch
    import torch.nn as nn

    print(f"\nMengekspor model ke ONNX: {out_path} ...")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cpu_model = copy.deepcopy(model).to("cpu").eval()

    class ModelWithSoftmax(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return torch.softmax(self.inner(x), dim=-1)

    wrapped = ModelWithSoftmax(cpu_model)
    dummy_input = torch.randn(1, seq_len, feat_dim, dtype=torch.float32)

    torch.onnx.export(
        wrapped,
        dummy_input,
        out_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["sequence"],
        output_names=["probabilities"],
        dynamic_axes={"sequence": {0: "batch"}, "probabilities": {0: "batch"}},
        dynamo=False,
    )
    print(f"Model ONNX berhasil diekspor ke: {out_path}")

    # Verifikasi kompatibilitas ONNX Runtime untuk Mobile
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        test_in = np.random.randn(1, seq_len, feat_dim).astype(np.float32)
        test_out = session.run(["probabilities"], {"sequence": test_in})[0]
        assert test_out.shape == (1, cpu_model.num_classes)
        print(f"  [OK] Validasi ONNX Runtime Mobile BERHASIL! (Output shape: {test_out.shape})")
    except Exception as e:
        print(f"  [WARNING] Validasi ONNX Runtime: {e}")

    return out_path


def export_model_tflite(
    model,
    out_path: str,
    labels_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    arch: str = "mgn",
    seq_len: int = SEQ_LEN,
    feat_dim: int = EXTENDED_FEAT_DIM,
    hidden1: int = DEFAULT_HIDDEN1,
    hidden2: int = DEFAULT_HIDDEN2,
    head_dim: int = DEFAULT_HEAD_DIM,
    flutter_dir: Optional[str] = None,
) -> str:
    """
    Ekspor model PyTorch (MGN / BiLSTM) ke TensorFlow Lite murni (TFLITE_BUILTINS)
    beserta labels.txt dan sinkronisasi ke folder mobile assets.
    """
    try:
        from export_tflite import build_keras_mgn_model, build_keras_lstm_model, transfer_weights
        import tensorflow as tf
        import shutil

        print(f"\nMengekspor model ke TensorFlow Lite (Mobile Ready): {out_path} ...")
        num_classes = len(class_names) if class_names else getattr(model, "num_classes", 20)
        if arch == "mgn":
            keras_model, layers_dict = build_keras_mgn_model(
                feat_dim=feat_dim,
                seq_len=seq_len,
                num_classes=num_classes,
                conv_dim=hidden1,
                lstm_hidden=hidden2,
                head_dim=head_dim,
            )
        else:
            keras_model, layers_dict = build_keras_lstm_model(
                feat_dim=feat_dim,
                seq_len=seq_len,
                num_classes=num_classes,
                hidden1=hidden1,
                hidden2=hidden2,
                head_dim=head_dim,
            )

        sd = model.state_dict()
        transfer_weights(arch, sd, layers_dict)

        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_bytes = converter.convert()

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(tflite_bytes)
        print(f"  [OK] Model TFLite berhasil disimpan di: {out_path} ({len(tflite_bytes)/1024:.1f} KB)")

        if labels_path and class_names:
            os.makedirs(os.path.dirname(os.path.abspath(labels_path)), exist_ok=True)
            with open(labels_path, "w", encoding="utf-8") as f:
                for name in class_names:
                    f.write(f"{name}\n")
            print(f"  [OK] Labels text berhasil disimpan di: {labels_path} ({len(class_names)} baris)")

        if flutter_dir and os.path.exists(flutter_dir):
            dest_tflite = os.path.join(flutter_dir, "sign_model.tflite")
            dest_labels = os.path.join(flutter_dir, "labels.txt")
            shutil.copyfile(out_path, dest_tflite)
            if labels_path and os.path.exists(labels_path):
                shutil.copyfile(labels_path, dest_labels)
            print(f"  [FLUTTER SYNC] Berhasil disinkronisasi ke: {flutter_dir}")

        return out_path
    except Exception as e:
        print(f"  [WARNING] Gagal mengekspor TFLite: {e}")
        return ""


# %%
# =========================================================
# 9. PIPELINE TRAINING LENGKAP
# =========================================================

def train_pipeline(
    arch: str = "mgn",
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    hidden1: int = DEFAULT_HIDDEN1,
    hidden2: int = DEFAULT_HIDDEN2,
    head_dim: int = DEFAULT_HEAD_DIM,
    dropout: float = DEFAULT_DROPOUT,
    smoke: bool = False,
    re_extract: bool = False,
    num_workers: int = 1,
    split_mode: str = "stratified",
    data_dir: str = DATA_DIR,
    cache_file: str = CACHE_FILE,
    labels_file: str = LABELS_FILE,
    model_save_pt: str = MODEL_SAVE_PT,
    model_save_onnx: str = MODEL_SAVE_ONNX,
    model_save_tflite: Optional[str] = MODEL_SAVE_TFLITE,
    labels_save_txt: Optional[str] = LABELS_SAVE_TXT,
    flutter_dir: Optional[str] = FLUTTER_ASSETS_DIR,
    pretrained_path: Optional[str] = None,
    feat_dim: Optional[int] = None,
    seed: Optional[int] = None,
):
    import torch
    import torch.nn as nn

    print_config()
    print(f"Dataset Dir  : {data_dir}")
    print(f"Cache File   : {cache_file}")
    print(f"Labels File  : {labels_file}")
    if pretrained_path:
        print(f"Pretrained   : {pretrained_path} (Transfer Learning Active)")
    print(f"Arsitektur   : {arch.upper()} ({'Multi-Granularity Network: Conv1D + BiLSTM + Attention' if arch == 'mgn' else 'Vanilla BiLSTM'})")

    # Set Random Seed
    actual_seed = SEED if seed is None else int(seed)
    import random
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    rng = np.random.default_rng(actual_seed)
    torch.manual_seed(actual_seed)

    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice yang digunakan: {device} | Random Seed: {actual_seed}")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(actual_seed)
        torch.backends.cudnn.benchmark = True

    # 2. Dataset Muat / Ekstrak
    seqs, lengths, labels, signers, filenames, class_names = build_or_load_dataset(
        data_dir=data_dir,
        cache_file=cache_file,
        labels_file=labels_file,
        re_extract=re_extract,
        smoke=smoke,
        num_workers=num_workers,
    )
    num_classes = len(class_names)
    print(f"Total Sampel: {len(labels)} | Jumlah Kelas: {num_classes}")
    assert len(labels) > 0, "Dataset kosong!"
    assert np.all(labels >= 0), "Ditemukan label kelas bernilai negatif!"
    assert np.all(labels < num_classes), f"Ditemukan label kelas >= {num_classes} di luar jangkauan class_names!"

    # 3. Train-Validation Split (Stratified atau Holdout Signer)
    if smoke:
        all_indices = np.arange(len(labels))
        rng.shuffle(all_indices)
        split_point = max(1, int(len(all_indices) * 0.8))
        train_idx = all_indices[:split_point]
        val_idx = all_indices[split_point:]
        if len(val_idx) == 0:
            val_idx = train_idx[:1]
    elif split_mode == "stratified":
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
        print(f"Mode Split: STRATIFIED 80/20 ({len(train_idx)} train, {len(val_idx)} val terdistribusi seimbang)")
    else:
        val_mask = np.isin(signers, VAL_SIGNERS)
        train_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]
        print(f"Mode Split: HOLDOUT SIGNER {VAL_SIGNERS} ({len(train_idx)} train, {len(val_idx)} val)")

    print(f"Train samples: {len(train_idx)} | Validation samples: {len(val_idx)}")
    if len(val_idx) == 0:
        raise ValueError("Set validasi kosong! Periksa konfigurasi split.")

    # 4. Bangun Model (MGN atau BiLSTM)
    # feat_dim default: MGN selalu 400D extended; LSTM default 186D, bisa di-override
    # via --feat-dim 400 untuk memanfaatkan velocity & relasi kinematik pada arsitektur LSTM.
    if feat_dim is None:
        feat_dim = EXTENDED_FEAT_DIM if arch == "mgn" else FEAT_DIM
    else:
        feat_dim = int(feat_dim)
    if arch == "mgn":
        model = build_mgn_model(
            feat_dim=feat_dim,
            num_classes=num_classes,
            conv_dim=hidden1,
            lstm_hidden=hidden2,
            head_dim=head_dim,
            dropout=dropout,
        ).to(device)
    else:
        model = build_lstm_model(
            feat_dim=feat_dim,
            num_classes=num_classes,
            hidden1=hidden1,
            hidden2=hidden2,
            head_dim=head_dim,
            dropout=dropout,
        ).to(device)

    # Transfer Learning: Muat bobot pretrained jika ada
    if pretrained_path and os.path.exists(pretrained_path):
        try:
            pt_ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
            pt_dict = pt_ckpt.get("ema_state_dict") or pt_ckpt.get("model_state_dict")
            m_dict = model.state_dict()
            filtered_dict = {
                k: v for k, v in pt_dict.items()
                if k in m_dict and v.shape == m_dict[k].shape and not k.startswith("head.4")
            }
            m_dict.update(filtered_dict)
            model.load_state_dict(m_dict)
            print(f"[TRANSFER LEARNING] Berhasil memuat {len(filtered_dict)} layer backbone dari: {pretrained_path}")
            head_params = list(model.head.parameters())
            backbone_params = [p for n, p in model.named_parameters() if not n.startswith("head.")]
            optimizer = torch.optim.AdamW([
                {"params": backbone_params, "lr": lr * 0.2},
                {"params": head_params, "lr": lr},
            ], weight_decay=weight_decay)
        except Exception as e:
            print(f"[TRANSFER LEARNING] Peringatan: Gagal memuat pretrained ({e}). Menggunakan inisialisasi acak.")
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ema = ModelEMA(model, decay=0.995)

    # Class Weights untuk mencegah bias/collapse ke kata tertentu
    train_labels = labels[train_idx]
    counts = np.bincount(train_labels, minlength=num_classes)
    class_weights = len(train_labels) / (num_classes * np.maximum(counts, 1).astype(np.float32))
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    FocalLossClass = get_focal_loss_class()
    criterion = FocalLossClass(gamma=2.0, weight=weights_tensor, label_smoothing=LABEL_SMOOTHING)
    print("Fungsi Loss : Focal Loss (gamma=2.0, label_smoothing=" + str(LABEL_SMOOTHING) + ") dengan Class Weighting")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
    )

    print("\nRingkasan Model:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameter yang Dilatih: {total_params:,} (Fitur input: {feat_dim} dim)")

    # Autocast untuk GPU RTX
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = 0.0
    best_epoch = 0

    print("\n" + "=" * 65)
    print(f"{'Epoch':^7} | {'Train Loss':^10} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^10} | {'LR':^8}")
    print("=" * 65)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x_b, y_b in iterate_batches(
            train_idx, seqs, lengths, labels,
            batch_size=batch_size, seq_len=SEQ_LEN,
            feat_dim=feat_dim, is_train=True, augment=True, rng=rng,
        ):
            x_b = x_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x_b)
                loss = criterion(logits, y_b)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            train_loss += float(loss.item()) * len(y_b)
            preds = logits.argmax(dim=-1)
            train_correct += int((preds == y_b).sum().item())
            train_total += len(y_b)

        epoch_train_loss = train_loss / max(1, train_total)
        epoch_train_acc = train_correct / max(1, train_total)

        # Evaluasi dengan model EMA
        val_loss, val_acc, _, _ = evaluate_model(
            ema.shadow, val_idx, seqs, lengths, labels,
            device=device, batch_size=batch_size, seq_len=SEQ_LEN,
            feat_dim=feat_dim,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)

        is_best = val_acc >= best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.shadow.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "meta": {
                    "arch": arch,
                    "seq_len": SEQ_LEN,
                    "feat_dim": feat_dim,
                    "num_classes": num_classes,
                    "target_fps": TARGET_FPS,
                    "hidden1": hidden1,
                    "hidden2": hidden2,
                    "head_dim": head_dim,
                    "class_names": class_names,
                },
            }
            os.makedirs(os.path.dirname(os.path.abspath(model_save_pt)), exist_ok=True)
            torch.save(checkpoint_data, model_save_pt)

        best_mark = " *" if is_best else ""
        print(
            f"{epoch:^7d} | {epoch_train_loss:^10.4f} | {epoch_train_acc:^10.4f} | "
            f"{val_loss:^10.4f} | {val_acc:^10.4f} | {current_lr:^8.1e}{best_mark}"
        )

    print("=" * 65)
    print(f"Training Selesai! Model terbaik di Epoch {best_epoch} dengan Akurasi Val: {best_val_acc:.4f}")
    print(f"Checkpoint tersimpan di: {model_save_pt}")

    # 5. Evaluasi Detail Model Terbaik
    print("\nMelakukan evaluasi konsensus video pada model terbaik...")
    best_checkpoint = torch.load(model_save_pt, map_location=device, weights_only=False)
    if arch == "mgn":
        eval_model = build_mgn_model(
            feat_dim=feat_dim,
            num_classes=num_classes,
            conv_dim=hidden1,
            lstm_hidden=hidden2,
            head_dim=head_dim,
            dropout=dropout,
        ).to(device)
    else:
        eval_model = build_lstm_model(
            feat_dim=feat_dim,
            num_classes=num_classes,
            hidden1=hidden1,
            hidden2=hidden2,
            head_dim=head_dim,
            dropout=dropout,
        ).to(device)

    eval_model.load_state_dict(best_checkpoint["ema_state_dict"])
    eval_model.eval()

    video_acc = evaluate_video_consensus(
        eval_model, val_idx, seqs, lengths, labels,
        device=device, seq_len=SEQ_LEN, feat_dim=feat_dim,
    )
    print(f"\n>>> AKURASI AKHIR KONSENSUS VIDEO (1 Video = 1 Keputusan): {video_acc * 100:.2f}% <<<")

    _, _, final_preds, final_targets = evaluate_model(
        eval_model, val_idx, seqs, lengths, labels,
        device=device, batch_size=batch_size, seq_len=SEQ_LEN, feat_dim=feat_dim,
    )

    print("\n=== CLASSIFICATION REPORT ===")
    unique_targets = sorted(list(set(final_targets.tolist() + final_preds.tolist())))
    target_names = [class_names[i] for i in unique_targets]
    print(classification_report(final_targets, final_preds, labels=unique_targets, target_names=target_names, zero_division=0))

    # 6. Ekspor ke ONNX
    export_model_onnx(eval_model, model_save_onnx, seq_len=SEQ_LEN, feat_dim=feat_dim)

    # 7. Ekspor ke TensorFlow Lite (Mobile Ready)
    if model_save_tflite:
        export_model_tflite(
            eval_model,
            out_path=model_save_tflite,
            labels_path=labels_save_txt,
            class_names=class_names,
            arch=arch,
            seq_len=SEQ_LEN,
            feat_dim=feat_dim,
            hidden1=hidden1,
            hidden2=hidden2,
            head_dim=head_dim,
            flutter_dir=flutter_dir,
        )

    # 8. Informasi Hasil Model
    print(f"\n[INFO] Model Inferensi Berhasil Disimpan:")
    print(f"  - PyTorch Checkpoint : {model_save_pt} (Arsitektur: {arch.upper()}, Dimensi: {feat_dim})")
    print(f"  - ONNX Model         : {model_save_onnx} (untuk inferensi webcam/desktop via camera.py)")
    if model_save_tflite:
        print(f"  - TFLite Model       : {model_save_tflite} (untuk inferensi mobile Flutter / Android)")
    if labels_save_txt:
        print(f"  - Labels Text        : {labels_save_txt}")

    return eval_model


# %%
# =========================================================
# 10. CLI ENTRY POINT
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Training Multi-Landmark BISINDO MGN / BiLSTM Classifier")
    parser.add_argument("--dataset", type=str, default="wl", choices=["wl", "manual", "gabungan"], help="Pilihan dataset: 'wl' (WL-BISINDO), 'manual' (Manual 20 kelas), atau 'gabungan' (Dataset Gabungan 55 Kelas: Manual + WL + Abjad)")
    parser.add_argument("--data-dir", type=str, default=None, help="Folder dataset (default: datasets/WL-BISINDO, datasets/manual, atau data)")
    parser.add_argument("--cache-file", type=str, default=None, help="File cache landmark .npz")
    parser.add_argument("--labels-file", type=str, default=None, help="File JSON daftar label")
    parser.add_argument("--model-pt", type=str, default=None, help="Nama file simpan PyTorch model .pt")
    parser.add_argument("--model-onnx", type=str, default=None, help="Nama file simpan ONNX model .onnx")
    parser.add_argument("--arch", type=str, default="mgn", choices=["mgn", "lstm"], help="Arsitektur model: 'mgn' (Multi-Granularity Network) atau 'lstm' (default: mgn)")
    parser.add_argument("--smoke", action="store_true", help="Jalankan smoke test cepat pada subset dataset")
    parser.add_argument("--re-extract", action="store_true", help="Paksa ekstraksi ulang landmark ke cache .npz")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=f"Jumlah epoch training (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=None, help="Ukuran batch (default: 32 untuk wl & gabungan, 16 untuk manual)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help=f"Learning rate awal (default: {DEFAULT_LR})")
    parser.add_argument("--hidden1", type=int, default=DEFAULT_HIDDEN1, help=f"Hidden/Conv size Layer 1 (default: {DEFAULT_HIDDEN1})")
    parser.add_argument("--hidden2", type=int, default=DEFAULT_HIDDEN2, help=f"Hidden size BiLSTM Layer 2 (default: {DEFAULT_HIDDEN2})")
    parser.add_argument("--finetune", action="store_true", help="Aktifkan transfer learning dari bisindo_lstm_best.pt")
    parser.add_argument("--pretrained", type=str, default=None, help="Path ke model pretrained PyTorch (.pt)")
    cpu_cnt = os.cpu_count() or 4
    default_workers = min(4, max(2, cpu_cnt // 2))
    parser.add_argument("--workers", type=int, default=default_workers, help=f"Jumlah worker process ekstraksi paralel (default: {default_workers}, rekomendasi 2-4 worker)")
    parser.add_argument("--split", type=str, default=None, choices=["stratified", "holdout"], help="Metode split: 'stratified' (rekomendasi seimbang) atau 'holdout' (unseen signer)")
    parser.add_argument("--feat-dim", type=int, default=None, help="Dimensi fitur input model. Default: 400 untuk MGN, 186 untuk LSTM. Gunakan 400 untuk melatih LSTM dengan fitur extended (velocity + kinematik).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed untuk training (default: 42). Gunakan seed berbeda untuk ensemble multi-model.")
    parser.add_argument("--model-tflite", type=str, default=None, help="Nama file simpan TFLite model .tflite (default: models/sign_model.tflite)")
    parser.add_argument("--model-labels-txt", type=str, default=None, help="Nama file simpan labels text .txt (default: models/labels.txt)")
    parser.add_argument("--flutter-dir", type=str, default=FLUTTER_ASSETS_DIR, help=f"Folder assets/models Flutter untuk auto-sync (default: {FLUTTER_ASSETS_DIR})")
    parser.add_argument("--no-tflite", action="store_true", help="Nonaktifkan ekspor model TFLite")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Tentukan path dan konfigurasi berdasarkan argumen --dataset
    model_save_tflite = None if args.no_tflite else (args.model_tflite or MODEL_SAVE_TFLITE)
    labels_save_txt = args.model_labels_txt or LABELS_SAVE_TXT
    flutter_dir = None if args.no_tflite else args.flutter_dir

    if args.dataset == "gabungan":
        data_dir = args.data_dir or "data"
        cache_candidates = [
            args.cache_file,
            "data/gabungan_holistic_lstm_cache.npz",
            "gabungan_holistic_lstm_cache.npz",
        ]
        cache_file = next((f for f in cache_candidates if f and os.path.exists(f)), "data/gabungan_holistic_lstm_cache.npz")
        candidate_labels = [
            args.labels_file,
            "labels/gabungan_labels.json",
            "gabungan_labels.json",
        ]
        labels_file = next((f for f in candidate_labels if f and os.path.exists(f)), "labels/gabungan_labels.json")
        model_save_pt = args.model_pt or "models/gabungan_lstm_best.pt"
        model_save_onnx = args.model_onnx or "models/gabungan_lstm.onnx"
        batch_size = args.batch_size or 32
        split_mode = args.split or "stratified"
        pretrained_path = args.pretrained
    elif args.dataset == "manual":
        data_dir = args.data_dir or "datasets/manual"
        cache_candidates = [
            args.cache_file,
            "data/manual_holistic_lstm_cache.npz",
            "manual_holistic_lstm_cache.npz",
        ]
        cache_file = next((f for f in cache_candidates if f and os.path.exists(f)), "data/manual_holistic_lstm_cache.npz")
        candidate_labels = [
            args.labels_file,
            "labels/manual_labels.json",
            os.path.join(data_dir, "labeled.json"),
            os.path.join(data_dir, "manual_label.json"),
            "manual_labels.json",
        ]
        labels_file = next((f for f in candidate_labels if f and os.path.exists(f)), "labels/manual_labels.json")
        model_save_pt = args.model_pt or "models/manual_lstm_best.pt"
        model_save_onnx = args.model_onnx or "models/manual_lstm.onnx"
        batch_size = args.batch_size or 16
        split_mode = args.split or "stratified"
        # Hanya gunakan transfer learning jika eksplisit diminta pengguna (--finetune atau --pretrained)
        if args.finetune or args.pretrained:
            pretrained_candidates = [
                args.pretrained,
                "models/bisindo_lstm_best.pt",
                "bisindo_lstm_best.pt",
            ]
            pretrained_path = next((f for f in pretrained_candidates if f and os.path.exists(f)), None)
        else:
            pretrained_path = None
    else:
        data_dir = args.data_dir or DATA_DIR
        cache_candidates = [
            args.cache_file,
            "data/wl_holistic_lstm_cache.npz",
            CACHE_FILE,
        ]
        cache_file = next((f for f in cache_candidates if f and os.path.exists(f)), "data/wl_holistic_lstm_cache.npz")
        label_candidates = [
            args.labels_file,
            "labels/wl_labels.json",
            LABELS_FILE,
        ]
        labels_file = next((f for f in label_candidates if f and os.path.exists(f)), "labels/wl_labels.json")
        model_save_pt = args.model_pt or MODEL_SAVE_PT
        model_save_onnx = args.model_onnx or MODEL_SAVE_ONNX
        batch_size = args.batch_size or DEFAULT_BATCH_SIZE
        split_mode = args.split or "stratified"
        pretrained_path = args.pretrained

    train_pipeline(
        arch=args.arch,
        epochs=args.epochs,
        batch_size=batch_size,
        lr=args.lr,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        smoke=args.smoke,
        re_extract=args.re_extract,
        num_workers=args.workers,
        split_mode=split_mode,
        data_dir=data_dir,
        cache_file=cache_file,
        labels_file=labels_file,
        model_save_pt=model_save_pt,
        model_save_onnx=model_save_onnx,
        model_save_tflite=model_save_tflite,
        labels_save_txt=labels_save_txt,
        flutter_dir=flutter_dir,
        pretrained_path=pretrained_path,
        feat_dim=args.feat_dim,
        seed=args.seed,
    )
