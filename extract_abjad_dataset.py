"""
extract_abjad_dataset.py - Ekstraksi Landmark Abjad BISINDO A-Z (186D) ke Cache Sekuens.

Membaca dataset gambar abjad BISINDO dari datasets/bisindo/images/ (A-Z),
mengekstrak 186D landmark (MediaPipe Hands & Pose) yang kompatibel dengan
holistic_extractor.py, mengonstruksi sekuens 32-frame dengan micro-jitter realistis,
dan menyimpannya ke data/abjad_holistic_lstm_cache.npz.
"""

import argparse
import glob
import os
import sys
import time
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from holistic_extractor import EXTRACTOR_VERSION, FEAT_DIM, SELECTED_POSE_INDICES

DEFAULT_IMAGES_DIR = "datasets/bisindo/images"
DEFAULT_OUT_CACHE = "data/abjad_holistic_lstm_cache.npz"
ALPHABET = [chr(i) for i in range(ord("A"), ord("Z") + 1)]


def extract_single_image(
    img_path: str,
    mp_hands,
    mp_pose,
) -> np.ndarray:
    """
    Mengekstrak vektor 186 dimensi dari 1 citra abjad.
    Menggunakan koordinat tubuh-sentris yang konsisten dengan holistic_extractor.py.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    max_dim = float(max(w, h)) if max(w, h) > 0 else 1.0
    sx = float(w) / max_dim
    sy = float(h) / max_dim
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    res_hands = mp_hands.process(rgb)
    if not res_hands.multi_hand_landmarks:
        return None

    res_pose = mp_pose.process(rgb)

    # 1. Pose & Torso Reference
    pose_present = res_pose.pose_landmarks is not None
    if pose_present:
        p_raw = np.array(
            [[lm.x * sx, lm.y * sy, lm.z * sx] for lm in res_pose.pose_landmarks.landmark],
            dtype=np.float32,
        )
        ls = p_raw[11]
        rs = p_raw[12]
        torso_center = (ls + rs) / 2.0
        torso_scale = float(np.linalg.norm(ls[:2] - rs[:2]))
        if torso_scale < 1e-3:
            torso_scale = 0.25
        pose_feat = (p_raw[SELECTED_POSE_INDICES] - torso_center) / torso_scale
    else:
        torso_center = np.array([0.5 * sx, 0.5 * sy, 0.0], dtype=np.float32)
        torso_scale = 0.25
        pose_feat = np.zeros((len(SELECTED_POSE_INDICES), 3), dtype=np.float32)

    # 2. Hands (Left & Right Assignment)
    lh_feat = np.zeros((21, 3), dtype=np.float32)
    rh_feat = np.zeros((21, 3), dtype=np.float32)
    has_lh, has_rh = False, False

    num_detected_hands = len(res_hands.multi_hand_landmarks)
    for lms, handedness in zip(res_hands.multi_hand_landmarks, res_hands.multi_handedness):
        lbl = handedness.classification[0].label  # 'Left' or 'Right'
        pts = np.array([[lm.x * sx, lm.y * sy, lm.z * sx] for lm in lms.landmark], dtype=np.float32)
        norm_pts = (pts - torso_center) / torso_scale

        # Jika 2 tangan terdeteksi, gunakan x pergelangan tangan untuk memisahkan kiri vs kanan
        wrist_x = pts[0, 0]
        if num_detected_hands >= 2:
            is_left_visual = wrist_x < (0.5 * sx)
        else:
            is_left_visual = (lbl == "Left")

        if is_left_visual:
            lh_feat = norm_pts
            has_lh = True
        else:
            rh_feat = norm_pts
            has_rh = True

    # 3. Flags kehadiran [has_lh, has_rh, has_pose]
    flags = np.array([float(has_lh), float(has_rh), float(pose_present)], dtype=np.float32)

    vec186 = np.concatenate([lh_feat.ravel(), rh_feat.ravel(), pose_feat.ravel(), flags])
    return vec186.astype(np.float32)


def build_abjad_dataset(
    images_dir: str = DEFAULT_IMAGES_DIR,
    out_cache_path: str = DEFAULT_OUT_CACHE,
    samples_per_letter: int = 40,
    seq_len: int = 32,
    jitter_std: float = 0.003,
    seed: int = 42,
) -> str:
    print(f"\n=========================================================")
    print(f" Memulai Ekstraksi Dataset Abjad BISINDO A-Z")
    print(f" Direktori Citra   : {images_dir}")
    print(f" Target Sampel/Huruf: {samples_per_letter} sekuens")
    print(f" Panjang Sekuens   : {seq_len} frame")
    print(f" Output Cache      : {out_cache_path}")
    print(f"=========================================================\n")

    rng = np.random.default_rng(seed)

    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.3,
    )
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.3,
    )

    all_sequences = []
    all_lengths = []
    all_labels = []
    all_filenames = []
    all_signers = []

    total_images_processed = 0
    t0 = time.time()

    for label_id, letter in enumerate(ALPHABET):
        # Kumpulkan file citra dari train dan val
        patterns = [
            os.path.join(images_dir, "train", letter, "*.jpg"),
            os.path.join(images_dir, "train", letter, "*.png"),
            os.path.join(images_dir, "val", letter, "*.jpg"),
            os.path.join(images_dir, "val", letter, "*.png"),
        ]
        image_files = []
        for pat in patterns:
            image_files.extend(glob.glob(pat))

        # Acak urutan agar dapat sampel beragam
        rng.shuffle(image_files)

        letter_seqs = 0
        for fpath in image_files:
            if letter_seqs >= samples_per_letter:
                break

            total_images_processed += 1
            vec186 = extract_single_image(fpath, mp_hands, mp_pose)
            if vec186 is None:
                continue

            # Buat sekuens 32 frame dengan natural micro-jitter pada koordinat aktif
            base_seq = np.tile(vec186, (seq_len, 1))
            if jitter_std > 0.0:
                noise = rng.normal(0.0, jitter_std, size=(seq_len, FEAT_DIM - 3)).astype(np.float32)
                # Terapkan noise hanya pada koordinat yang aktif (tidak 0)
                mask = (base_seq[:, :-3] != 0.0).astype(np.float32)
                base_seq[:, :-3] += noise * mask

            all_sequences.append(base_seq)
            all_lengths.append(seq_len)
            all_labels.append(label_id)
            all_filenames.append(os.path.basename(fpath))
            all_signers.append(99)  # Signer ID 99 untuk dataset citra abjad
            letter_seqs += 1

        print(f"  [{letter}] Berhasil mengekstrak {letter_seqs}/{samples_per_letter} sampel sekuens.")

    mp_hands.close()
    mp_pose.close()

    elapsed = time.time() - t0
    num_samples = len(all_sequences)
    print(f"\n[Selesai] Memproses {total_images_processed} gambar dalam {elapsed:.1f}s ({num_samples} sekuens berhasil diekstrak).")

    # Konversi ke NumPy Array
    # Pad array ke MAX_FRAMES_PER_VIDEO (120) agar kompatibel penuh dengan skrip training
    max_frames = 120
    padded_seqs = np.zeros((num_samples, max_frames, FEAT_DIM), dtype=np.float32)
    for i, s in enumerate(all_sequences):
        padded_seqs[i, :seq_len] = s

    os.makedirs(os.path.dirname(os.path.abspath(out_cache_path)), exist_ok=True)
    np.savez_compressed(
        out_cache_path,
        sequences=padded_seqs,
        lengths=np.array(all_lengths, dtype=np.int32),
        labels=np.array(all_labels, dtype=np.int64),
        signers=np.array(all_signers, dtype=np.int32),
        filenames=np.array(all_filenames),
        class_names=np.array(ALPHABET),
        extractor_version=EXTRACTOR_VERSION,
    )
    cache_mb = os.path.getsize(out_cache_path) / (1024 * 1024)
    print(f"Cache Abjad berhasil disimpan di: {out_cache_path} ({cache_mb:.2f} MB)")
    return out_cache_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ekstraksi Dataset Abjad BISINDO A-Z")
    parser.add_argument("--images-dir", type=str, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--out-cache", type=str, default=DEFAULT_OUT_CACHE)
    parser.add_argument("--samples-per-letter", type=int, default=40)
    args = parser.parse_args()

    build_abjad_dataset(
        images_dir=args.images_dir,
        out_cache_path=args.out_cache,
        samples_per_letter=args.samples_per_letter,
    )
