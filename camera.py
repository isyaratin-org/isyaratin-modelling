"""
Testing Real-time Webcam & Video untuk Multi-Landmark BISINDO LSTM.

Fitur:
  - Live Webcam & Video file testing (--video path/ke/video.mp4).
  - MediaPipe Holistic: Menangkap 2 Tangan + Lengan/Bahu + Kepala/Wajah (186 dimensi).
  - Normalisasi Anatomis Body-Centric identik dengan train-new.py.
  - Mendukung checkpoint PyTorch (.pt) via CUDA/CPU atau ONNX Runtime (.onnx).
  - Stabilizer pengenalan kata (Debounce & Hold jeda) agar tidak berkedip.
  - Sentence Builder: Mengakumulasikan kata-kata isyarat menjadi kalimat.
  - Visualisasi landmark lengkap (Tangan, Lengan, Kepala, Titik Pusat Torso).

Cara Pakai:
  uv run python camera.py                           # Webcam live (PyTorch)
  uv run python camera.py --onnx bisindo_lstm.onnx  # Webcam live (ONNX CPU)
  uv run python camera.py --video sample.mp4        # Uji dari file video

Kontrol Keyboard:
  'q' atau 'ESC' : Keluar
  'c'            : Hapus riwayat kalimat & reset buffer
  'm'            : Toggle tampil/sembunyikan visualisasi landmark
  'h'            : Toggle bantuan hotkey
"""

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter, deque
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from holistic_extractor import (
    FEAT_DIM,
    EXTENDED_FEAT_DIM,
    SELECTED_POSE_INDICES,
    HEAD_INDICES,
    ARM_INDICES,
    TORSO_INDICES,
    extract_holistic_frame,
    compute_extended_features,
)
from kinematic_engine import (
    disambiguate_predictions,
    apply_bigram_prior,
    evaluate_idle_gate,
    extract_motion_kinematics,
)
DEFAULT_SEQ_LEN = 32
DEFAULT_TARGET_FPS = 12.0


# =========================================================
# 2. MODEL LOADER (PyTorch & ONNX - Mendukung MGN & BiLSTM)
# =========================================================

def load_inference_model(
    model_pt: Optional[str] = "models/gabungan_lstm_best.pt",
    model_onnx: Optional[str] = None,
    labels_path: str = "labels/gabungan_labels.json",
):
    """
    Muat model inferensi (MGN atau BiLSTM) dan metadata label.
    Otomatis mendeteksi arsitektur dan jumlah kelas dari bobot model.
    """
    # Auto-resolve path ke folder models/ atau labels/ jika hanya nama file yang dioper
    if model_pt and not os.path.exists(model_pt) and os.path.exists(os.path.join("models", model_pt)):
        model_pt = os.path.join("models", model_pt)
    if model_onnx and not os.path.exists(model_onnx) and os.path.exists(os.path.join("models", model_onnx)):
        model_onnx = os.path.join("models", model_onnx)
    if labels_path and not os.path.exists(labels_path) and os.path.exists(os.path.join("labels", labels_path)):
        labels_path = os.path.join("labels", labels_path)
    elif labels_path and not os.path.exists(labels_path) and os.path.exists(os.path.join("models", labels_path)):
        labels_path = os.path.join("models", labels_path)

    labels = []
    if labels_path and os.path.exists(labels_path):
        if labels_path.endswith(".txt"):
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = [line.strip() for line in f if line.strip()]
        else:
            with open(labels_path, "r", encoding="utf-8") as f:
                raw_labels = json.load(f)
                if isinstance(raw_labels, list):
                    labels = [item.get("gloss") or item.get("name") if isinstance(item, dict) else str(item) for item in raw_labels]
                else:
                    labels = []

    if model_onnx and os.path.exists(model_onnx):
        import onnxruntime as ort

        print(f"Memuat model ONNX: {model_onnx}")
        session = ort.InferenceSession(model_onnx, providers=["CPUExecutionProvider"])
        in_name = session.get_inputs()[0].name
        out_name = session.get_outputs()[0].name
        input_shape = session.get_inputs()[0].shape
        seq_len = int(input_shape[1]) if len(input_shape) > 1 and input_shape[1] is not None else DEFAULT_SEQ_LEN
        in_dim = int(input_shape[2]) if len(input_shape) > 2 and input_shape[2] is not None else FEAT_DIM
        is_extended = in_dim == EXTENDED_FEAT_DIM

        def predict_fn(window_np: np.ndarray) -> np.ndarray:
            # window_np: (SeqLen, 186)
            if is_extended and window_np.shape[-1] == FEAT_DIM:
                window_np = compute_extended_features(window_np)
            inp = window_np[None].astype(np.float32)
            probs = session.run([out_name], {in_name: inp})[0][0]
            return probs

        arch_name = f"ONNX (CPU, {'MGN 386d' if is_extended else f'LSTM {len(labels)} Kelas'})"
        return predict_fn, labels, seq_len, arch_name

    if model_pt and os.path.exists(model_pt):
        import torch
        import torch.nn as nn

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Memuat model PyTorch: {model_pt} pada device: {device}")
        ckpt = torch.load(model_pt, map_location=device, weights_only=False)

        meta = ckpt.get("meta", {})
        if not meta:
            meta = {
                "arch": ckpt.get("arch", "lstm"),
                "seq_len": int(ckpt.get("seq_len", DEFAULT_SEQ_LEN)),
                "num_classes": int(ckpt.get("num_classes", len(labels) or 55)),
                "hidden1": int(ckpt.get("hidden1", 128)),
                "hidden2": int(ckpt.get("hidden2", 96)),
                "head_dim": int(ckpt.get("head_dim", 192)),
                "class_names": ckpt.get("class_names", labels),
            }

        seq_len = int(meta.get("seq_len", DEFAULT_SEQ_LEN))
        num_classes = int(meta.get("num_classes", len(labels) or 55))
        hidden1 = int(meta.get("hidden1", 128))
        hidden2 = int(meta.get("hidden2", 96))
        head_dim = int(meta.get("head_dim", 192))
        if ("class_names" in meta or "class_names" in ckpt) and not labels:
            labels = meta.get("class_names") or ckpt.get("class_names", [])

        state_dict = ckpt.get("ema_state_dict") or ckpt.get("model_state_dict") or ckpt.get("state_dict")

        # Deteksi Arsitektur: MGN vs Vanilla LSTM
        is_mgn = "conv_in.weight" in state_dict or meta.get("arch") == "mgn"
        feat_dim = int(meta.get("feat_dim", EXTENDED_FEAT_DIM if is_mgn else FEAT_DIM))

        if is_mgn:
            class TemporalAttention(nn.Module):
                def __init__(self, hidden_dim: int):
                    super().__init__()
                    self.attn = nn.Sequential(
                        nn.Linear(hidden_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1),
                    )

                def forward(self, x):
                    weights = torch.softmax(self.attn(x), dim=1)
                    pooled = torch.sum(x * weights, dim=1)
                    return pooled, weights

            class BisindoMGNClassifier(nn.Module):
                def __init__(self):
                    super().__init__()
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
                        dropout=0.3,
                    )
                    bilstm_out_dim = hidden2 * 2
                    self.attn_pool = TemporalAttention(bilstm_out_dim)
                    pooled_dim = bilstm_out_dim * 2
                    self.head = nn.Sequential(
                        nn.Linear(pooled_dim, head_dim),
                        nn.GELU(),
                        nn.Dropout(0.3),
                        nn.LayerNorm(head_dim),
                        nn.Linear(head_dim, num_classes),
                    )

                def forward(self, x):
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
                    return torch.softmax(self.head(pooled), dim=-1)

                @torch.no_grad()
                def predict_with_attention(self, x):
                    x = self.input_norm(x)
                    x_conv = self.conv_in(x.transpose(1, 2))
                    res = x_conv
                    c3 = self.conv3(x_conv)
                    c5 = self.conv5(x_conv)
                    x_micro = self.conv_act(self.conv_bn(c3 + c5) + res)
                    x_macro_in = x_micro.transpose(1, 2)
                    self.lstm.flatten_parameters()
                    h, _ = self.lstm(x_macro_in)
                    attn_out, weights = self.attn_pool(h)
                    max_out = h.amax(dim=1)
                    pooled = torch.cat([attn_out, max_out], dim=-1)
                    probs = torch.softmax(self.head(pooled), dim=-1)
                    return probs, weights

            model = BisindoMGNClassifier().to(device)
            arch_str = f"MGN-Attention ({device})"
        else:
            class BisindoLSTMClassifier(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.input_norm = nn.LayerNorm(feat_dim)
                    self.lstm1 = nn.LSTM(feat_dim, hidden1, batch_first=True, bidirectional=True)
                    self.drop1 = nn.Dropout(0.3)
                    self.lstm2 = nn.LSTM(hidden1 * 2, hidden2, batch_first=True, bidirectional=True)
                    self.drop2 = nn.Dropout(0.3)
                    self.head = nn.Sequential(
                        nn.Linear(hidden2 * 4, head_dim),
                        nn.GELU(),
                        nn.Dropout(0.3),
                        nn.LayerNorm(head_dim),
                        nn.Linear(head_dim, num_classes),
                    )

                def forward(self, x):
                    x = self.input_norm(x)
                    h1, _ = self.lstm1(x)
                    h1 = self.drop1(h1)
                    h2, _ = self.lstm2(h1)
                    h2 = self.drop2(h2)
                    pool = torch.cat([h2.mean(dim=1), h2.amax(dim=1)], dim=-1)
                    return torch.softmax(self.head(pool), dim=-1)

            model = BisindoLSTMClassifier().to(device)
            arch_str = f"BiLSTM ({device})"

        model.load_state_dict(state_dict)
        model.eval()

        @torch.no_grad()
        def predict_fn(window_np: np.ndarray) -> np.ndarray:
            # window_np: (SeqLen, 186)
            if is_mgn and window_np.shape[-1] == FEAT_DIM:
                window_np = compute_extended_features(window_np)
            t = torch.from_numpy(window_np[None].astype(np.float32)).to(device)
            probs = model(t)[0].cpu().numpy()
            return probs

        return predict_fn, labels, seq_len, arch_str

    raise FileNotFoundError(f"Model tidak ditemukan di {model_pt} maupun {model_onnx}")


# =========================================================
# 3. VISUALISASI LANDMARK & HUD TAMPILAN
# =========================================================

def draw_custom_holistic(frame: np.ndarray, res):
    """
    Gambar landmark anatomis dengan warna estetik:
      - Tangan: Cyan
      - Bahu & Lengan: Kuning
      - Kepala / Wajah: Biru Neon
    """
    h, w = frame.shape[:2]

    # 1. Gambar Tangan Kiri & Kanan
    for hand_lms, color in [(res.left_hand_landmarks, (255, 230, 0)), (res.right_hand_landmarks, (0, 230, 255))]:
        if hand_lms:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms.landmark]
            # Gambar titik
            for p in pts:
                cv2.circle(frame, p, 3, color, -1)
            # Hubungkan jari (tulang tangan)
            connections = mp.solutions.hands.HAND_CONNECTIONS
            for start_idx, end_idx in connections:
                cv2.line(frame, pts[start_idx], pts[end_idx], (200, 200, 200), 1)

    # 2. Gambar Pose (Lengan & Kepala)
    if res.pose_landmarks:
        p_pts = [(int(lm.x * w), int(lm.y * h)) for lm in res.pose_landmarks.landmark]

        # Kepala / Wajah (0..10)
        for idx in HEAD_INDICES:
            cv2.circle(frame, p_pts[idx], 3, (255, 120, 50), -1)

        # Lengan & Bahu
        arm_connections = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
        for s, e in arm_connections:
            if s < len(p_pts) and e < len(p_pts):
                cv2.line(frame, p_pts[s], p_pts[e], (0, 255, 128), 2)
                cv2.circle(frame, p_pts[s], 4, (0, 255, 255), -1)
                cv2.circle(frame, p_pts[e], 4, (0, 255, 255), -1)

        # Titik pusat torso
        ls, rs = p_pts[11], p_pts[12]
        center = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
        cv2.circle(frame, center, 6, (0, 255, 0), -1)


def draw_hud(
    frame: np.ndarray,
    status_text: str,
    top_preds: List[Tuple[str, float]],
    sentence: List[str],
    state: str,
    hold_remain: float,
    fps: float,
    model_info: str,
    mode: str = "hands_free",
    rec_progress: float = 0.0,
    duration_sec: float = 4.8,
    exit_delay_remain: float = 0.0,
    exit_delay_total: float = 1.5,
    show_help: bool = False,
):
    """
    Gambar Overlay HUD UI modern di atas frame webcam.
    """
    h, w = frame.shape[:2]

    # Bar Atas (Header Status)
    header_h = 75
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Info Model, FPS & Mode
    if mode == "hands_free":
        mode_desc = f"HANDS-FREE (Auto Tangan | Jeda Verif {exit_delay_total:.1f}s)"
    elif mode == "trigger":
        mode_desc = f"MANUAL [SPASI] ({duration_sec:.1f}s)"
    else:
        mode_desc = "AUTO STREAM"
    cv2.putText(frame, f"Model: {model_info} | FPS: {fps:.1f} | Mode: {mode_desc} (TAB)", (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

    # Status Utama
    if state == "recording":
        # Progress bar animasi perekaman isyarat
        cv2.rectangle(frame, (0, header_h - 6), (w, header_h), (35, 35, 45), -1)
        bar_w = int(w * max(0.0, min(1.0, rec_progress)))
        cv2.rectangle(frame, (0, header_h - 6), (bar_w, header_h), (0, 230, 255), -1)
        cv2.putText(frame, status_text, (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 230, 255), 2, cv2.LINE_AA)
    elif state == "verifying":
        # Progress bar hitung mundur delay verifikasi tangan hilang
        cv2.rectangle(frame, (0, header_h - 6), (w, header_h), (35, 35, 45), -1)
        frac = max(0.0, min(1.0, exit_delay_remain / max(0.01, exit_delay_total)))
        bar_w = int(w * (1.0 - frac))
        cv2.rectangle(frame, (0, header_h - 6), (bar_w, header_h), (0, 165, 255), -1)
        cv2.putText(frame, status_text, (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (0, 165, 255), 2, cv2.LINE_AA)
    elif state == "holding" and top_preds:
        label, conf = top_preds[0]
        status_str = f"TERKUNCI: {label.upper()} ({conf * 100:.1f}%) | Jeda {hold_remain:.1f}s"
        cv2.putText(frame, status_str, (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 128), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, status_text, (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

    # Sidebar Kanan (Top-3 Prediksi)
    if top_preds:
        side_w = 260
        side_x = w - side_w - 15
        side_y = 90
        side_h = 130
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (side_x, side_y), (side_x + side_w, side_y + side_h), (20, 20, 25), -1)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (side_x, side_y), (side_x + side_w, side_y + side_h), (80, 80, 90), 1)

        cv2.putText(frame, "PREDIKSI TERATAS:", (side_x + 10, side_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 200, 255), 1, cv2.LINE_AA)
        for i, (name, prob) in enumerate(top_preds[:3]):
            row_y = side_y + 48 + (i * 26)
            color = (0, 255, 128) if i == 0 else (220, 220, 220)
            cv2.putText(frame, f"{i+1}. {name}", (side_x + 10, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            cv2.putText(frame, f"{prob * 100:4.1f}%", (side_x + side_w - 65, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

            # Bar Probabilitas Mini
            bar_w = int((side_w - 90) * prob)
            cv2.rectangle(frame, (side_x + 10, row_y + 4), (side_x + 10 + bar_w, row_y + 7), color, -1)

    # Bar Bawah (Sentence Builder)
    bottom_h = 55
    overlay_bot = frame.copy()
    cv2.rectangle(overlay_bot, (0, h - bottom_h), (w, h), (15, 15, 20), -1)
    cv2.addWeighted(overlay_bot, 0.8, frame, 0.2, 0, frame)

    sentence_str = " ".join(sentence[-6:]) if sentence else "(Belum ada isyarat terkunci)"
    cv2.putText(frame, "KALIMAT: ", (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, sentence_str, (120, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


# =========================================================
# 4. WEBCAM & VIDEO INFERENCE LOOP
# =========================================================

def run_camera(
    source=0,
    model_pt: Optional[str] = "models/bisindo_lstm_best.pt",
    model_onnx: Optional[str] = None,
    labels_path: str = "labels/wl_labels.json",
    min_conf: float = 0.60,
    stable_n: int = 3,
    hold_sec: float = 1.5,
    checkpoint_every: int = 2,
    show_landmarks: bool = True,
    mode: str = "hands_free",  # "hands_free" (auto deteksi tangan + delay), "trigger" (manual SPASI), atau "auto" (continuous stream)
    duration_sec: float = 4.8,
    exit_delay_sec: float = 1.5,
    max_duration_sec: float = 7.0,
    min_duration_sec: float = 0.8,
):
    # 1. Muat Model
    predict_fn, labels, seq_len, model_info = load_inference_model(model_pt, model_onnx, labels_path)
    total_record_frames = max(seq_len, int(round(duration_sec * DEFAULT_TARGET_FPS)))
    print(f"\nModel aktif: {model_info} | Input Model: {seq_len} frames | Jendela Durasi: {duration_sec:.1f}s | Label: {len(labels)}")

    # 2. Buka Sumber Video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak dapat membuka input video/kamera: {source}")

    # Set resolusi webcam
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 3. Inisialisasi MediaPipe Holistic
    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    buffer = deque(maxlen=total_record_frames)
    record_buffer = []
    rec_progress = 0.0

    prob_history = deque(maxlen=4)
    recent_candidates = deque(maxlen=stable_n)
    motion_history = deque(maxlen=6)

    sample_interval = 1.0 / DEFAULT_TARGET_FPS  # 1.0 / 12.0 = 0.0833 detik
    last_sample_time = 0.0
    prev_coords = None
    prev_flags = None

    state = "idle" if mode in ["hands_free", "trigger"] else "listening"
    hold_until = 0.0
    locked_label = None
    frame_counter = 0
    sentence = []
    top_preds = []

    # State untuk Hands-free Trigger
    hand_lost_time = None
    rec_start_time = 0.0
    exit_delay_remain = 0.0

    fps_t0 = time.time()
    fps_counter = 0
    current_fps = 0.0

    window_name = "BISINDO Multi-Landmark Tester"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 680)

    print("\n=== KONTROL AKTIF ===")
    print("Mode 'HANDS-FREE'  : Gerakan otomatis direkam saat tangan muncul;")
    print(f"                     Saat tangan hilang/turun, verifikasi {exit_delay_sec:.1f}s lalu hasil langsung keluar.")
    print(f"Tekan [SPASI]      : Paksa mulai rekam isyarat ({duration_sec:.1f}s)")
    print("Tekan [TAB]        : Ganti Mode (HANDS-FREE -> MANUAL -> AUTO STREAM)")
    print("Tekan 'c'          : Reset Kalimat & Buffer")
    print("Tekan 'm'          : Toggle Tampilkan Landmark")
    print("Tekan 'q' atau ESC : Keluar")
    print(f"\nMode Awal: {mode.upper()}")
    print("Mulai membaca frame kamera...\n")

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            if isinstance(source, str):
                print("\nSelesai membaca file video.")
                break
            continue

        h, w = raw_frame.shape[:2]
        now = time.time()

        # Hitung FPS
        fps_counter += 1
        if now - fps_t0 >= 1.0:
            current_fps = fps_counter / (now - fps_t0)
            fps_counter = 0
            fps_t0 = now

        # MediaPipe Holistic Extraction pada frame ASLI (tanpa flip)
        rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(rgb)
        feat_vec = extract_holistic_frame(res, w, h)

        # Hitung Magnitudo Gerakan Tangan (Motion Energy Gate)
        coords = feat_vec[:-3].reshape(61, 3)
        flags = feat_vec[-3:]  # [lh_present, rh_present, pose_present]
        lh_present = flags[0] > 0.5
        rh_present = flags[1] > 0.5
        hand_detected = lh_present or rh_present

        instant_motion = 0.0
        if prev_coords is not None and prev_flags is not None:
            if lh_present and prev_flags[0] > 0.5:
                instant_motion += float(np.linalg.norm(coords[0] - prev_coords[0]))
            if rh_present and prev_flags[1] > 0.5:
                instant_motion += float(np.linalg.norm(coords[21] - prev_coords[21]))
        prev_coords = coords.copy()
        prev_flags = flags.copy()
        motion_history.append(instant_motion)
        avg_motion = float(np.mean(motion_history)) if motion_history else 0.0

        is_moving = hand_detected and (avg_motion > 0.035)

        # 12 FPS Throttled Tick
        is_sample_tick = (now - last_sample_time) >= sample_interval

        # ===================================================
        # STATE MACHINE: HANDS-FREE VS TRIGGER VS AUTO STREAM
        # ===================================================
        if mode == "hands_free":
            if state == "idle":
                exit_delay_remain = 0.0
                rec_progress = 0.0
                status_text = "Siap! Angkat tangan untuk mulai gerakan isyarat..."
                if hand_detected:
                    state = "recording"
                    record_buffer.clear()
                    rec_start_time = now
                    hand_lost_time = None
                    last_sample_time = now
                    record_buffer.append(feat_vec)
                    print("\n[HANDS-FREE] Tangan terdeteksi! Mulai merekam gerakan...")

            elif state == "recording":
                exit_delay_remain = 0.0
                cur_dur = now - rec_start_time
                rec_progress = min(1.0, cur_dur / max(1.0, max_duration_sec))

                if is_sample_tick:
                    last_sample_time = now
                    record_buffer.append(feat_vec)

                status_text = f"MEREKAM ISYARAT... {cur_dur:.1f}s ({len(record_buffer)} frames)"

                # Cek jika tangan menghilang
                if not hand_detected:
                    state = "verifying"
                    hand_lost_time = now
                    print(f"  [VERIFIKASI] Tangan turun. Menunggu delay {exit_delay_sec:.1f}s untuk konfirmasi selesai...")
                elif cur_dur >= max_duration_sec:
                    # Durasi maksimal tercapai -> lakukan inferensi langsung
                    print(f"  [BATAS WAKTU] Durasi maksimal ({max_duration_sec:.1f}s) tercapai, memproses...")
                    state = "verifying"
                    hand_lost_time = now - exit_delay_sec

            elif state == "verifying":
                elapsed_lost = now - hand_lost_time if hand_lost_time else exit_delay_sec
                exit_delay_remain = max(0.0, exit_delay_sec - elapsed_lost)

                if hand_detected:
                    # Tangan muncul kembali sebelum timeout -> Lanjutkan gerakan!
                    state = "recording"
                    hand_lost_time = None
                    exit_delay_remain = 0.0
                    print("  [LANJUT] Tangan terdeteksi kembali! Melanjutkan rekaman gerakan...")
                    if is_sample_tick:
                        last_sample_time = now
                        record_buffer.append(feat_vec)
                else:
                    status_text = f"TANGAN TURUN - Verifikasi dlm {exit_delay_remain:.1f}s (Angkat tgn jika lanjut)"

                    if elapsed_lost >= exit_delay_sec:
                        # Grace period habis! Gerakan dinyatakan tuntas.
                        min_frames = max(8, int(round(min_duration_sec * DEFAULT_TARGET_FPS)))
                        if len(record_buffer) >= min_frames:
                            rec_arr = np.stack(record_buffer)
                            sample_idx = np.linspace(0, len(rec_arr) - 1, seq_len).round().astype(np.int64)
                            window_arr = rec_arr[sample_idx]
                            raw_probs = predict_fn(window_arr)
                            # 1. Kinematic Disambiguation: Pembeda anatomis (Rumah vs 1-Tangan/A, Makan vs Terima Kasih vs Kamu, Tuli vs Ingat)
                            probs, reasons = disambiguate_predictions(raw_probs, labels, rec_arr)
                            # 2. Bigram Language Model Contextual Prior: Prioritas kelanjutan kata alami
                            last_word = sentence[-1] if sentence else None
                            probs, bg_reasons = apply_bigram_prior(probs, labels, last_word=last_word)
                            if bg_reasons:
                                reasons.extend(bg_reasons)
                            if reasons:
                                for r in reasons:
                                    print(f"  {r}")
                            order = np.argsort(probs)[::-1]
                            top_preds = [(labels[i] if i < len(labels) else f"Class_{i}", float(probs[i])) for i in order[:3]]

                            # 3. Idle / OOD Gate: Tolak gerakan bebas / bukan isyarat
                            km = extract_motion_kinematics(rec_arr)
                            is_valid_sign, gate_reason = evaluate_idle_gate(top_preds, km, min_conf=min_conf)

                            if not is_valid_sign:
                                state = "idle"
                                status_text = f"Ditolak: {gate_reason}"
                                print(f"  [GERAKAN NETRAL] {gate_reason}")
                            else:
                                best_label, best_conf = top_preds[0]
                                sentence.append(best_label)
                                state = "holding"
                                hold_until = now + hold_sec
                                dur_actual = len(record_buffer) / DEFAULT_TARGET_FPS
                                print(f"  [TERDETEKSI] -> {best_label.upper()} ({best_conf * 100:.1f}%) | Durasi: {dur_actual:.1f}s ({len(record_buffer)} frames)")
                                status_text = f"TERDETEKSI: {best_label.upper()} ({best_conf * 100:.1f}%)"
                        else:
                            state = "idle"
                            status_text = "Gerakan terlalu singkat (<0.8s), diabaikan. Siap..."
                            print("  [BATAL] Gerakan terlalu singkat (< 0.8s), diabaikan.")

                        record_buffer.clear()
                        hand_lost_time = None
                        exit_delay_remain = 0.0

            elif state == "holding":
                remain = hold_until - now
                exit_delay_remain = 0.0
                if remain <= 0:
                    state = "idle"
                    record_buffer.clear()
                    rec_progress = 0.0
                    status_text = "Siap untuk isyarat berikutnya (Angkat tangan)..."
                else:
                    status_text = f"Hasil: {top_preds[0][0].upper()} ({top_preds[0][1]*100:.0f}%) | Jeda {remain:.1f}s"

        elif mode == "trigger":
            if state == "idle":
                status_text = "Siap! Tekan [SPASI] untuk mulai isyarat..."
                rec_progress = 0.0
            elif state == "recording":
                if is_sample_tick:
                    last_sample_time = now
                    record_buffer.append(feat_vec)
                    rec_progress = len(record_buffer) / float(total_record_frames)

                    if len(record_buffer) >= total_record_frames:
                        rec_arr = np.stack(record_buffer)
                        sample_idx = np.linspace(0, len(rec_arr) - 1, seq_len).round().astype(np.int64)
                        window_arr = rec_arr[sample_idx]
                        raw_probs = predict_fn(window_arr)
                        probs, reasons = disambiguate_predictions(raw_probs, labels, rec_arr)
                        last_word = sentence[-1] if sentence else None
                        probs, bg_reasons = apply_bigram_prior(probs, labels, last_word=last_word)
                        if bg_reasons:
                            reasons.extend(bg_reasons)
                        if reasons:
                            for r in reasons:
                                print(f"  {r}")
                        order = np.argsort(probs)[::-1]
                        top_preds = [(labels[i] if i < len(labels) else f"Class_{i}", float(probs[i])) for i in order[:3]]

                        km = extract_motion_kinematics(rec_arr)
                        is_valid_sign, gate_reason = evaluate_idle_gate(top_preds, km, min_conf=min_conf)
                        if not is_valid_sign:
                            state = "idle"
                            status_text = f"Ditolak: {gate_reason}"
                            print(f"  [GERAKAN NETRAL] {gate_reason}")
                        else:
                            best_label, best_conf = top_preds[0]
                            sentence.append(best_label)
                            state = "holding"
                            hold_until = now + hold_sec
                            print(f"  [TERDETEKSI] -> {best_label.upper()} ({best_conf * 100:.1f}%)")
                            status_text = f"TERDETEKSI: {best_label.upper()} ({best_conf * 100:.1f}%)"
            elif state == "holding":
                remain = hold_until - now
                if remain <= 0:
                    state = "idle"
                    record_buffer.clear()
                    rec_progress = 0.0
                    status_text = f"Siap! Tekan [SPASI] untuk isyarat berikutnya ({duration_sec:.1f}s)..."
                else:
                    status_text = f"Hasil: {top_preds[0][0].upper()} ({top_preds[0][1]*100:.0f}%) | Jeda {remain:.1f}s"

        else:
            # MODE AUTO STREAM
            if is_sample_tick:
                last_sample_time = now
                buffer.append(feat_vec)

            if state == "holding":
                remain = hold_until - now
                if remain <= 0:
                    state = "listening"
                    locked_label = None
                    buffer.clear()
                    prob_history.clear()
                    recent_candidates.clear()
                    frame_counter = 0
                    status_text = "Siap untuk isyarat berikutnya..."
                else:
                    status_text = f"Kata terkunci! Istirahat jeda {remain:.1f}s..."
            else:
                if len(buffer) < seq_len:
                    status_text = f"Mengisi buffer sequence ({len(buffer)}/{total_record_frames})..."
                elif not hand_detected:
                    status_text = "(Menunggu tangan terdeteksi...)"
                    recent_candidates.clear()
                    prob_history.clear()
                elif not is_moving:
                    status_text = "(Tangan diam/relaks - Menunggu gerakan isyarat...)"
                    recent_candidates.clear()
                    prob_history.clear()
                else:
                    frame_counter += 1
                    if frame_counter % max(1, checkpoint_every) == 0:
                        buf_arr = np.array(buffer)
                        sample_idx = np.linspace(0, len(buf_arr) - 1, seq_len).round().astype(np.int64)
                        window_arr = buf_arr[sample_idx]
                        raw_probs = predict_fn(window_arr)
                        probs, reasons = disambiguate_predictions(raw_probs, labels, buf_arr)
                        prob_history.append(probs)
                        mean_probs = np.mean(prob_history, axis=0)

                        order = np.argsort(mean_probs)[::-1]
                        top_preds = [(labels[i] if i < len(labels) else f"Class_{i}", float(mean_probs[i])) for i in order[:3]]

                        best_label, best_conf = top_preds[0]
                        if best_conf >= min_conf:
                            recent_candidates.append(best_label)
                        else:
                            recent_candidates.append("?")

                        if (
                            len(recent_candidates) == recent_candidates.maxlen
                            and recent_candidates[0] != "?"
                            and len(set(recent_candidates)) == 1
                        ):
                            locked_label = recent_candidates[0]
                            hold_until = now + hold_sec
                            state = "holding"
                            sentence.append(locked_label)
                            print(f"  [TERDETEKSI] -> {locked_label.upper()} ({best_conf * 100:.1f}%)")
                            status_text = f"TERKUNCI: {locked_label.upper()}"
                        else:
                            agree = sum(1 for c in recent_candidates if c == best_label)
                            status_text = f"Kandidat: {best_label} ({best_conf * 100:.0f}%) | Stabil {agree}/{recent_candidates.maxlen}"
                    else:
                        status_text = "Menganalisis gerakan tangan & tubuh..."

        # Visualisasi Landmark pada raw_frame
        if show_landmarks:
            draw_custom_holistic(raw_frame, res)

        # Mirror tampilan webcam untuk kenyamanan pengguna
        if isinstance(source, int):
            display_frame = cv2.flip(raw_frame, 1)
        else:
            display_frame = raw_frame

        # Gambar UI HUD (teks tidak terbalik karena digambar setelah flip)
        hold_remain = max(0.0, hold_until - now) if state == "holding" else 0.0
        draw_hud(
            display_frame,
            status_text=status_text,
            top_preds=top_preds,
            sentence=sentence,
            state=state,
            hold_remain=hold_remain,
            fps=current_fps,
            model_info=model_info,
            mode=mode,
            rec_progress=rec_progress,
            duration_sec=duration_sec,
            exit_delay_remain=exit_delay_remain,
            exit_delay_total=exit_delay_sec,
        )

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break
        elif key == 32:  # SPASI: Paksa mulai rekam isyarat
            if state != "recording":
                state = "recording"
                record_buffer.clear()
                rec_start_time = now
                hand_lost_time = None
                rec_progress = 0.0
                print(f"\n[REKAM MANUAL] Mulai merekam gerakan isyarat...")
        elif key == 9:  # TAB: Toggle Mode Berputar: hands_free -> trigger -> auto -> hands_free
            if mode == "hands_free":
                mode = "trigger"
                state = "idle"
            elif mode == "trigger":
                mode = "auto"
                state = "listening"
            else:
                mode = "hands_free"
                state = "idle"

            record_buffer.clear()
            buffer.clear()
            prob_history.clear()
            recent_candidates.clear()
            hand_lost_time = None
            print(f"\n[MODE] Beralih ke MODE: {mode.upper()}")
        elif key == ord("c"):
            sentence.clear()
            record_buffer.clear()
            buffer.clear()
            prob_history.clear()
            recent_candidates.clear()
            hand_lost_time = None
            state = "idle" if mode in ["hands_free", "trigger"] else "listening"
            print("Riwayat kalimat & buffer dibersihkan.")
        elif key == ord("m"):
            show_landmarks = not show_landmarks
            print(f"Landmark visual: {'Aktif' if show_landmarks else 'Non-aktif'}")

    cap.release()
    holistic.close()
    cv2.destroyAllWindows()
    print("Camera test selesai.")


# =========================================================
# 5. CLI ENTRY POINT
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Live Camera & Video Tester Multi-Landmark BISINDO LSTM")
    parser.add_argument("--dataset", type=str, default="gabungan", choices=["gabungan", "manual", "wl"], help="Pilihan model dataset: 'gabungan' (default: 55 kelas), 'manual' (20 kelas), atau 'wl' (28 kelas)")
    parser.add_argument("--mode", type=str, default="hands_free", choices=["hands_free", "trigger", "auto"], help="Mode deteksi: 'hands_free' (auto tangan + delay verifikasi), 'trigger' (SPASI), 'auto' (stream)")
    parser.add_argument("--exit-delay", type=float, default=1.5, help="Jeda verifikasi selesai saat tangan hilang dalam detik (default: 1.5s, rentang 1-2 detik)")
    parser.add_argument("--source", type=str, default="0", help="Indeks kamera (misal 0) atau path ke file video .mp4")
    parser.add_argument("--video", type=str, default=None, help="Shortcut untuk menguji file video .mp4")
    parser.add_argument("--pt", type=str, default=None, help="Path checkpoint PyTorch (.pt)")
    parser.add_argument("--onnx", type=str, default=None, help="Path model ONNX (.onnx) opsional")
    parser.add_argument("--labels", type=str, default=None, help="Path file labels.json atau labels.txt")
    parser.add_argument("--min-conf", type=float, default=0.45, help="Ambang keyakinan minimum (default: 0.45)")
    parser.add_argument("--stable-n", type=int, default=3, help="Jumlah window berturut-turut yang harus setuju (default: 3)")
    parser.add_argument("--hold", type=float, default=1.2, help="Durasi jeda detik setelah kata terkunci (default: 1.2s)")
    parser.add_argument("--duration", type=float, default=None, help="Durasi jendela perekaman isyarat dalam detik (default: 3.5s untuk gabungan, 5.0s manual, 2.7s wl)")
    parser.add_argument("--max-duration", type=float, default=7.0, help="Durasi maksimal gerakan hands-free sebelum auto-cut (default: 7.0s)")
    parser.add_argument("--min-duration", type=float, default=0.8, help="Durasi minimal gerakan hands-free agar tidak false trigger (default: 0.8s)")
    parser.add_argument("--no-landmarks", action="store_true", help="Sembunyikan titik visual landmark")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    source = args.video if args.video else (int(args.source) if args.source.isdigit() else args.source)

    if args.dataset == "gabungan":
        pt_candidates = [
            args.pt,
            "models/gabungan_lstm_best.pt",
            "gabungan_lstm_best.pt",
        ]
        pt_path = next((f for f in pt_candidates if f and os.path.exists(f)), "models/gabungan_lstm_best.pt")
        onnx_candidates = [
            args.onnx,
            "models/gabungan_lstm.onnx",
            "gabungan_lstm.onnx",
        ]
        onnx_path = next((f for f in onnx_candidates if f and os.path.exists(f)), None)
        candidate_labels = [
            args.labels,
            "labels/gabungan_labels.json",
            "models/labels.txt",
            "gabungan_labels.json",
        ]
        labels_path = next((f for f in candidate_labels if f and os.path.exists(f)), "labels/gabungan_labels.json")
        duration_sec = args.duration if args.duration is not None else 3.5
    elif args.dataset == "manual":
        pt_candidates = [
            args.pt,
            "models/manual_lstm_best.pt",
            "manual_lstm_best.pt",
        ]
        pt_path = next((f for f in pt_candidates if f and os.path.exists(f)), "models/manual_lstm_best.pt")
        onnx_candidates = [
            args.onnx,
            "models/manual_lstm.onnx",
            "manual_lstm.onnx",
        ]
        onnx_path = next((f for f in onnx_candidates if f and os.path.exists(f)), None)
        candidate_labels = [
            args.labels,
            "labels/manual_labels.json",
            "datasets/manual/labeled.json",
            "datasets/manual/manual_label.json",
            "manual_labels.json",
        ]
        labels_path = next((f for f in candidate_labels if f and os.path.exists(f)), "labels/manual_labels.json")
        duration_sec = args.duration if args.duration is not None else 5.0
    else:
        pt_candidates = [
            args.pt,
            "models/bisindo_lstm_best.pt",
            "bisindo_lstm_best.pt",
        ]
        pt_path = next((f for f in pt_candidates if f and os.path.exists(f)), "models/bisindo_lstm_best.pt")
        onnx_candidates = [
            args.onnx,
            "models/bisindo_lstm.onnx",
            "bisindo_lstm.onnx",
        ]
        onnx_path = next((f for f in onnx_candidates if f and os.path.exists(f)), None)
        candidate_labels = [
            args.labels,
            "labels/wl_labels.json",
            "wl_labels.json",
        ]
        labels_path = next((f for f in candidate_labels if f and os.path.exists(f)), "labels/wl_labels.json")
        duration_sec = args.duration if args.duration is not None else 2.7

    run_camera(
        source=source,
        model_pt=pt_path,
        model_onnx=onnx_path,
        labels_path=labels_path,
        min_conf=args.min_conf,
        stable_n=args.stable_n,
        hold_sec=args.hold,
        show_landmarks=not args.no_landmarks,
        mode=args.mode,
        duration_sec=duration_sec,
        exit_delay_sec=args.exit_delay,
        max_duration_sec=args.max_duration,
        min_duration_sec=args.min_duration,
    )
