"""
Worker module untuk multiprocessing ekstraksi MediaPipe Holistic.
Dipisahkan dari train-new.py agar proses spawn di Windows tidak memicu konflik DLL / shm.dll.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from typing import Dict, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np

EXTRACTOR_VERSION = "2.1.0"

# Landmark Indices dari MediaPipe Pose (19 points)
HEAD_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ARM_INDICES = [11, 12, 13, 14, 15, 16]
TORSO_INDICES = [23, 24]
SELECTED_POSE_INDICES = HEAD_INDICES + ARM_INDICES + TORSO_INDICES

NUM_HAND_LANDMARKS = 21
NUM_POSE_LANDMARKS = len(SELECTED_POSE_INDICES)
FEAT_DIM = (NUM_HAND_LANDMARKS * 3 * 2) + (NUM_POSE_LANDMARKS * 3) + 3 # 186
# Extended Features: 186 static coords + 183 velocity (dP/dt) + 31 relational kinematic distances = 400
EXTENDED_FEAT_DIM = FEAT_DIM + (FEAT_DIM - 3) + 31 # 400

_WORKER_HOLISTIC = None


class CausalTorsoTracker:
    """
    Causal EMA tracker untuk torso center dan scale.
    - Menghitung center dan scale (jarak 2D XY antar-bahu) secara kausal dengan smoothing EMA (alpha=0.4).
    - Mempertahankan center & scale valid terakhir (coasting) jika pose hilang sementara.
    - Fallback ke center=(0.5*sx, 0.5*sy, 0.0) dan default_scale=0.25 jika pose belum pernah terdeteksi sama sekali.
    - Jarak 2D XY bahu digunakan untuk mencegah noise kedalaman (z) mengacaukan normalisasi skala.
    """

    def __init__(self, alpha: float = 0.4, default_scale: float = 0.25):
        self.alpha = float(alpha)
        self.default_scale = float(default_scale)
        self.center: Optional[np.ndarray] = None
        self.scale: Optional[float] = None
        self.has_valid_pose: bool = False

    def reset(self):
        self.center = None
        self.scale = None
        self.has_valid_pose = False

    def update(
        self,
        pose_landmarks_raw: Optional[np.ndarray],
        sx: float = 1.0,
        sy: float = 1.0,
    ) -> Tuple[np.ndarray, float, bool]:
        if pose_landmarks_raw is not None:
            ls = pose_landmarks_raw[11]
            rs = pose_landmarks_raw[12]
            inst_center = (ls + rs) / 2.0
            # 2D XY shoulder distance untuk stabilitas skala terhadap noise kedalaman z
            inst_scale = float(np.linalg.norm(ls[:2] - rs[:2]))
            if inst_scale < 1e-3:
                inst_scale = self.default_scale

            if not self.has_valid_pose or self.center is None or self.scale is None:
                self.center = inst_center.copy()
                self.scale = inst_scale
                self.has_valid_pose = True
            else:
                self.center = self.alpha * inst_center + (1.0 - self.alpha) * self.center
                self.scale = self.alpha * inst_scale + (1.0 - self.alpha) * self.scale
            return self.center.copy(), float(self.scale), True
        else:
            if self.has_valid_pose and self.center is not None and self.scale is not None:
                # Coasting pada state valid terakhir
                return self.center.copy(), float(self.scale), True
            else:
                fallback_center = np.array([0.5 * sx, 0.5 * sy, 0.0], dtype=np.float32)
                return fallback_center, self.default_scale, False


def init_worker():
    global _WORKER_HOLISTIC
    _WORKER_HOLISTIC = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def extract_holistic_frame(
    res,
    width: int,
    height: int,
    torso_tracker: Optional[CausalTorsoTracker] = None,
) -> np.ndarray:
    w = float(width)
    h = float(height)
    max_dim = max(w, h) if max(w, h) > 0.0 else 1.0
    sx = w / max_dim
    sy = h / max_dim

    # 1. Pose
    pose_present = res.pose_landmarks is not None
    if pose_present:
        p_raw = np.array(
            [[lm.x * sx, lm.y * sy, lm.z * sx] for lm in res.pose_landmarks.landmark],
            dtype=np.float32,
        )
    else:
        p_raw = None

    if torso_tracker is not None:
        torso_center, torso_scale, pose_valid_or_coasting = torso_tracker.update(p_raw, sx, sy)
    else:
        if pose_present and p_raw is not None:
            ls = p_raw[11]
            rs = p_raw[12]
            torso_center = (ls + rs) / 2.0
            torso_scale = float(np.linalg.norm(ls[:2] - rs[:2]))
            if torso_scale < 1e-3:
                torso_scale = 0.25
        else:
            torso_center = np.array([0.5 * sx, 0.5 * sy, 0.0], dtype=np.float32)
            torso_scale = 0.25

    if pose_present and p_raw is not None:
        pose_feat = (p_raw[SELECTED_POSE_INDICES] - torso_center) / torso_scale
    else:
        pose_feat = np.zeros((NUM_POSE_LANDMARKS, 3), dtype=np.float32)

    # 2. Left Hand
    lh_present = res.left_hand_landmarks is not None
    if lh_present:
        lh_raw = np.array(
            [[lm.x * sx, lm.y * sy, lm.z * sx] for lm in res.left_hand_landmarks.landmark],
            dtype=np.float32,
        )
        lh_feat = np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float32)
        lh_feat[:, :2] = (lh_raw[:, :2] - torso_center[:2]) / torso_scale
        # Penyelarasan kedalaman z: jika pose terdeteksi, selaraskan kedalaman relatif terhadap pergelangan tangan pose (landmark 15)
        if pose_present and p_raw is not None:
            pose_wrist_l_z = (p_raw[15, 2] - torso_center[2]) / torso_scale
            lh_feat[:, 2] = pose_wrist_l_z + (lh_raw[:, 2] / torso_scale)
        else:
            lh_feat[:, 2] = lh_raw[:, 2] / torso_scale
    else:
        lh_feat = np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float32)

    # 3. Right Hand
    rh_present = res.right_hand_landmarks is not None
    if rh_present:
        rh_raw = np.array(
            [[lm.x * sx, lm.y * sy, lm.z * sx] for lm in res.right_hand_landmarks.landmark],
            dtype=np.float32,
        )
        rh_feat = np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float32)
        rh_feat[:, :2] = (rh_raw[:, :2] - torso_center[:2]) / torso_scale
        # Penyelarasan kedalaman z: jika pose terdeteksi, selaraskan kedalaman relatif terhadap pergelangan tangan pose (landmark 16)
        if pose_present and p_raw is not None:
            pose_wrist_r_z = (p_raw[16, 2] - torso_center[2]) / torso_scale
            rh_feat[:, 2] = pose_wrist_r_z + (rh_raw[:, 2] / torso_scale)
        else:
            rh_feat[:, 2] = rh_raw[:, 2] / torso_scale
    else:
        rh_feat = np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float32)

    # 4. Presence Flags
    flags = np.array(
        [float(lh_present), float(rh_present), float(pose_present)],
        dtype=np.float32,
    )

    vec = np.concatenate([lh_feat.ravel(), rh_feat.ravel(), pose_feat.ravel(), flags])
    return vec.astype(np.float32)


def compute_extended_features(feats_186: np.ndarray) -> np.ndarray:
    """
    Mengekspansi fitur statis 186 dimensi menjadi 400 dimensi Spasio-Temporal:
      1. Fitur Statis Landmark Asli (186 dimensi: 61 coords x 3 + 3 flags)
      2. Instantaneous Velocity dP/dt (183 dimensi): dimasking agar tidak spike saat tangan baru muncul.
      3. Fitur Kinematik Spasial Kunci (31 dimensi): dimasking dengan flags kehadiran tangan/pose.
      Total = 186 + 183 + 31 = 400 dimensi.

    Mendukung input:
      - Single sequence : (seq_len, 186) -> (seq_len, 400)
      - Batch sequences : (batch, seq_len, 186) -> (batch, seq_len, 400)
    """
    is_batch = feats_186.ndim == 3
    if not is_batch:
        feats_186 = feats_186[None, ...]

    b_size, seq_len, _ = feats_186.shape
    coords = feats_186[:, :, :-3].reshape(b_size, seq_len, 61, 3)
    flags = feats_186[:, :, -3:]  # [has_lh, has_rh, has_pose]

    has_lh = flags[:, :, 0] > 0.5
    has_rh = flags[:, :, 1] > 0.5
    has_pose = flags[:, :, 2] > 0.5

    # 1. Instantaneous Velocity dengan Temporal Masking (Mencegah spike saat transisi kemunculan tangan)
    vel = np.zeros_like(coords)
    if seq_len > 1:
        lh_cont = has_lh[:, 1:] & has_lh[:, :-1]
        rh_cont = has_rh[:, 1:] & has_rh[:, :-1]
        pose_cont = has_pose[:, 1:] & has_pose[:, :-1]

        vel_mask = np.zeros((b_size, seq_len - 1, 61, 1), dtype=np.float32)
        vel_mask[:, :, :21, 0] = lh_cont[:, :, None]
        vel_mask[:, :, 21:42, 0] = rh_cont[:, :, None]
        vel_mask[:, :, 42:61, 0] = pose_cont[:, :, None]

        raw_diff = coords[:, 1:] - coords[:, :-1]
        vel[:, 1:] = raw_diff * vel_mask
        vel[:, 0] = vel[:, 1]
    vel_flat = vel.reshape(b_size, seq_len, -1)  # (b_size, seq_len, 183)

    # 2. Titik Landmark Referensi Tubuh & Wajah
    lh_wrist = coords[:, :, 0]    # Left Hand Wrist
    lh_thumb = coords[:, :, 4]    # Left Hand Thumb Tip
    lh_index = coords[:, :, 8]    # Left Hand Index Tip

    rh_wrist = coords[:, :, 21]   # Right Hand Wrist
    rh_thumb = coords[:, :, 25]   # Right Hand Thumb Tip
    rh_index = coords[:, :, 29]   # Right Hand Index Tip

    nose = coords[:, :, 42]
    left_ear = coords[:, :, 49]
    right_ear = coords[:, :, 50]
    mouth_center = (coords[:, :, 51] + coords[:, :, 52]) * 0.5
    ls = coords[:, :, 53]         # Left Shoulder
    rs = coords[:, :, 54]         # Right Shoulder
    torso_center = (ls + rs) * 0.5

    def _dist(p1, p2):
        return np.linalg.norm(p1 - p2, axis=-1, keepdims=True)

    # Mask kehadiran komponen untuk fitur relasional
    m_lh = (flags[:, :, 0:1] > 0.5).astype(np.float32)
    m_rh = (flags[:, :, 1:2] > 0.5).astype(np.float32)
    m_pose = (flags[:, :, 2:3] > 0.5).astype(np.float32)
    m_bi = m_lh * m_rh
    m_lh_pose = m_lh * m_pose
    m_rh_pose = m_rh * m_pose

    # A. Relasi Antar Tangan (2 dim)
    d_hands = _dist(lh_wrist, rh_wrist) * m_bi
    d_index_tips = _dist(lh_index, rh_index) * m_bi

    # B. Relasi Wajah & Hidung (2 dim)
    d_lh_nose = _dist(lh_wrist, nose) * m_lh_pose
    d_rh_nose = _dist(rh_wrist, nose) * m_rh_pose

    # C. Relasi Bibir / Mulut (Kunci Pembeda 'Makan' vs 'Cari') (6 dim)
    d_lh_mouth = _dist(lh_wrist, mouth_center) * m_lh_pose
    d_rh_mouth = _dist(rh_wrist, mouth_center) * m_rh_pose
    d_lh_index_mouth = _dist(lh_index, mouth_center) * m_lh_pose
    d_rh_index_mouth = _dist(rh_index, mouth_center) * m_rh_pose
    d_lh_thumb_mouth = _dist(lh_thumb, mouth_center) * m_lh_pose
    d_rh_thumb_mouth = _dist(rh_thumb, mouth_center) * m_rh_pose

    # D. Relasi Telinga (Kunci Pembeda 'Tuli' vs 'Cari') (4 dim)
    d_lh_index_ear = _dist(lh_index, left_ear) * m_lh_pose
    d_rh_index_ear = _dist(rh_index, right_ear) * m_rh_pose
    d_lh_wrist_ear = _dist(lh_wrist, left_ear) * m_lh_pose
    d_rh_wrist_ear = _dist(rh_wrist, right_ear) * m_rh_pose

    # E. Relasi Bahu & Torso / Dada (Kunci 'Saya', 'Kapan', 'Rumah') (4 dim)
    d_lh_ls = _dist(lh_wrist, ls) * m_lh_pose
    d_rh_rs = _dist(rh_wrist, rs) * m_rh_pose
    d_lh_index_torso = _dist(lh_index, torso_center) * m_lh_pose
    d_rh_index_torso = _dist(rh_index, torso_center) * m_rh_pose

    # F. Elevasi Vertikal (Tinggi Relatif Y) (3 dim)
    rel_y_lh_nose = (lh_index[:, :, 1:2] - nose[:, :, 1:2]) * m_lh_pose
    rel_y_rh_nose = (rh_index[:, :, 1:2] - nose[:, :, 1:2]) * m_rh_pose
    rel_y_lh_torso = (lh_wrist[:, :, 1:2] - torso_center[:, :, 1:2]) * m_lh_pose

    # G. Finger Curls (Bentuk Kepalan Tangan) (10 dim)
    lh_tips = [coords[:, :, idx] for idx in [4, 8, 12, 16, 20]]
    rh_tips = [coords[:, :, idx] for idx in [25, 29, 33, 37, 41]]
    lh_curls = [_dist(lh_wrist, tip) * m_lh for tip in lh_tips]
    rh_curls = [_dist(rh_wrist, tip) * m_rh for tip in rh_tips]

    rel_feats = np.concatenate(
        [
            d_hands, d_index_tips,
            d_lh_nose, d_rh_nose,
            d_lh_mouth, d_rh_mouth,
            d_lh_index_mouth, d_rh_index_mouth,
            d_lh_thumb_mouth, d_rh_thumb_mouth,
            d_lh_index_ear, d_rh_index_ear,
            d_lh_wrist_ear, d_rh_wrist_ear,
            d_lh_ls, d_rh_rs,
            d_lh_index_torso, d_rh_index_torso,
            rel_y_lh_nose, rel_y_rh_nose, rel_y_lh_torso,
        ]
        + lh_curls + rh_curls,
        axis=-1,
    )  # (b_size, seq_len, 31)

    out = np.concatenate([feats_186, vel_flat, rel_feats], axis=-1)
    if not is_batch:
        out = out[0]
    return out.astype(np.float32)


POSE_FLIP_PAIRS = [
    (43, 46),  # eye_inner_l <-> eye_inner_r
    (44, 47),  # eye_l <-> eye_r
    (45, 48),  # eye_outer_l <-> eye_outer_r
    (49, 50),  # ear_l <-> ear_r
    (51, 52),  # mouth_l <-> mouth_r
    (53, 54),  # shoulder_l <-> shoulder_r
    (55, 56),  # elbow_l <-> elbow_r
    (57, 58),  # wrist_l <-> wrist_r
    (59, 60),  # hip_l <-> hip_r
]


def flip_horizontal_landmarks(feats_186: np.ndarray) -> np.ndarray:
    """
    Membalik secara horizontal (flip mirror kiri <-> kanan) untuk fitur landmark 186 dimensi:
      1. Koordinat X dibalik (-x).
      2. Tangan Kiri (points 0..20) dan Tangan Kanan (points 21..41) ditukar posisinya.
      3. Flag deteksi tangan kiri (flag 0) dan tangan kanan (flag 1) ditukar.
      4. Titik-titik pose bilateral (mata, telinga, bibir, bahu, siku, pergelangan, pinggul) ditukar.
    Mendukung input:
      - (seq_len, 186)
      - (batch, seq_len, 186)
    """
    is_batch = feats_186.ndim == 3
    arr = feats_186.copy()
    if not is_batch:
        arr = arr[None, ...]

    b_size, seq_len, _ = arr.shape
    coords = arr[:, :, :-3].reshape(b_size, seq_len, 61, 3)
    flags = arr[:, :, -3:]

    # 1. Balik koordinat X
    coords[:, :, :, 0] = -coords[:, :, :, 0]

    # 2. Tukar Tangan Kiri <-> Tangan Kanan
    lh_coords = coords[:, :, :21, :].copy()
    rh_coords = coords[:, :, 21:42, :].copy()
    coords[:, :, :21, :] = rh_coords
    coords[:, :, 21:42, :] = lh_coords

    # Tukar Flag LH <-> RH
    lh_flag = flags[:, :, 0].copy()
    rh_flag = flags[:, :, 1].copy()
    flags[:, :, 0] = rh_flag
    flags[:, :, 1] = lh_flag

    # 3. Tukar titik bilateral Pose
    for l_idx, r_idx in POSE_FLIP_PAIRS:
        tmp = coords[:, :, l_idx, :].copy()
        coords[:, :, l_idx, :] = coords[:, :, r_idx, :]
        coords[:, :, r_idx, :] = tmp

    out = np.concatenate([coords.reshape(b_size, seq_len, -1), flags], axis=-1)
    if not is_batch:
        out = out[0]
    return out.astype(np.float32)


def extract_video_features(
    video_path: str,
    holistic_detector,
    target_fps: float = 12.0,
    max_frames: int = 120,
    max_side: int = 512,
) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    # Reset state internal Holistic untuk mencegah kebocoran tracking antar-video
    if hasattr(holistic_detector, "reset"):
        holistic_detector.reset()

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(src_fps) or src_fps <= 1.0:
        src_fps = 25.0

    torso_tracker = CausalTorsoTracker()
    time_step = 1.0 / float(target_fps)
    next_target_time = 0.0
    half_src_period = 0.5 / src_fps

    frames_feats = []
    frame_idx = 0

    while len(frames_feats) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_time = frame_idx / src_fps
        frame_idx += 1

        if (frame_time + half_src_period) < next_target_time:
            continue
        next_target_time += time_step

        h, w = frame.shape[:2]
        if max(h, w) > max_side:
            s = max_side / float(max(h, w))
            frame = cv2.resize(
                frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA
            )
            h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic_detector.process(rgb)
        feat = extract_holistic_frame(res, w, h, torso_tracker=torso_tracker)
        frames_feats.append(feat)

    cap.release()

    if not frames_feats:
        return None

    return np.stack(frames_feats).astype(np.float32)


def process_video_task(task_tuple) -> Dict:
    v_path, label_id, signer_id, sample_id, filename = task_tuple
    global _WORKER_HOLISTIC
    if _WORKER_HOLISTIC is None:
        init_worker()
    try:
        feats = extract_video_features(v_path, _WORKER_HOLISTIC)
        if feats is None or len(feats) == 0:
            return {"ok": False, "filename": filename, "error": "Gagal membaca frame"}
        return {
            "ok": True,
            "filename": filename,
            "feats": feats,
            "length": len(feats),
            "label": label_id,
            "signer": signer_id,
            "sample": sample_id,
        }
    except Exception as e:
        return {"ok": False, "filename": filename, "error": str(e)}
