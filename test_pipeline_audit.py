"""
Unit Tests untuk Validasi Audit Pipeline Hand Sign Recognition (Context AI Agent).

Menguji:
  1. EMA BatchNorm Buffers: ModelEMA.update() menyinkronkan parameters dan running buffers (running_mean, running_var).
  2. Augmentation Masking: Landmark yang tidak terdeteksi (flags=0) HARUS tetap nol setelah augmentasi.
  3. Involutori Flip: flip(flip(x)) == x dan menukar LH <-> RH serta bilateral pose.
  4. Velocity Masking: Transisi 0 <-> valid tidak boleh memicu spike velocity.
  5. Relational Feature Masking: Tangan absen tidak boleh menghasilkan jarak/finger curl palsu.
  6. Torso Tracker Causal: Mempertahankan center & scale saat pose hilang sementara (coasting).
  7. Motion Trimming Spike Guard: Kemunculan tangan tiba-tiba tidak memicu spike energi gerak palsu.
"""

import sys
import unittest
import numpy as np
import torch
import torch.nn as nn

# Import dari holistic_extractor
from holistic_extractor import (
    FEAT_DIM,
    EXTENDED_FEAT_DIM,
    EXTRACTOR_VERSION,
    CausalTorsoTracker,
    flip_horizontal_landmarks,
    compute_extended_features,
)

# Import dari train-new
import importlib
train_new = importlib.import_module("train-new")
ModelEMA = train_new.ModelEMA
augment_sequence_window = train_new.augment_sequence_window
find_motion_trimmed_bounds = train_new.find_motion_trimmed_bounds


class TestPipelineAudit(unittest.TestCase):

    def test_ema_batchnorm_buffers_sync(self):
        """Uji bahwa ModelEMA.update menyinkronkan parameters dan running buffers BatchNorm."""
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(186, 32, kernel_size=3, padding=1)
                self.bn = nn.BatchNorm1d(32)

            def forward(self, x):
                return self.bn(self.conv(x))

        model = DummyModel()
        x = torch.randn(4, 186, 32)
        model.train()
        _ = model(x)  # Memperbarui running_mean & running_var di BatchNorm

        ema = ModelEMA(model, decay=0.99)
        # Jalankan beberapa batch lagi pada model utama
        for _ in range(3):
            _ = model(torch.randn(4, 186, 32) * 5.0 + 2.0)
            ema.update(model)

        # Verifikasi bahwa buffer BatchNorm di shadow sama persis dengan model utama
        for s_b, m_b in zip(ema.shadow.buffers(), model.buffers()):
            diff = torch.max(torch.abs(s_b - m_b)).item()
            self.assertAlmostEqual(diff, 0.0, places=5, msg="Buffer BatchNorm di EMA shadow harus sinkron")

    def test_horizontal_flip_involutory(self):
        """Uji bahwa flip(flip(x)) menghasilkan input awal secara identik (involutori)."""
        rng = np.random.default_rng(42)
        sample = rng.normal(0.0, 1.0, size=(32, 186)).astype(np.float32)
        sample[:, -3:] = np.array([1.0, 1.0, 1.0])  # LH, RH, Pose aktif

        flipped_once = flip_horizontal_landmarks(sample)
        flipped_twice = flip_horizontal_landmarks(flipped_once)

        diff = np.max(np.abs(sample - flipped_twice))
        self.assertAlmostEqual(float(diff), 0.0, places=5, msg="flip(flip(x)) harus identik dengan x")

    def test_horizontal_flip_swaps_hands_and_flags(self):
        """Uji bahwa flip menukar koordinat dan flag LH <-> RH dengan benar."""
        seq = np.zeros((16, 186), dtype=np.float32)
        # LH aktif (points 0..20), RH absen
        seq[:, :63] = 1.5
        seq[:, 63:126] = 0.0
        seq[:, -3] = 1.0  # has_lh = 1
        seq[:, -2] = 0.0  # has_rh = 0
        seq[:, -1] = 1.0  # has_pose = 1

        flipped = flip_horizontal_landmarks(seq)

        # Setelah flip: LH harus nol & flag 0; RH harus terisi & flag 1
        self.assertEqual(float(flipped[0, -3]), 0.0, "Flag LH harus 0 setelah flip")
        self.assertEqual(float(flipped[0, -2]), 1.0, "Flag RH harus 1 setelah flip")
        self.assertAlmostEqual(float(np.max(np.abs(flipped[:, :63]))), 0.0, places=5, msg="LH harus bernilai nol setelah flip")
        self.assertGreater(float(np.mean(np.abs(flipped[:, 63:126]))), 0.0, msg="RH harus bernilai nonzero setelah flip")

    def test_velocity_masking_on_missing_landmarks(self):
        """Uji bahwa transisi landmark dari 0 ke valid tidak memicu spike velocity palsu."""
        seq = np.zeros((10, 186), dtype=np.float32)
        # Frame 0..4: LH absen (0.0)
        # Frame 5..9: LH muncul tiba-tiba di (0.5, 0.5, 0.5)
        seq[5:, :63] = 0.5
        seq[5:, -3] = 1.0  # has_lh = 1 hanya pada frame 5..9

        ext = compute_extended_features(seq)
        vel_lh = ext[:, 186 : 186 + 63]

        # Velocity pada frame 5 (transisi pertama dari 0 ke 0.5) HARUS nol karena frame 4 tidak valid!
        spike_at_transition = np.max(np.abs(vel_lh[5]))
        self.assertAlmostEqual(float(spike_at_transition), 0.0, places=5,
                               msg=f"Velocity pada frame transisi harus 0 (ditemukan spike: {spike_at_transition})")

    def test_relational_features_masking_when_hand_missing(self):
        """Uji bahwa relasi tangan dan finger curls dimasking ke nol ketika tangan tidak terdeteksi."""
        seq = np.zeros((10, 186), dtype=np.float32)
        # Kedua tangan absen (flags = 0), hanya pose aktif
        seq[:, -1] = 1.0
        seq[:, 126:183] = np.random.uniform(0.1, 0.8, size=(10, 57)).astype(np.float32)

        ext = compute_extended_features(seq)
        rel_feats = ext[:, 369:400]  # 31 fitur relasional

        # Relasi bimanual (d_hands, d_index_tips) dan finger curls HARUS 0.0 karena tangan absen
        d_hands = rel_feats[:, 0]
        d_index_tips = rel_feats[:, 1]
        lh_curls = rel_feats[:, 21:26]
        rh_curls = rel_feats[:, 26:31]

        self.assertAlmostEqual(float(np.max(np.abs(d_hands))), 0.0, places=5, msg="d_hands harus 0 saat tangan absen")
        self.assertAlmostEqual(float(np.max(np.abs(d_index_tips))), 0.0, places=5, msg="d_index_tips harus 0 saat tangan absen")
        self.assertAlmostEqual(float(np.max(np.abs(lh_curls))), 0.0, places=5, msg="lh_curls harus 0 saat tangan absen")
        self.assertAlmostEqual(float(np.max(np.abs(rh_curls))), 0.0, places=5, msg="rh_curls harus 0 saat tangan absen")

    def test_augmentation_masking_on_missing_landmarks(self):
        """Uji bahwa augmentasi (shift, noise, jitter) TIDAK mengubah koordinat landmark yang absen (flags=0) menjadi nonzero."""
        rng = np.random.default_rng(123)
        window = np.zeros((32, 186), dtype=np.float32)
        # Hanya RH dan Pose yang terdeteksi, LH absen sama sekali
        window[:, 63:126] = 0.5   # RH aktif
        window[:, 126:183] = 0.3  # Pose aktif
        window[:, -2] = 1.0       # has_rh = 1
        window[:, -1] = 1.0       # has_pose = 1
        window[:, -3] = 0.0       # has_lh = 0

        # Lakukan augmentasi berulang kali dengan flip dimatikan untuk fokus pada LH
        for _ in range(5):
            aug = augment_sequence_window(window, rng=rng, flip_prob=0.0, noise_std=0.05)
            # LH koordinat (0..62) HARUS bernilai 0.0 persis!
            lh_coords = aug[:, :63]
            max_lh = float(np.max(np.abs(lh_coords)))
            self.assertEqual(max_lh, 0.0, msg=f"Koordinat LH absen berubah menjadi nonzero ({max_lh}) setelah augmentasi!")

    def test_causal_torso_tracker(self):
        """Uji bahwa CausalTorsoTracker melakukan EMA smoothing, coasting saat pose hilang, dan fallback default."""
        tracker = CausalTorsoTracker(alpha=0.5, default_scale=0.25)
        # Frame 0: Pose terdeteksi di bahu ls=(0.4, 0.4, 0.0), rs=(0.6, 0.4, 0.0) -> center=(0.5, 0.4, 0.0), scale=0.2
        pose_raw = np.zeros((33, 3), dtype=np.float32)
        pose_raw[11] = [0.4, 0.4, 0.0]
        pose_raw[12] = [0.6, 0.4, 0.0]

        center, scale, valid = tracker.update(pose_raw)
        self.assertTrue(valid)
        self.assertAlmostEqual(float(center[0]), 0.5, places=4)
        self.assertAlmostEqual(float(scale), 0.2, places=4)

        # Frame 1: Pose hilang sementara (None) -> Coasting pada center & scale sebelumnya
        center_lost, scale_lost, valid_lost = tracker.update(None)
        self.assertTrue(valid_lost)
        self.assertAlmostEqual(float(center_lost[0]), 0.5, places=4)
        self.assertAlmostEqual(float(scale_lost), 0.2, places=4)

        # Reset tracker -> Pose None -> Fallback default
        tracker.reset()
        center_def, scale_def, valid_def = tracker.update(None, sx=1.0, sy=1.0)
        self.assertFalse(valid_def)
        self.assertAlmostEqual(float(scale_def), 0.25, places=4)

    def test_motion_trimming_no_spike_on_hand_appearance(self):
        """Uji bahwa transisi kemunculan tangan tidak memicu batas trimming palsu."""
        feats = np.zeros((30, 186), dtype=np.float32)
        # Frame 0..9: tangan absen
        # Frame 10..29: tangan muncul dan melakukan gerakan
        feats[10:, 63:126] = 0.5
        feats[10:, -2] = 1.0  # has_rh = 1
        # Berikan gerakan nyata pada frame 15..25
        for f in range(15, 26):
            feats[f, 63:126] = 0.5 + (f - 15) * 0.02

        s_bound, e_bound = find_motion_trimmed_bounds(feats, min_frames=16)
        # Batas awal harus berada di sekitar frame aktif (>=10), bukan di frame 0
        self.assertGreaterEqual(s_bound, 8)
        self.assertLessEqual(e_bound, 29)


if __name__ == "__main__":
    unittest.main()
