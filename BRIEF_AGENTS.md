# BRIEF_AGENTS.md — Isyaratin (BISINDO Sign Language Recognition)

Ringkasan cepat untuk AI agents yang bekerja di repo ini. Baca ini dulu sebelum mengubah kode.

## 1. Apa Ini?

**Isyaratin** adalah sistem pengenalan bahasa isyarat **BISINDO** (Bahasa Isyarat Indonesia) berbasis visi komputer. Sistem menangkap video tangan/tubuh via MediaPipe Holistic, mengekstrak fitur spasio-temporal, lalu mengklasifikasikan gestur menjadi kata BISINDO menggunakan model BiLSTM/MGN.

**Status saat ini**: Dataset manual **20 kelas / 300 video**, akurasi konsensus video **99.00%** (dengan TTA, hanya 3 error), Macro F1 **0.99**.

> **Catatan eksperimen (2026-10-17)**:
>
> - Fitur extended 400D (`--feat-dim 400`) menghasilkan **93%** — lebih rendah dari 186D. Keputusan: tetap 186D.
> - **Test-Time Augmentation (TTA)** (`--tta` di `evaluate.py`): 95.33% → 96.00%.
> - **Model lebih besar** (hidden1=192, hidden2=128, 1.2M param) + TTA: **99.00%** (3 error). Ini model produksi saat ini.
> - Ensemble 3-seed + TTA: 96.67% (tidak mengalahkan model besar tunggal).

## 2. Arsitektur Pipeline

```mermaid
flowchart LR
    A[Video / Webcam] --> B[holistic_extractor.py]
    B --> C[Fitur 186D / 400D]
    C --> D[train-new.py]
    D --> E[Model .pt / .onnx]
    E --> F[camera.py Live Inference]
    F --> G[kinematic_engine.py]
    G --> H[Hasil Kata BISINDO]
```

- **Ekstraksi fitur**: [`holistic_extractor.py`](holistic_extractor.py) — MediaPipe Holistic → 186D statis (2 tangan + pose) atau 400D extended (186 statis + 183 velocity + 31 kinematik).
- **Training**: [`train-new.py`](train-new.py) — BiLSTM 2-layer ramping (`--arch lstm`) atau MGN (`--arch mgn`). Default manual = LSTM 186D, from scratch (tanpa pretrained).
- **Inferensi live**: [`camera.py`](camera.py) — webcam/video, hands-free trigger, sentence builder, stabilizer debounce.
- **Disambiguasi kinematik**: [`kinematic_engine.py`](kinematic_engine.py) — aturan deklaratif dari [`bisindo_rules.json`](bisindo_rules.json) untuk memisahkan pasangan gestur ambigu (Rumah vs A, Makan vs Terima Kasih, dll).
- **Evaluasi**: [`evaluate.py`](evaluate.py) — confusion matrix, ROC, PR, laporan ke `reports/evaluation/`.

## 3. Struktur Direktori

| Path                                          | Isi                                                                                                                |
| :-------------------------------------------- | :----------------------------------------------------------------------------------------------------------------- |
| [`datasets/manual/`](datasets/manual)         | Dataset manual: subfolder per kelas, tiap kelas 15 video `.mp4`                                                    |
| [`datasets/WL-BISINDO/`](datasets/WL-BISINDO) | Dataset WL-BISINDO (28 kelas awal)                                                                                 |
| [`labels/`](labels)                           | Definisi kelas & metadata: `manual_labels.json`, `manual_labels_meta.json`, `wl_labels.json`, `bisindo_rules.json` |
| [`models/`](models)                           | Checkpoint `.pt` & ekspor `.onnx`                                                                                  |
| [`data/`](data)                               | Cache fitur biner `.npz` hasil ekstraksi                                                                           |
| [`reports/evaluation/`](reports/evaluation)   | Laporan evaluasi & grafik                                                                                          |
| [`feature.md`](feature.md)                    | **Referensi teknis lengkap** 400D feature mapping & normalisasi                                                    |
| [`pembelajaran.md`](pembelajaran.md)          | Catatan eksperimen & pembelajaran (baca untuk konteks keputusan)                                                   |

## 4. Perintah Utama (via `uv`)

```bash
# Training dataset manual (LSTM, from scratch, re-extract cache)
uv run python train-new.py --dataset manual --arch lstm --re-extract

# Training model produksi gabungan 55 kelas (Manual 20 + WL 9 + Abjad 26)
uv run python build_dataset_gabungan.py
uv run python train-new.py --dataset gabungan --arch lstm --epochs 45

# Training model produksi manual (LSTM besar 1.2M param, hidden1=192 hidden2=128)
uv run python train-new.py --dataset manual --arch lstm --hidden1 192 --hidden2 128 --epochs 40

# Ekspor TFLite & sinkronisasi aset Flutter
uv run python export_tflite.py

# Evaluasi model gabungan 55 kelas
uv run python evaluate.py --dataset gabungan

# Evaluasi model manual (semua signer) — gunakan --tta untuk akurasi terbaik (99%)
uv run python evaluate.py --dataset manual --model models/manual_lstm_best.pt --all-signers --tta

# Live webcam (Default: Model 55 Kelas Gabungan)
uv run python camera.py

# Live webcam (Manual model 20 kelas)
uv run python camera.py --dataset manual

# Uji dari file video
uv run python camera.py --video path/to/video.mp4

# Ekstrak label dataset manual (setelah menambah video baru)
uv run python datasets/manual/extract_label.py

# Visualisasi dataset
uv run python dataset_visualizer.py --all
```

**Catatan penting**:

- Gunakan `uv run` (bukan `pip`). Python 3.11, PyTorch CUDA 12.8 dari index khusus di [`pyproject.toml`](pyproject.toml).
- `--re-extract` wajib setelah menambah/mengubah video dataset agar cache `.npz` diperbarui.
- Setelah menambah folder kelas baru di `datasets/manual/`, jalankan `extract_label.py` dulu agar label tersinkronisasi.

## 5. Konvensi & Aturan Penting

1. **Zero torch top-level import** di [`train-new.py`](train-new.py) — jangan `import torch` di level atas (menyebabkan MemoryError pada multiprocessing spawn Windows). Gunakan lazy import / factory.
2. **Normalisasi isotropik** — koordinat dinormalisasi dengan `scale = max(W, H)` (bukan per-axis), torso-centric, body-scale. Jangan ubah tanpa membaca [`feature.md`](feature.md) §3.
3. **Aturan disambiguasi deklaratif** — tambah kosakata baru = tambah rule di `labels/bisindo_rules.json` (versi lengkap WL + manual), bukan hardcode di Python. [`kinematic_engine.py`](kinematic_engine.py) memprioritaskan `labels/` di atas root `bisindo_rules.json`.
4. **Motion-trimmed cropping** — training manual memangkas frame pasif sebelum resampling 32 frame (Strategi 2).
5. **Hands-free inference** — entry saat tangan muncul, exit grace period 1.5s.
6. **Bahasa kode**: komentar/docstring dalam Bahasa Indonesia (konsisten dengan codebase).
7. **Model manual default**: `models/manual_lstm_best.pt` (LSTM besar, 186D, hidden1=192 hidden2=128, 20 kelas, 1.2M param). Jangan pakai pretrained WL untuk manual kecuali `--finetune`.

## 6. Kosakata Saat Ini (20 Kelas Manual)

`100 ribu rupiah`, `baik`, `bantuannya`, `bapak`, `berapa`, `buka`, `buku rekening`, `disini`, `formulir`, `isi`, `ktp`, `mau`, `membawa`, `minimal`, `sama-sama`, `selamat siang`, `setoran awal`, `silakan`, `tanda tangan`, `terima kasih`

## 7. Area Perbaikan Diketahui (Confusion)

Berdasarkan evaluasi terakhir (14 error / 300 video, model 186D):

- `baik` → `sama-sama` (5 error) — gestur sopan/akui mirip.
- `selamat siang` → `100 ribu rupiah` (5 error) — elevasi tangan mirip.
- `disini` → `sama-sama` (1), `tanda tangan` → `isi` (1), `silakan` → `mau` (1), `berapa` → `mau` (1).

**Aturan kinematik baru ditambahkan** di [`labels/bisindo_rules.json`](labels/bisindo_rules.json) untuk pasangan ini (diterapkan saat inferensi live di `camera.py`, bukan di evaluasi raw):

- `selamat_siang_vs_100ribu` (y_travel), `tanda_tangan_vs_mau` (chest_motion_std), `terima_kasih_vs_isi` (chest_motion_std), `baik_vs_sama_sama` (bimanual_open_spread).

Divalidasi 100% lulus di [`test_kinematic_cases.py`](test_kinematic_cases.py) (14 skenario). Lihat [`reports/evaluation/manual/evaluation_report.md`](reports/evaluation/manual/evaluation_report.md) untuk detail lengkap.
