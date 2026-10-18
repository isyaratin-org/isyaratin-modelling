# Laporan Hasil Evaluasi Model Isyarat BISINDO

> **Tanggal Evaluasi**: 2026-10-18 09:12:40
> **Model**: `models/gabungan_lstm_best.pt` (Arsitektur: MGN, Input Dim: 400)
> **Data Evaluasi**: `Stratified Holdout Split 20% (377 video)` (Total Sampel Video: 377, Kelas: 55)

## 1. Ringkasan Metrik Utama (Executive Summary)

| Metrik | Nilai | Keterangan |
| :--- | :---: | :--- |
| **Video Consensus Accuracy (Top-1)** | **93.37%** | 1 Video = 1 Keputusan Konsensus |
| **Top-3 Accuracy** | **97.08%** | Label benar masuk 3 probabilitas tertinggi |
| **Top-5 Accuracy** | **97.61%** | Label benar masuk 5 probabilitas tertinggi |
| **Macro F1-Score** | 0.9395 | Rata-rata F1 seluruh kelas tanpa bobot |
| **Weighted F1-Score** | 0.9328 | Rata-rata F1 berbobot proporsi sampel |
| **ROC Macro AUC** | 0.9974 | Kemampuan separasi kurva ROC makro |
| **ROC Micro AUC** | 0.9955 | Kemampuan separasi kurva ROC mikro |
| **PR Micro AP** | 0.9418 | Average Precision pada kurva Precision-Recall |

## 2. Visualisasi Evaluasi (Charts & Curves)

### Normalized Confusion Matrix (%)
![Normalized Confusion Matrix (%)](confusion_matrix_normalized.png)

### Raw Count Confusion Matrix
![Raw Count Confusion Matrix](confusion_matrix_raw.png)

### Multi-Class ROC Curves (One-vs-Rest)
![Multi-Class ROC Curves (One-vs-Rest)](roc_curves.png)

### Precision-Recall (PR) Curves
![Precision-Recall (PR) Curves](precision_recall_curves.png)

### Performa F1-Score per Kelas Isyarat
![Performa F1-Score per Kelas Isyarat](per_class_f1_bar.png)

### Top Confused Pairs (Pasangan Isyarat Tertukar)
![Top Confused Pairs (Pasangan Isyarat Tertukar)](top_confused_pairs.png)

## 3. Analisis & Insight

- **Akurasi Konsensus vs Window**: Menggunakan pooling temporal konsensus seluruh frame video menghasilkan akurasi yang lebih tinggi dan stabil dibandingkan keputusan per frame window terisolasi.
- **Karakteristik Confusion Matrix**: Gerakan isyarat yang memiliki titik orientasi serupa (misal isyarat berbasis waktu seperti Pagi/Siang/Sore/Malam, atau warna Merah/Kuning/Hijau) memiliki tingkat kebingungan lebih tinggi jika hanya mengandalkan koordinat posisi. Fitur *Instantaneous Velocity* $\Delta P_t$ dan *Relational Distance* membantu membedakan arah gerakan tersebut.
- **Distribusi Signer**: Evaluasi pada holdout Signer 3 merefleksikan kemampuan generalisasi model pada pengguna baru (*unseen signer*).
