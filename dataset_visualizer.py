"""
dataset_visualizer.py - Alat Profiling & Visualisasi Multi-Dataset BISINDO / Isyarat Tangan.

Mendukung:
  1. Deteksi otomatis file 'labeled.json' (atau 'labels.json') di dalam setiap folder dataset.
  2. Pemindaian video (.mp4, .avi, .mov, .webm) atau gambar / folder kelas.
  3. Ekstraksi statistik:
     - Distribusi kelas (jumlah sampel per kata isyarat/gloss).
     - Distribusi signer / peraga isyarat (jika ada pola signer).
     - Distribusi durasi video, FPS, frame count, resolusi video.
     - Rasio keseimbangan dataset (class imbalance score).
  4. Pembuatan visualisasi kaya (PNG) di direktori reports/datasets/<nama_dataset>/:
     - class_distribution.png
     - duration_fps_distribution.png
     - signer_heatmap.png
     - dataset_summary_card.png
  5. Perbandingan multi-dataset jika terdapat lebih dari satu dataset di folder datasets/.

Cara Pakai:
  uv run python dataset_visualizer.py --dataset WL-BISINDO
  uv run python dataset_visualizer.py --all
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# File ekstensi yang didukung
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Pola nama file standar signer{S}_label{L}_sample{N}
SIGNER_LABEL_PATTERN = re.compile(
    r"signer(?P<signer>\d+)[_-]label(?P<label>\d+)[_-]sample(?P<sample>\d+)",
    re.IGNORECASE,
)


def load_dataset_labels(dataset_dir: Path) -> Tuple[Dict[int, Dict[str, Any]], List[str]]:
    """
    Mencari dan membaca file label di folder dataset.
    Prioritas pencarian: labeled.json -> labels.json -> classes.json -> wl_labels.json
    Mendukung format list of dicts, list of strings, atau dict id->name.
    """
    candidates = [
        dataset_dir / "labeled.json",
        dataset_dir / "labels.json",
        dataset_dir / "classes.json",
        dataset_dir / "wl_labels.json",
        Path("wl_labels_meta.json"),
        Path("wl_labels.json"),
    ]

    label_file = None
    for c in candidates:
        if c.exists():
            label_file = c
            break

    label_map: Dict[int, Dict[str, Any]] = {}
    class_names: List[str] = []

    if label_file and label_file.exists():
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for idx, item in enumerate(data):
                    if isinstance(item, dict):
                        l_id = int(item.get("id", idx))
                        gloss = item.get("gloss", item.get("name", f"Class_{l_id}"))
                        english = item.get("english", item.get("translation", ""))
                        label_map[l_id] = {
                            "id": l_id,
                            "gloss": gloss,
                            "english": english,
                        }
                    elif isinstance(item, str):
                        label_map[idx] = {
                            "id": idx,
                            "gloss": item,
                            "english": "",
                        }
            elif isinstance(data, dict):
                # jika format {"labels": [...]}
                if "labels" in data and isinstance(data["labels"], list):
                    for idx, item in enumerate(data["labels"]):
                        if isinstance(item, dict):
                            l_id = int(item.get("id", idx))
                            label_map[l_id] = {
                                "id": l_id,
                                "gloss": item.get("gloss", f"Class_{l_id}"),
                                "english": item.get("english", ""),
                            }
                        else:
                            label_map[idx] = {"id": idx, "gloss": str(item), "english": ""}
                else:
                    for k, v in data.items():
                        try:
                            l_id = int(k)
                        except ValueError:
                            l_id = len(label_map)
                        if isinstance(v, dict):
                            label_map[l_id] = {
                                "id": l_id,
                                "gloss": v.get("gloss", v.get("name", str(k))),
                                "english": v.get("english", ""),
                            }
                        else:
                            label_map[l_id] = {
                                "id": l_id,
                                "gloss": str(v),
                                "english": "",
                            }

            max_id = max(label_map.keys()) if label_map else -1
            class_names = [label_map.get(i, {}).get("gloss", f"Class_{i}") for i in range(max_id + 1)]
            print(f"[{dataset_dir.name}] Membaca {len(label_map)} label dari: {label_file.name}")
        except Exception as e:
            print(f"[{dataset_dir.name}] Gagal membaca {label_file}: {e}")

    return label_map, class_names


def extract_video_metadata(video_path: Path) -> Dict[str, Any]:
    """Ekstraksi metadata dasar dari file video menggunakan OpenCV."""
    meta = {
        "duration_sec": 0.0,
        "frame_count": 0,
        "fps": 0.0,
        "width": 0,
        "height": 0,
        "filesize_mb": round(video_path.stat().st_size / (1024 * 1024), 2),
    }
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        meta["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        meta["fps"] = round(fps, 2) if fps > 0 else 0.0
        meta["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        meta["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if meta["fps"] > 0 and meta["frame_count"] > 0:
            meta["duration_sec"] = round(meta["frame_count"] / meta["fps"], 2)
        cap.release()
    return meta


def scan_dataset(dataset_dir: Path, sample_metadata_limit: int = 500) -> Dict[str, Any]:
    """
    Memindai seluruh isi folder dataset:
      - Mencari file video dan struktur folder.
      - Memetakan label, signer, dan durasi.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {dataset_dir}")

    label_map, class_names = load_dataset_labels(dataset_dir)

    records = []
    subdirs = [d for d in dataset_dir.iterdir() if d.is_dir()]

    # 1. Cek struktur flat (seperti datasets/WL-BISINDO)
    all_files = list(dataset_dir.glob("*.*"))
    video_files = [f for f in all_files if f.suffix.lower() in VIDEO_EXTS]
    image_files = [f for f in all_files if f.suffix.lower() in IMAGE_EXTS]

    # 2. Cek subdirektori jika file di root sedikit/tidak ada
    if len(video_files) == 0 and len(image_files) == 0 and subdirs:
        for sdir in subdirs:
            for vf in sdir.glob("*.*"):
                if vf.suffix.lower() in VIDEO_EXTS:
                    video_files.append(vf)
                elif vf.suffix.lower() in IMAGE_EXTS:
                    image_files.append(vf)

    print(f"[{dataset_dir.name}] Terdeteksi {len(video_files)} video, {len(image_files)} gambar.")

    # Parsing setiap file
    for idx, f in enumerate(video_files):
        signer_id = None
        label_id = None
        sample_id = None

        # Coba regex standar
        m = SIGNER_LABEL_PATTERN.search(f.name)
        if m:
            signer_id = int(m.group("signer"))
            label_id = int(m.group("label"))
            sample_id = int(m.group("sample"))
        else:
            # Cek apakah nama parent folder adalah nama label atau id
            parent_name = f.parent.name
            if parent_name.isdigit():
                label_id = int(parent_name)
            else:
                # cari apakah cocok dengan class_names
                for c_id, c_data in label_map.items():
                    if c_data.get("gloss", "").lower() == parent_name.lower():
                        label_id = c_id
                        break

        # Tentukan gloss & english
        if label_id is not None and label_id in label_map:
            gloss = label_map[label_id]["gloss"]
            english = label_map[label_id].get("english", "")
        elif label_id is not None:
            gloss = f"Label_{label_id}"
            english = ""
        else:
            gloss = "Unknown"
            english = ""

        row = {
            "filename": f.name,
            "filepath": str(f),
            "signer_id": signer_id,
            "label_id": label_id,
            "gloss": gloss,
            "english": english,
            "type": "video",
        }

        # Sampling ekstraksi metadata OpenCV untuk efisiensi
        if idx < sample_metadata_limit:
            v_meta = extract_video_metadata(f)
            row.update(v_meta)
        else:
            row.update({
                "duration_sec": None,
                "frame_count": None,
                "fps": None,
                "width": None,
                "height": None,
                "filesize_mb": round(f.stat().st_size / (1024 * 1024), 2),
            })

        records.append(row)

    df = pd.DataFrame(records)
    return {
        "dataset_name": dataset_dir.name,
        "dataset_path": str(dataset_dir),
        "total_samples": len(df),
        "label_map": label_map,
        "class_names": class_names,
        "dataframe": df,
    }


def generate_visualizations(scan_result: Dict[str, Any], output_dir: Path) -> List[Path]:
    """Membuat visualisasi statistik grafik dan dashboard untuk dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = scan_result["dataframe"]
    dataset_name = scan_result["dataset_name"]
    label_map = scan_result["label_map"]
    saved_plots = []

    if df.empty:
        print(f"[{dataset_name}] Dataframe kosong, tidak ada plot yang dibuat.")
        return saved_plots

    # Konfigurasi gaya visualisasi modern
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#CCCCCC"
    plt.rcParams["axes.linewidth"] = 0.8
    sns.set_theme(style="whitegrid")

    # =========================================================================
    # 1. Class Distribution Plot
    # =========================================================================
    fig_w = 12 if len(label_map) <= 15 else 16
    fig_h = max(7, len(label_map) * 0.28)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

    class_counts = df["gloss"].value_counts().reset_index()
    class_counts.columns = ["gloss", "count"]

    # Urutkan berdasarkan id label jika ada
    if label_map:
        gloss_to_id = {v["gloss"]: k for k, v in label_map.items()}
        class_counts["id"] = class_counts["gloss"].map(lambda g: gloss_to_id.get(g, 999))
        class_counts = class_counts.sort_values(by="id")

    mean_count = class_counts["count"].mean()

    # Warna harmonis
    palette = sns.color_palette("viridis", len(class_counts))
    bars = sns.barplot(
        data=class_counts,
        x="count",
        y="gloss",
        ax=ax,
        palette=palette,
        hue="gloss",
        legend=False,
    )

    ax.axvline(mean_count, color="#E74C3C", linestyle="--", linewidth=1.5, label=f"Rata-rata: {mean_count:.1f} sampel")
    ax.set_title(
        f"Distribusi Sampel per Kelas Kata Isyarat - {dataset_name}\n(Total Sampel: {len(df):,}, Total Kelas: {len(class_counts)})",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Jumlah Video / Sampel", fontsize=11, labelpad=10)
    ax.set_ylabel("Gloss / Kata Isyarat", fontsize=11)
    ax.legend(loc="lower right", frameon=True)

    # Tambahkan label angka di ujung bar
    for p in bars.patches:
        val = int(p.get_width())
        ax.annotate(
            f"{val}",
            (p.get_width() + 0.3, p.get_y() + p.get_height() / 2),
            ha="left",
            va="center",
            fontsize=8,
            color="#2C3E50",
            fontweight="semibold",
        )

    plt.tight_layout()
    p1 = output_dir / "class_distribution.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(p1)
    print(f"[{dataset_name}] Disimpan: {p1.name}")

    # =========================================================================
    # 2. Signer Heatmap / Distribution (jika ada data signer)
    # =========================================================================
    if df["signer_id"].notna().sum() > 0:
        fig, ax = plt.subplots(figsize=(14, max(8, len(class_counts) * 0.28)), dpi=150)
        cross = pd.crosstab(df["gloss"], df["signer_id"])
        
        # urutkan sesuai id jika ada
        if label_map:
            gloss_order = [label_map[i]["gloss"] for i in sorted(label_map.keys()) if label_map[i]["gloss"] in cross.index]
            cross = cross.reindex(gloss_order)

        sns.heatmap(
            cross,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            cbar_kws={"label": "Jumlah Sampel"},
            ax=ax,
            linewidths=0.5,
            linecolor="#F0F0F0",
        )
        ax.set_title(
            f"Matriks Distribusi Sampel Signer vs Kata Isyarat - {dataset_name}",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel("ID Signer (Peraga Isyarat)", fontsize=11, labelpad=10)
        ax.set_ylabel("Gloss / Kata Isyarat", fontsize=11)

        plt.tight_layout()
        p2 = output_dir / "signer_heatmap.png"
        fig.savefig(p2, bbox_inches="tight")
        plt.close(fig)
        saved_plots.append(p2)
        print(f"[{dataset_name}] Disimpan: {p2.name}")

    # =========================================================================
    # 3. Video Duration & Frame Count Distribution
    # =========================================================================
    valid_durations = df[df["duration_sec"].notna()]["duration_sec"]
    valid_frames = df[df["frame_count"].notna()]["frame_count"]

    if len(valid_durations) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        # Plot Durasi Detik
        sns.histplot(valid_durations, kde=True, color="#2980B9", ax=ax1, bins=20)
        ax1.axvline(valid_durations.mean(), color="#C0392B", linestyle="--", label=f"Mean: {valid_durations.mean():.2f}s")
        ax1.axvline(valid_durations.median(), color="#27AE60", linestyle=":", label=f"Median: {valid_durations.median():.2f}s")
        ax1.set_title(f"Distribusi Durasi Video (Detik)\n[Min: {valid_durations.min():.2f}s, Max: {valid_durations.max():.2f}s]", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Durasi (detik)")
        ax1.set_ylabel("Frekuensi")
        ax1.legend()

        # Plot Jumlah Frame
        sns.histplot(valid_frames, kde=True, color="#8E44AD", ax=ax2, bins=20)
        ax2.axvline(valid_frames.mean(), color="#C0392B", linestyle="--", label=f"Mean: {valid_frames.mean():.1f} frame")
        ax2.set_title(f"Distribusi Jumlah Frame per Video\n[Min: {int(valid_frames.min())}, Max: {int(valid_frames.max())}]", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Jumlah Frame")
        ax2.set_ylabel("Frekuensi")
        ax2.legend()

        plt.tight_layout()
        p3 = output_dir / "duration_fps_distribution.png"
        fig.savefig(p3, bbox_inches="tight")
        plt.close(fig)
        saved_plots.append(p3)
        print(f"[{dataset_name}] Disimpan: {p3.name}")

    # =========================================================================
    # 4. Summary Card (Infographic Overview)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.axis("off")

    num_signers = df["signer_id"].nunique() if df["signer_id"].notna().any() else 0
    total_classes = df["gloss"].nunique()
    total_samples = len(df)
    total_size_mb = df["filesize_mb"].sum()
    avg_dur = valid_durations.mean() if len(valid_durations) > 0 else 0.0
    total_dur_min = (valid_durations.sum() / 60.0) if len(valid_durations) > 0 else 0.0

    min_samples = class_counts["count"].min() if not class_counts.empty else 0
    max_samples = class_counts["count"].max() if not class_counts.empty else 0
    imbalance_ratio = round(max_samples / max(1, min_samples), 2)

    # Box styling
    rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=True, facecolor="#F8F9FA", edgecolor="#BDC3C7", linewidth=2, transform=ax.transAxes, zorder=0)
    ax.add_patch(rect)

    ax.text(0.5, 0.88, f"DATASET PROFILE: {dataset_name.upper()}", fontsize=16, fontweight="bold", ha="center", color="#2C3E50", transform=ax.transAxes)
    ax.text(0.5, 0.82, f"Analisis Komprehensif Karakteristik & Keseimbangan Data Isyarat", fontsize=10, ha="center", color="#7F8C8D", transform=ax.transAxes)

    metrics = [
        ("Total Sampel Video", f"{total_samples:,}"),
        ("Jumlah Kelas (Glosses)", f"{total_classes}"),
        ("Jumlah Signer (Peraga)", f"{num_signers if num_signers > 0 else 'N/A'}"),
        ("Ukuran Total File", f"{total_size_mb:.1f} MB (~{total_size_mb/1024:.2f} GB)"),
        ("Rata-rata Durasi Video", f"{avg_dur:.2f} detik"),
        ("Estimasi Total Durasi", f"{total_dur_min:.1f} menit ({total_dur_min/60:.2f} jam)"),
        ("Sampel Min / Max per Kelas", f"{min_samples} / {max_samples}"),
        ("Imbalance Ratio (Max/Min)", f"{imbalance_ratio}x ({'Sangat Seimbang' if imbalance_ratio == 1.0 else 'Cukup Seimbang' if imbalance_ratio <= 1.5 else 'Timpang'})"),
    ]

    start_y = 0.70
    line_spacing = 0.07
    for i, (label, val) in enumerate(metrics):
        curr_y = start_y - (i * line_spacing)
        ax.text(0.10, curr_y, label, fontsize=11, fontweight="medium", color="#34495E", transform=ax.transAxes)
        ax.text(0.65, curr_y, val, fontsize=11, fontweight="bold", color="#16A085" if "Seimbang" in val else "#2980B9", transform=ax.transAxes)
        # divider line
        if i < len(metrics) - 1:
            ax.plot([0.10, 0.90], [curr_y - 0.02, curr_y - 0.02], color="#E2E8F0", linewidth=1, transform=ax.transAxes)

    p4 = output_dir / "dataset_summary_card.png"
    fig.savefig(p4, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(p4)
    print(f"[{dataset_name}] Disimpan: {p4.name}")

    # =========================================================================
    # 5. Simpan Ringkasan JSON
    # =========================================================================
    summary_data = {
        "dataset_name": dataset_name,
        "total_samples": int(total_samples),
        "total_classes": int(total_classes),
        "total_signers": int(num_signers),
        "total_size_mb": float(round(total_size_mb, 2)),
        "avg_duration_sec": float(round(avg_dur, 2)),
        "imbalance_ratio": float(imbalance_ratio),
        "class_counts": class_counts.to_dict(orient="records"),
    }
    json_path = output_dir / "dataset_profile.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"[{dataset_name}] Ringkasan JSON disimpan: {json_path.name}")

    return saved_plots


def compare_datasets(scan_results: List[Dict[str, Any]], output_dir: Path):
    """Membuat grafik komparasi berdampingan jika ada beberapa dataset."""
    if len(scan_results) < 2:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_list = []
    for sr in scan_results:
        df = sr["dataframe"]
        if df.empty or "gloss" not in df.columns:
            continue
        summary_list.append({
            "Dataset": sr["dataset_name"],
            "Samples": len(df),
            "Classes": df["gloss"].nunique(),
            "Size_MB": round(df["filesize_mb"].sum(), 1),
        })

    if len(summary_list) < 2:
        return

    comp_df = pd.DataFrame(summary_list)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    sns.barplot(data=comp_df, x="Dataset", y="Samples", ax=ax1, palette="mako")
    ax1.set_title("Perbandingan Total Sampel Video", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Jumlah Sampel")
    for p in ax1.patches:
        ax1.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height() + 5), ha="center", fontsize=9, fontweight="bold")

    sns.barplot(data=comp_df, x="Dataset", y="Classes", ax=ax2, palette="crest")
    ax2.set_title("Perbandingan Jumlah Kelas Kata Isyarat", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Jumlah Kelas")
    for p in ax2.patches:
        ax2.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height() + 0.5), ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    p_comp = output_dir / "multidataset_comparison.png"
    fig.savefig(p_comp, bbox_inches="tight")
    plt.close(fig)
    print(f"[KOMPARASI MULTI-DATASET] Disimpan: {p_comp}")


def main():
    parser = argparse.ArgumentParser(description="Visualizer Profil Multi-Dataset BISINDO")
    parser.add_argument("--dataset", type=str, default="WL-BISINDO", help="Nama folder dataset di dalam 'datasets/' (default: WL-BISINDO)")
    parser.add_argument("--all", action="store_true", help="Pindai semua folder di dalam 'datasets/' dan buat komparasi")
    parser.add_argument("--reports-dir", type=str, default="reports/datasets", help="Folder tujuan penyimpanan laporan visual")
    args = parser.parse_args()

    base_dataset_dir = Path("datasets")
    reports_base = Path(args.reports_dir)

    if args.all:
        subdirs = [d for d in base_dataset_dir.iterdir() if d.is_dir() and d.name != "raw"]
        print(f"Memindai seluruh ({len(subdirs)}) folder dataset di {base_dataset_dir} ...")
        all_results = []
        for sdir in subdirs:
            print("\n" + "=" * 50)
            res = scan_dataset(sdir)
            out_d = reports_base / sdir.name
            generate_visualizations(res, out_d)
            all_results.append(res)

        if len(all_results) >= 2:
            compare_datasets(all_results, reports_base)
    else:
        target_dir = base_dataset_dir / args.dataset
        if not target_dir.exists():
            # Cek jika path absolut atau relatif langsung
            target_dir = Path(args.dataset)
        print(f"Memindai dataset: {target_dir} ...")
        res = scan_dataset(target_dir)
        out_d = reports_base / target_dir.name
        generate_visualizations(res, out_d)

    print("\nVisualisasi dataset selesai!")


if __name__ == "__main__":
    main()
