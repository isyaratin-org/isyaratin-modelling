"""
build_dataset_gabungan.py - Pembuatan & Penyelarasan Dataset Gabungan BISINDO 55 Kelas.

Menggabungkan 3 sumber dataset:
  1. Manual Dataset (20 kelas perbankan/layanan: 100 Ribu Rupiah, Buka, Buku Rekening, dll.)
  2. WL-BISINDO Terpilih (9 kelas: 5W1H [Apa, Siapa, Kapan, Di Mana, Mengapa, Bagaimana] + Saya + Cari + Lagi)
  3. Abjad BISINDO (26 kelas: A s.d. Z)

Total: 55 Kelas Unik dengan penamaan rapi (Canonical Title Case), bebas duplikasi,
dan siap dilatih dalam 1 model sekuensial tunggal (BiLSTM / MGN).
"""

import argparse
import json
import os
import shutil
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from holistic_extractor import EXTRACTOR_VERSION, FEAT_DIM

# Path Default
MANUAL_CACHE_PATH = "data/manual_holistic_lstm_cache.npz"
WL_CACHE_PATH = "data/wl_holistic_lstm_cache.npz"
ABJAD_CACHE_PATH = "data/abjad_holistic_lstm_cache.npz"

OUT_CACHE_PATH = "data/gabungan_holistic_lstm_cache.npz"
OUT_LABELS_JSON = "labels/gabungan_labels.json"
OUT_LABELS_META = "labels/gabungan_labels_meta.json"
OUT_LABELS_TXT = "models/labels.txt"
FLUTTER_MODELS_DIRS = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "isyaratin", "assets", "models"),
    r"d:\Project\4mlai\isyaratin\isyaratin\assets\models",
    r"d:\Project\4mlai\apapun\isyaratin\assets\models",
]

# 1. Definisi Canonical Manual (20 Kelas)
MANUAL_NAME_MAP = {
    "100 ribu rupiah": "100 Ribu Rupiah",
    "baik": "Baik",
    "bantuannya": "Bantuannya",
    "bapak": "Bapak",
    "berapa": "Berapa",
    "buka": "Buka",
    "buku rekening": "Buku Rekening",
    "disini": "Disini",
    "formulir": "Formulir",
    "isi": "Isi",
    "ktp": "KTP",
    "mau": "Mau",
    "membawa": "Membawa",
    "minimal": "Minimal",
    "sama-sama": "Sama-Sama",
    "selamat siang": "Selamat Siang",
    "setoran awal": "Setoran Awal",
    "silakan": "Silakan",
    "tanda tangan": "Tanda Tangan",
    "terima kasih": "Terima Kasih",
}

# 2. Definisi WL-BISINDO Terpilih
# Default: hanya menyisakan kata umum non-tanya ('Lagi', 'Saya')
WL_SELECTED_MAP = {
    "lagi": "Lagi",
    "saya": "Saya",
}

# Daftar lengkap 9 kata WL-BISINDO (termasuk kata tanya 5W1H)
WL_ALL_MAP = {
    "apa": "Apa",
    "bagaimana": "Bagaimana",
    "cari": "Cari",
    "di mana": "Di Mana",
    "kapan": "Kapan",
    "lagi": "Lagi",
    "mengapa": "Mengapa",
    "saya": "Saya",
    "siapa": "Siapa",
}

# 3. Definisi Abjad (26 Kelas: A-Z)
ABJAD_MAP = {chr(i).lower(): chr(i).upper() for i in range(ord("A"), ord("Z") + 1)}


def build_merged_dataset(
    manual_path: str = MANUAL_CACHE_PATH,
    wl_path: str = WL_CACHE_PATH,
    abjad_path: str = ABJAD_CACHE_PATH,
    out_cache_path: str = OUT_CACHE_PATH,
    out_labels_json: str = OUT_LABELS_JSON,
    out_labels_meta: str = OUT_LABELS_META,
    out_labels_txt: str = OUT_LABELS_TXT,
    flutter_dir: Optional[str] = None,
    include_abjad: bool = False,
    include_all_wl: bool = False,
    max_samples_per_class: int = 40,
):
    active_wl_map = WL_ALL_MAP if include_all_wl else WL_SELECTED_MAP
    all_canonical_classes = set()
    all_canonical_classes.update(MANUAL_NAME_MAP.values())
    all_canonical_classes.update(active_wl_map.values())
    if include_abjad:
        all_canonical_classes.update(ABJAD_MAP.values())

    sorted_canonical_classes = sorted(list(all_canonical_classes), key=lambda x: x.lower())
    num_classes = len(sorted_canonical_classes)

    print("\n=========================================================")
    print(f" MEMBANGUN DATASET GABUNGAN {num_classes} KELAS BISINDO")
    print(f" (Max sampel per kelas: {max_samples_per_class} | Abjad: {include_abjad} | WL Semua: {include_all_wl})")
    print("=========================================================\n")

    # Validasi keberadaan file cache
    required_caches = [("Manual", manual_path), ("WL-BISINDO", wl_path)]
    if include_abjad:
        required_caches.append(("Abjad", abjad_path))

    for name, p in required_caches:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File cache {name} tidak ditemukan di: {p}")

    manual_data = np.load(manual_path, allow_pickle=True)
    wl_data = np.load(wl_path, allow_pickle=True)
    abjad_data = np.load(abjad_path, allow_pickle=True) if include_abjad else None

    class_to_idx = {c: idx for idx, c in enumerate(sorted_canonical_classes)}
    class_lower_to_idx = {c.lower(): idx for idx, c in enumerate(sorted_canonical_classes)}

    print(f"[1/4] Daftar {num_classes} Kelas Kanonikal Berhasil Dibuat:")
    for idx, c in enumerate(sorted_canonical_classes):
        print(f"  {idx:2d}: {c}")

    merged_sequences = []
    merged_lengths = []
    merged_labels = []
    merged_filenames = []
    merged_signers = []
    class_sample_counts = {c: 0 for c in sorted_canonical_classes}

    # 2. Proses Dataset Manual (Semua 20 Kelas)
    mn_seqs = manual_data["sequences"]
    mn_lens = manual_data["lengths"]
    mn_lbls = manual_data["labels"]
    mn_classes = [str(c).strip().lower() for c in manual_data["class_names"]]
    mn_files = list(manual_data["filenames"]) if "filenames" in manual_data else [f"manual_{i}.mp4" for i in range(len(mn_seqs))]
    mn_signers = list(manual_data["signers"]) if "signers" in manual_data else [0] * len(mn_seqs)

    mn_count = 0
    for seq, length, old_lbl, fname, signer in zip(mn_seqs, mn_lens, mn_lbls, mn_files, mn_signers):
        raw_name = mn_classes[old_lbl]
        canonical_name = MANUAL_NAME_MAP.get(raw_name)
        if canonical_name and canonical_name in class_to_idx:
            if class_sample_counts[canonical_name] >= max_samples_per_class:
                continue
            new_lbl = class_to_idx[canonical_name]
            merged_sequences.append(seq[:length])
            merged_lengths.append(length)
            merged_labels.append(new_lbl)
            merged_filenames.append(fname)
            merged_signers.append(signer)
            class_sample_counts[canonical_name] += 1
            mn_count += 1
    print(f"\n[2/4] Manual Dataset: {mn_count} sampel dari 20 kelas berhasil dimasukkan.")

    # 3. Proses Dataset WL-BISINDO (Kelas Terpilih, capped at max_samples_per_class)
    wl_seqs = wl_data["sequences"]
    wl_lens = wl_data["lengths"]
    wl_lbls = wl_data["labels"]
    wl_classes = [str(c).strip().lower() for c in wl_data["class_names"]]
    wl_files = list(wl_data["filenames"]) if "filenames" in wl_data else [f"wl_{i}.mp4" for i in range(len(wl_seqs))]
    wl_signers = list(wl_data["signers"]) if "signers" in wl_data else [1] * len(wl_seqs)

    wl_count = 0
    for seq, length, old_lbl, fname, signer in zip(wl_seqs, wl_lens, wl_lbls, wl_files, wl_signers):
        raw_name = wl_classes[old_lbl]
        canonical_name = active_wl_map.get(raw_name)
        if canonical_name and canonical_name in class_to_idx:
            if class_sample_counts[canonical_name] >= max_samples_per_class:
                continue
            new_lbl = class_to_idx[canonical_name]
            merged_sequences.append(seq[:length])
            merged_lengths.append(length)
            merged_labels.append(new_lbl)
            merged_filenames.append(fname)
            merged_signers.append(signer + 10)  # Offset signer id agar unik
            class_sample_counts[canonical_name] += 1
            wl_count += 1
    print(f"[2/4] WL-BISINDO Dataset: {wl_count} sampel dari {len(active_wl_map)} kelas terpilih dimasukkan (max {max_samples_per_class}/kelas).")

    # 4. Proses Dataset Abjad BISINDO (26 Kelas A-Z jika diaktifkan)
    if include_abjad and abjad_data is not None:
        ab_seqs = abjad_data["sequences"]
        ab_lens = abjad_data["lengths"]
        ab_lbls = abjad_data["labels"]
        ab_classes = [str(c).strip().lower() for c in abjad_data["class_names"]]
        ab_files = list(abjad_data["filenames"]) if "filenames" in abjad_data else [f"abjad_{i}.jpg" for i in range(len(ab_seqs))]
        ab_signers = list(abjad_data["signers"]) if "signers" in abjad_data else [99] * len(ab_seqs)

        ab_count = 0
        for seq, length, old_lbl, fname, signer in zip(ab_seqs, ab_lens, ab_lbls, ab_files, ab_signers):
            raw_name = ab_classes[old_lbl]
            canonical_name = ABJAD_MAP.get(raw_name)
            if canonical_name and canonical_name in class_to_idx:
                if class_sample_counts[canonical_name] >= max_samples_per_class:
                    continue
                new_lbl = class_to_idx[canonical_name]
                merged_sequences.append(seq[:length])
                merged_lengths.append(length)
                merged_labels.append(new_lbl)
                merged_filenames.append(fname)
                merged_signers.append(signer)
                class_sample_counts[canonical_name] += 1
                ab_count += 1
        print(f"[2/4] Abjad BISINDO: {ab_count} sampel dari 26 huruf (A-Z) berhasil dimasukkan.")
    else:
        print("[2/4] Abjad BISINDO A-Z dilewati.")

    total_samples = len(merged_sequences)
    print(f"\n=> Total Sampel Keseluruhan Gabungan: {total_samples} sampel.")

    # 5. Susun Array NumPy dengan Padded Length
    max_frames = 120
    padded_seqs = np.zeros((total_samples, max_frames, FEAT_DIM), dtype=np.float32)
    for i, seq in enumerate(merged_sequences):
        L = min(len(seq), max_frames)
        padded_seqs[i, :L] = seq[:L]

    # Simpan Cache Gabungan
    os.makedirs(os.path.dirname(os.path.abspath(out_cache_path)), exist_ok=True)
    np.savez_compressed(
        out_cache_path,
        sequences=padded_seqs,
        lengths=np.array(merged_lengths, dtype=np.int32),
        labels=np.array(merged_labels, dtype=np.int64),
        signers=np.array(merged_signers, dtype=np.int32),
        filenames=np.array(merged_filenames),
        class_names=np.array(sorted_canonical_classes),
        extractor_version=EXTRACTOR_VERSION,
    )
    cache_mb = os.path.getsize(out_cache_path) / (1024 * 1024)
    print(f"[3/4] Cache Gabungan tersimpan di: {out_cache_path} ({cache_mb:.2f} MB)")

    # 6. Simpan File Label JSON & Metadata
    os.makedirs(os.path.dirname(os.path.abspath(out_labels_json)), exist_ok=True)
    with open(out_labels_json, "w", encoding="utf-8") as f:
        json.dump(sorted_canonical_classes, f, indent=2, ensure_ascii=False)
    print(f"[4/4] File label JSON tersimpan di: {out_labels_json}")

    meta_list = []
    for idx, c in enumerate(sorted_canonical_classes):
        if c in MANUAL_NAME_MAP.values():
            category = "Manual Perbankan & Layanan"
        elif c in WL_ALL_MAP.values():
            category = "Percakapan & Kata Umum (WL-BISINDO)"
        else:
            category = "Abjad BISINDO A-Z"

        meta_list.append({
            "id": idx,
            "name": c,
            "category": category,
            "sample_count": class_sample_counts[c],
        })

    with open(out_labels_meta, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, indent=2, ensure_ascii=False)
    print(f"[4/4] Metadata label tersimpan di: {out_labels_meta}")

    # Simpan labels.txt
    os.makedirs(os.path.dirname(os.path.abspath(out_labels_txt)), exist_ok=True)
    with open(out_labels_txt, "w", encoding="utf-8") as f:
        for c in sorted_canonical_classes:
            f.write(f"{c}\n")
    print(f"[4/4] File labels.txt tersimpan di: {out_labels_txt}")

    # Auto-sync labels.txt ke seluruh folder Flutter Assets yang valid
    candidate_flutter_dirs = list(FLUTTER_MODELS_DIRS)
    if flutter_dir:
        candidate_flutter_dirs.insert(0, flutter_dir)

    for fdir in set(candidate_flutter_dirs):
        if fdir and os.path.exists(fdir):
            dest_txt = os.path.join(fdir, "labels.txt")
            shutil.copyfile(out_labels_txt, dest_txt)
            print(f"[FLUTTER SYNC] labels.txt berhasil disinkronisasi ke: {dest_txt}")

    return out_cache_path, out_labels_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Penyatuan & Harmonisasi Dataset Gabungan BISINDO")
    parser.add_argument("--manual-cache", type=str, default=MANUAL_CACHE_PATH)
    parser.add_argument("--wl-cache", type=str, default=WL_CACHE_PATH)
    parser.add_argument("--abjad-cache", type=str, default=ABJAD_CACHE_PATH)
    parser.add_argument("--out-cache", type=str, default=OUT_CACHE_PATH)
    parser.add_argument("--out-labels", type=str, default=OUT_LABELS_JSON)
    parser.add_argument("--flutter-dir", type=str, default=None, help="Folder Flutter assets/models tujuan sync")
    parser.add_argument("--include-abjad", action="store_true", help="Sertakan dataset abjad BISINDO A-Z")
    parser.add_argument("--include-all-wl", action="store_true", help="Sertakan seluruh 9 kelas WL termasuk kata tanya 5W1H")
    parser.add_argument("--max-samples", type=int, default=40, help="Batas maksimal sampel per kelas (default: 40)")
    args = parser.parse_args()

    build_merged_dataset(
        manual_path=args.manual_cache,
        wl_path=args.wl_cache,
        abjad_path=args.abjad_cache,
        out_cache_path=args.out_cache,
        out_labels_json=args.out_labels,
        flutter_dir=args.flutter_dir,
        include_abjad=args.include_abjad,
        include_all_wl=args.include_all_wl,
        max_samples_per_class=args.max_samples,
    )
