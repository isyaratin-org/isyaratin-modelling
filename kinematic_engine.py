"""
Modul Kinematic Disambiguation Engine untuk BISINDO (Config / Schema-Driven).

Membedakan pasangan isyarat yang memiliki kesamaan spasial/visual namun berbeda
secara anatomis dan artikulasi linguistik. Aturan disambiguasi dimuat secara
otomatis dari file konfigurasi JSON ('bisindo_rules.json') sehingga scalable:
penambahan kosakata baru cukup dengan mendaftarkan aturan deklaratif di JSON
tanpa perlu memodifikasi kode Python.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

_CACHED_CONFIG = None


def load_rules_config(config_path: str = "bisindo_rules.json") -> Dict[str, Any]:
    """
    Memuat aturan disambiguasi dari file konfigurasi JSON.
    Menggunakan in-memory cache agar cepat dan efisien.
    """
    global _CACHED_CONFIG
    if _CACHED_CONFIG is not None:
        return _CACHED_CONFIG

    # Prioritas: versi di labels/ (berisi aturan manual + WL) lebih lengkap daripada
    # versi root. Root bisindo_rules.json hanya berisi aturan WL lama.
    candidate_paths = [
        os.path.join("labels", config_path),
        "labels/bisindo_rules.json",
        config_path,
        "bisindo_rules.json",
    ]
    resolved_path = next((p for p in candidate_paths if p and os.path.exists(p)), None)

    if resolved_path:
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                _CACHED_CONFIG = json.load(f)
                return _CACHED_CONFIG
        except Exception as e:
            print(f"[Kinematic Engine] Peringatan: Gagal memuat {resolved_path}: {e}")

    _CACHED_CONFIG = {}
    return _CACHED_CONFIG


def extract_motion_kinematics(window_arr: np.ndarray) -> Dict[str, float]:
    """
    Mengekstrak profil kinematik dan metrik anatomis dari sequence frame (T, 186) atau (T, 400).
    Input:
      window_arr: (T, 186) atau (T, >=186) hasil normalisasi body-centric.
    Output:
      Dictionary berisi metrik spasial-temporal yang relevan.
    """
    feats_186 = window_arr[:, :186]
    seq_len = len(feats_186)
    if seq_len == 0:
        return {}

    coords = feats_186[:, :-3].reshape(seq_len, 61, 3)
    flags = feats_186[:, -3:]  # [lh_present, rh_present, pose_present]

    lh_present = flags[:, 0] > 0.5
    rh_present = flags[:, 1] > 0.5
    bimanual_frames = lh_present & rh_present
    bimanual_ratio = float(np.mean(bimanual_frames))
    any_hand_ratio = float(np.mean(lh_present | rh_present))

    # Landmark Kunci
    # Tangan Kiri (0..20)
    lh_wrist = coords[:, 0]
    lh_thumb_tip = coords[:, 4]
    lh_index_tip = coords[:, 8]
    lh_middle_tip = coords[:, 12]
    lh_ring_tip = coords[:, 16]
    lh_pinky_tip = coords[:, 20]

    # Tangan Kanan (21..41)
    rh_wrist = coords[:, 21]
    rh_thumb_tip = coords[:, 25]
    rh_index_tip = coords[:, 29]
    rh_middle_tip = coords[:, 33]
    rh_ring_tip = coords[:, 37]
    rh_pinky_tip = coords[:, 41]

    # Titik Pose & Kepala (42..60)
    nose = coords[:, 42]
    temple_l = coords[:, 45]     # eye_outer_l (pelipis kiri)
    temple_r = coords[:, 48]     # eye_outer_r (pelipis kanan)
    ear_l = coords[:, 49]        # ear_l (telinga kiri)
    ear_r = coords[:, 50]        # ear_r (telinga kanan)
    mouth_l = coords[:, 51]
    mouth_r = coords[:, 52]
    mouth_center = (mouth_l + mouth_r) * 0.5

    ls = coords[:, 53]           # Bahu kiri
    rs = coords[:, 54]           # Bahu kanan
    torso_center = (ls + rs) * 0.5

    # 1. Jarak Antar Tangan (Bimanual Roof / Atap Rumah)
    dist_index_contact = np.linalg.norm(lh_index_tip - rh_index_tip, axis=-1)
    dist_thumb_contact = np.linalg.norm(lh_thumb_tip - rh_thumb_tip, axis=-1)

    valid_bi = np.where(bimanual_frames)[0]
    if len(valid_bi) > 0:
        min_index_contact = float(np.min(dist_index_contact[valid_bi]))
        min_thumb_contact = float(np.min(dist_thumb_contact[valid_bi]))
        min_roof_contact = min(min_index_contact, min_thumb_contact)
        mean_bi_y = float(np.mean(((lh_index_tip[valid_bi, 1] + rh_index_tip[valid_bi, 1]) * 0.5) - torso_center[valid_bi, 1]))
    else:
        min_index_contact = 999.0
        min_thumb_contact = 999.0
        min_roof_contact = 999.0
        mean_bi_y = 0.0

    # 2. Kedekatan dengan Wajah (Mulut, Telinga, Pelipis)
    dist_lh_mouth = np.linalg.norm(lh_index_tip - mouth_center, axis=-1)
    dist_rh_mouth = np.linalg.norm(rh_index_tip - mouth_center, axis=-1)

    dist_lh_ear = np.linalg.norm(lh_index_tip - ear_l, axis=-1)
    dist_rh_ear = np.linalg.norm(rh_index_tip - ear_r, axis=-1)

    dist_lh_temple = np.linalg.norm(lh_index_tip - temple_l, axis=-1)
    dist_rh_temple = np.linalg.norm(rh_index_tip - temple_r, axis=-1)

    valid_lh = np.where(lh_present)[0]
    valid_rh = np.where(rh_present)[0]

    min_dist_mouth_lh = float(np.min(dist_lh_mouth[valid_lh])) if len(valid_lh) > 0 else 999.0
    min_dist_mouth_rh = float(np.min(dist_rh_mouth[valid_rh])) if len(valid_rh) > 0 else 999.0
    min_dist_mouth = min(min_dist_mouth_lh, min_dist_mouth_rh)

    min_dist_ear_lh = float(np.min(dist_lh_ear[valid_lh])) if len(valid_lh) > 0 else 999.0
    min_dist_ear_rh = float(np.min(dist_rh_ear[valid_rh])) if len(valid_rh) > 0 else 999.0
    min_dist_ear = min(min_dist_ear_lh, min_dist_ear_rh)

    min_dist_temple_lh = float(np.min(dist_lh_temple[valid_lh])) if len(valid_lh) > 0 else 999.0
    min_dist_temple_rh = float(np.min(dist_rh_temple[valid_rh])) if len(valid_rh) > 0 else 999.0
    min_dist_temple = min(min_dist_temple_lh, min_dist_temple_rh)

    # 3. Bentuk Jari (Handshape: Pinch / Menguncup vs Flat / Terbuka vs Pointing)
    pinch_lh = (np.linalg.norm(lh_thumb_tip - lh_index_tip, axis=-1) + np.linalg.norm(lh_thumb_tip - lh_middle_tip, axis=-1)) * 0.5
    pinch_rh = (np.linalg.norm(rh_thumb_tip - rh_index_tip, axis=-1) + np.linalg.norm(rh_thumb_tip - rh_middle_tip, axis=-1)) * 0.5
    mean_pinch_lh = float(np.mean(pinch_lh[valid_lh])) if len(valid_lh) > 0 else 999.0
    mean_pinch_rh = float(np.mean(pinch_rh[valid_rh])) if len(valid_rh) > 0 else 999.0
    mean_pinch = min(mean_pinch_lh, mean_pinch_rh)

    # Pointing Isolation (Telunjuk lurus, jari tengah dan manis tertekuk)
    ext_lh_index = np.linalg.norm(lh_wrist - lh_index_tip, axis=-1)
    ext_lh_middle = np.linalg.norm(lh_wrist - lh_middle_tip, axis=-1)
    pointing_lh = ext_lh_index - ext_lh_middle
    mean_pointing_lh = float(np.mean(pointing_lh[valid_lh])) if len(valid_lh) > 0 else -1.0

    ext_rh_index = np.linalg.norm(rh_wrist - rh_index_tip, axis=-1)
    ext_rh_middle = np.linalg.norm(rh_wrist - rh_middle_tip, axis=-1)
    pointing_rh = ext_rh_index - ext_rh_middle
    mean_pointing_rh = float(np.mean(pointing_rh[valid_rh])) if len(valid_rh) > 0 else -1.0
    pointing_isolation = max(mean_pointing_lh, mean_pointing_rh)

    # Kepalan Tangan Rapat (Fist Clench - Khas Alfabet A)
    fist_lh = (
        np.linalg.norm(lh_wrist - lh_index_tip, axis=-1)
        + np.linalg.norm(lh_wrist - lh_middle_tip, axis=-1)
        + np.linalg.norm(lh_wrist - lh_ring_tip, axis=-1)
        + np.linalg.norm(lh_wrist - lh_pinky_tip, axis=-1)
    ) * 0.25
    fist_rh = (
        np.linalg.norm(rh_wrist - rh_index_tip, axis=-1)
        + np.linalg.norm(rh_wrist - rh_middle_tip, axis=-1)
        + np.linalg.norm(rh_wrist - rh_ring_tip, axis=-1)
        + np.linalg.norm(rh_wrist - rh_pinky_tip, axis=-1)
    ) * 0.25
    mean_fist_lh = float(np.mean(fist_lh[valid_lh])) if len(valid_lh) > 0 else 999.0
    mean_fist_rh = float(np.mean(fist_rh[valid_rh])) if len(valid_rh) > 0 else 999.0
    mean_fist_curl = min(mean_fist_lh, mean_fist_rh)

    # 4. Arah Gerak Maju (Forward Push Z-trajectory - Khas 'Terima Kasih')
    forward_push = 0.0
    if len(valid_rh) > 5:
        z_start = np.mean(rh_wrist[valid_rh[:3], 2])
        z_end = np.mean(rh_wrist[valid_rh[-3:], 2])
        y_start = np.mean(rh_wrist[valid_rh[:3], 1])
        y_end = np.mean(rh_wrist[valid_rh[-3:], 1])
        forward_push = float(z_start - z_end) + float(y_end - y_start)
    elif len(valid_lh) > 5:
        z_start = np.mean(lh_wrist[valid_lh[:3], 2])
        z_end = np.mean(lh_wrist[valid_lh[-3:], 2])
        y_start = np.mean(lh_wrist[valid_lh[:3], 1])
        y_end = np.mean(lh_wrist[valid_lh[-3:], 1])
        forward_push = float(z_start - z_end) + float(y_end - y_start)

    # 5. Elevasi & Lintasan Vertikal (Pagi vs Siang vs Sore)
    # y_travel: positif = tangan bergerak NAIK (Y mengecil ke atas layar), negatif = tangan bergerak TURUN
    # min_y_rel_nose: negatif = pergelangan tangan berada di atas ketinggian hidung/kepala
    y_travel = 0.0
    min_y_rel_nose = 999.0
    active_wrist = rh_wrist if len(valid_rh) >= len(valid_lh) else lh_wrist
    active_valid = valid_rh if len(valid_rh) >= len(valid_lh) else valid_lh

    if len(active_valid) > 5:
        y_s = float(np.mean(active_wrist[active_valid[:4], 1]))
        y_e = float(np.mean(active_wrist[active_valid[-4:], 1]))
        y_travel = y_s - y_e
        min_y_rel_nose = float(np.min(active_wrist[active_valid, 1] - nose[active_valid, 1]))

    # 6. Arah Translasi Spasial Relatif terhadap Tubuh (Datang vs Berangkat)
    # torso_approach: positif = tangan ditarik mendekat ke torso, negatif = tangan didorong menjauh ke depan
    torso_approach = 0.0
    if len(active_valid) > 5:
        d_start_torso = float(np.mean(np.linalg.norm(active_wrist[active_valid[:4]] - torso_center[active_valid[:4]], axis=-1)))
        d_end_torso = float(np.mean(np.linalg.norm(active_wrist[active_valid[-4:]] - torso_center[active_valid[-4:]], axis=-1)))
        torso_approach = d_start_torso - d_end_torso

    # 7. Kontak Telunjuk ke Pergelangan Tangan Lawan (Watch Tap - Khas 'Kapan')
    dist_rh_idx_lh_w = np.linalg.norm(rh_index_tip - lh_wrist, axis=-1)
    dist_lh_idx_rh_w = np.linalg.norm(lh_index_tip - rh_wrist, axis=-1)
    if len(valid_bi) > 0:
        min_wrist_tap = min(float(np.min(dist_rh_idx_lh_w[valid_bi])), float(np.min(dist_lh_idx_rh_w[valid_bi])))
    else:
        min_wrist_tap = 999.0

    # 8. Artikulasi di Dada: Menunjuk Tenang (Saya) vs Mengusap Melingkar (Maaf)
    dist_lh_torso = np.linalg.norm(lh_index_tip - torso_center, axis=-1)
    dist_rh_torso = np.linalg.norm(rh_index_tip - torso_center, axis=-1)
    min_dist_torso = min(
        float(np.min(dist_lh_torso[valid_lh])) if len(valid_lh) > 0 else 999.0,
        float(np.min(dist_rh_torso[valid_rh])) if len(valid_rh) > 0 else 999.0,
    )
    chest_motion_std = 0.0
    if len(active_valid) > 5:
        chest_motion_std = float(np.mean(np.std(active_wrist[active_valid, :2], axis=0)))

    # 9. Dinamika Membuka Buku Bimanual (Buku Rekening vs Tanda Tangan)
    bimanual_open_spread = 0.0
    if len(valid_bi) > 5:
        dx_start = np.mean(np.abs(rh_wrist[valid_bi[:3], 0] - lh_wrist[valid_bi[:3], 0]))
        dx_end = np.mean(np.abs(rh_wrist[valid_bi[-3:], 0] - lh_wrist[valid_bi[-3:], 0]))
        bimanual_open_spread = float(dx_end - dx_start)

    return {
        "bimanual_ratio": bimanual_ratio,
        "any_hand_ratio": any_hand_ratio,
        "min_roof_contact": min_roof_contact,
        "min_index_contact": min_index_contact,
        "mean_bi_y": mean_bi_y,
        "min_dist_mouth": min_dist_mouth,
        "min_dist_ear": min_dist_ear,
        "min_dist_temple": min_dist_temple,
        "mean_pinch": mean_pinch,
        "pointing_isolation": pointing_isolation,
        "mean_fist_curl": mean_fist_curl,
        "forward_push": forward_push,
        "y_travel": y_travel,
        "min_y_rel_nose": min_y_rel_nose,
        "torso_approach": torso_approach,
        "min_wrist_tap": min_wrist_tap,
        "min_dist_torso": min_dist_torso,
        "chest_motion_std": chest_motion_std,
        "bimanual_open_spread": bimanual_open_spread,
    }


def disambiguate_predictions(
    probs: np.ndarray,
    labels: List[str],
    window_arr: np.ndarray,
    rules_config: Optional[dict] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Menyesuaikan vektor probabilitas output model berdasarkan verifikasi fakta kinematik.
    Aturan dimuat secara dinamis dari file JSON ('bisindo_rules.json') atau config argument.

    Return:
      (adjusted_probs, reasoning_notes)
    """
    adj_probs = probs.copy().astype(np.float32)
    reasoning: List[str] = []

    label_to_idx = {name.lower(): i for i, name in enumerate(labels)}
    km = extract_motion_kinematics(window_arr)
    if not km:
        return adj_probs, reasoning

    cfg = rules_config or load_rules_config()
    rules = cfg.get("disambiguation_rules", [])

    for rule in rules:
        r_type = rule.get("type")

        # -------------------------------------------------------------
        # 1. Tipe: bimanual_requirement (Contoh: Rumah vs 1-Tangan / A)
        # -------------------------------------------------------------
        if r_type == "bimanual_requirement":
            target_class = rule.get("target_class", "").lower()
            idx = label_to_idx.get(target_class)
            if idx is not None and adj_probs[idx] > 0.15:
                p_curr = float(adj_probs[idx])
                min_bi = rule.get("min_bimanual_ratio", 0.28)
                max_contact = rule.get("max_roof_contact_dist", 0.38)

                is_bimanual = km["bimanual_ratio"] >= min_bi
                has_contact = km["min_roof_contact"] <= max_contact

                if not is_bimanual or not has_contact:
                    action_viol = rule.get("action_if_violated", {})
                    scale = action_viol.get("prob_scale", 0.05)
                    adj_probs[idx] *= scale
                    reason_txt = action_viol.get("reason", "Syarat 2 tangan tidak terpenuhi.")
                    reasoning.append(
                        f"[{rule.get('id', 'BimanualGate')}] {labels[idx]} diredam ({p_curr*100:.1f}% -> {adj_probs[idx]*100:.1f}%): {reason_txt}"
                    )
                else:
                    action_sat = rule.get("action_if_satisfied", {})
                    scale = action_sat.get("prob_scale", 1.35)
                    adj_probs[idx] *= scale
                    reason_txt = action_sat.get("reason", "Kontak atap kedua tangan terverifikasi.")
                    reasoning.append(
                        f"[{rule.get('id', 'BimanualGate')}] {labels[idx]} dikonfirmasi (+{int((scale-1)*100)}%): {reason_txt}"
                    )

        # -------------------------------------------------------------
        # 2. Tipe: proximity_contrast (Contoh: Tuli vs Ingat)
        # -------------------------------------------------------------
        elif r_type == "proximity_contrast":
            cls_a = rule.get("class_a", "").lower()
            cls_b = rule.get("class_b", "").lower()
            idx_a = label_to_idx.get(cls_a)
            idx_b = label_to_idx.get(cls_b)

            if idx_a is not None or idx_b is not None:
                p_a = float(adj_probs[idx_a]) if idx_a is not None else 0.0
                p_b = float(adj_probs[idx_b]) if idx_b is not None else 0.0

                if (p_a > 0.10) or (p_b > 0.10):
                    tgt_a = rule.get("target_a", "ear")
                    tgt_b = rule.get("target_b", "temple")
                    d_a = km.get(f"min_dist_{tgt_a}", 999.0)
                    d_b = km.get(f"min_dist_{tgt_b}", 999.0)
                    diff = d_a - d_b
                    margin = rule.get("margin", 0.04)
                    boost = rule.get("boost_scale", 1.60)
                    suppress = rule.get("suppress_scale", 0.30)

                    if diff < -margin:
                        if idx_a is not None:
                            adj_probs[idx_a] *= boost
                        if idx_b is not None:
                            adj_probs[idx_b] *= suppress
                        reasoning.append(
                            f"[{rule.get('id', 'ProximityGate')}] {rule.get('reason_a', f'Jari lebih dekat ke {tgt_a}')}"
                        )
                    elif diff > margin:
                        if idx_b is not None:
                            adj_probs[idx_b] *= boost
                        if idx_a is not None:
                            adj_probs[idx_a] *= suppress
                        reasoning.append(
                            f"[{rule.get('id', 'ProximityGate')}] {rule.get('reason_b', f'Jari lebih dekat ke {tgt_b}')}"
                        )

        # -------------------------------------------------------------
        # 3. Tipe: handshape_group (Contoh: Makan vs Terima Kasih vs Kamu)
        # -------------------------------------------------------------
        elif r_type == "handshape_group":
            group_classes = [c.lower() for c in rule.get("classes", [])]
            # Cek apakah ada anggota grup yang relevan
            active_group = [label_to_idx[c] for c in group_classes if c in label_to_idx and adj_probs[label_to_idx[c]] > 0.10]
            if active_group:
                criteria_list = rule.get("criteria", [])
                for crit in criteria_list:
                    c_name = crit.get("target_class", "").lower()
                    c_idx = label_to_idx.get(c_name)
                    if c_idx is None:
                        continue

                    conds = crit.get("conditions", {})
                    passed = True
                    if "max_dist_to_mouth" in conds and km["min_dist_mouth"] > conds["max_dist_to_mouth"]:
                        passed = False
                    if "max_pinch" in conds and km["mean_pinch"] > conds["max_pinch"]:
                        passed = False
                    if "min_pinch" in conds and km["mean_pinch"] < conds["min_pinch"]:
                        passed = False
                    if "min_forward_push" in conds and km["forward_push"] < conds["min_forward_push"]:
                        passed = False
                    if "min_pointing_isolation" in conds and km["pointing_isolation"] < conds["min_pointing_isolation"]:
                        passed = False

                    if passed:
                        boost_val = crit.get("boost", 1.50)
                        suppress_val = crit.get("suppress_others", 0.35)
                        adj_probs[c_idx] *= boost_val

                        # Redam anggota lain dalam grup
                        for other_c in group_classes:
                            if other_c != c_name and other_c in label_to_idx:
                                adj_probs[label_to_idx[other_c]] *= suppress_val

                        reasoning.append(
                            f"[{rule.get('id', 'HandshapeGate')}] {crit.get('reason', f'Memenuhi ciri {labels[c_idx]}')}"
                        )
                        break

        # -------------------------------------------------------------
        # 4. Tipe: trajectory_contrast (Contoh: Datang vs Berangkat, Pagi vs Sore)
        # -------------------------------------------------------------
        elif r_type == "trajectory_contrast":
            cls_a = rule.get("class_a", "").lower()
            cls_b = rule.get("class_b", "").lower()
            idx_a = label_to_idx.get(cls_a)
            idx_b = label_to_idx.get(cls_b)

            if idx_a is not None or idx_b is not None:
                p_a = float(adj_probs[idx_a]) if idx_a is not None else 0.0
                p_b = float(adj_probs[idx_b]) if idx_b is not None else 0.0

                if (p_a > 0.10) or (p_b > 0.10):
                    metric_name = rule.get("metric", "torso_approach")
                    val = km.get(metric_name, 0.0)
                    threshold = rule.get("threshold", 0.06)
                    boost = rule.get("boost_scale", 1.50)
                    suppress = rule.get("suppress_scale", 0.35)

                    if val > threshold:
                        if idx_a is not None:
                            adj_probs[idx_a] *= boost
                        if idx_b is not None:
                            adj_probs[idx_b] *= suppress
                        reasoning.append(
                            f"[{rule.get('id', 'TrajectoryGate')}] {rule.get('reason_a', f'{metric_name} positif ({val:.2f}) -> {labels[idx_a]}')}"
                        )
                    elif val < -threshold:
                        if idx_b is not None:
                            adj_probs[idx_b] *= boost
                        if idx_a is not None:
                            adj_probs[idx_a] *= suppress
                        reasoning.append(
                            f"[{rule.get('id', 'TrajectoryGate')}] {rule.get('reason_b', f'{metric_name} negatif ({val:.2f}) -> {labels[idx_b]}')}"
                        )

        # -------------------------------------------------------------
        # 5. Tipe: wrist_tap_requirement (Contoh: Kapan vs Kata Tanya Lain)
        # -------------------------------------------------------------
        elif r_type == "wrist_tap_requirement":
            target_class = rule.get("target_class", "").lower()
            idx = label_to_idx.get(target_class)
            if idx is not None and adj_probs[idx] > 0.10:
                p_curr = float(adj_probs[idx])
                max_tap = rule.get("max_wrist_tap_dist", 0.28)
                is_tapping = km["min_wrist_tap"] <= max_tap

                if is_tapping:
                    boost = rule.get("boost_scale", 1.60)
                    adj_probs[idx] *= boost
                    reasoning.append(
                        f"[{rule.get('id', 'WristTapGate')}] {labels[idx]} dikonfirmasi: Mengetuk pergelangan tangan ({km['min_wrist_tap']:.2f} <= {max_tap:.2f})"
                    )
                else:
                    suppress = rule.get("suppress_scale", 0.30)
                    adj_probs[idx] *= suppress
                    reasoning.append(
                        f"[{rule.get('id', 'WristTapGate')}] {labels[idx]} diredam: Tidak ada kontak ketukan pergelangan tangan."
                    )

        # -------------------------------------------------------------
        # 6. Tipe: chest_contrast (Contoh: Saya vs Maaf)
        # -------------------------------------------------------------
        elif r_type == "chest_contrast":
            cls_a = rule.get("class_a", "Saya").lower()
            cls_b = rule.get("class_b", "Maaf").lower()
            idx_a = label_to_idx.get(cls_a)
            idx_b = label_to_idx.get(cls_b)

            if idx_a is not None or idx_b is not None:
                p_a = float(adj_probs[idx_a]) if idx_a is not None else 0.0
                p_b = float(adj_probs[idx_b]) if idx_b is not None else 0.0

                if (p_a > 0.10) or (p_b > 0.10):
                    boost = rule.get("boost_scale", 1.50)
                    suppress = rule.get("suppress_scale", 0.35)

                    # Jika telunjuk menunjuk terisolasi dan gerakan dada tenang -> Saya
                    if km["pointing_isolation"] >= 0.08 and km["chest_motion_std"] <= 0.04:
                        if idx_a is not None:
                            adj_probs[idx_a] *= boost
                        if idx_b is not None:
                            adj_probs[idx_b] *= suppress
                        reasoning.append(
                            f"[{rule.get('id', 'ChestGate')}] Telunjuk menunjuk tenang di dada -> Konfirmasi '{labels[idx_a]}'."
                        )
                    # Jika telapak tangan terbuka dan mengusap dada melingkar -> Maaf
                    elif km["chest_motion_std"] > 0.04 and km["mean_pinch"] > 0.20:
                        if idx_b is not None:
                            adj_probs[idx_b] *= boost
                        if idx_a is not None:
                            adj_probs[idx_a] *= suppress
                        reasoning.append(
                            f"[{rule.get('id', 'ChestGate')}] Telapak mengusap melingkar di dada -> Konfirmasi '{labels[idx_b]}'."
                        )

    # Re-normalisasi probabilitas akhir
    total = np.sum(adj_probs)
    if total > 1e-6:
        adj_probs = adj_probs / total

    return adj_probs, reasoning


def apply_bigram_prior(
    probs: np.ndarray,
    labels: List[str],
    last_word: Optional[str] = None,
    rules_config: Optional[dict] = None,
    boost_weight: float = 0.25,
) -> Tuple[np.ndarray, List[str]]:
    """
    Menyesuaikan probabilitas kata kandidat berikutnya berdasarkan kata sebelumnya dalam kalimat (Sentence Context)
    menggunakan Bigram Language Priors dari 'bisindo_rules.json'.

    Contoh:
      Jika last_word == 'Saya', kata seperti 'Makan', 'Belajar', 'Rumah' mendapat dorongan probabilitas (+25%),
      sedangkan kata tanya 'Kapan' / 'Mengapa' tidak didorong.
    """
    if not last_word:
        return probs, []

    adj_probs = probs.copy().astype(np.float32)
    cfg = rules_config or load_rules_config()
    bigrams = cfg.get("bigram_priors", {})
    if not bigrams:
        return adj_probs, []

    # Case-insensitive lookup untuk kata sebelumnya
    matching_key = None
    for k in bigrams:
        if k.lower() == last_word.lower():
            matching_key = k
            break

    if not matching_key:
        return adj_probs, []

    expected_next = [w.lower() for w in bigrams[matching_key]]
    label_to_idx = {name.lower(): i for i, name in enumerate(labels)}

    logs = []
    boosted_words = []
    for w in expected_next:
        idx = label_to_idx.get(w)
        if idx is not None and adj_probs[idx] > 0.05:
            adj_probs[idx] *= (1.0 + boost_weight)
            boosted_words.append(labels[idx])

    if boosted_words:
        total = np.sum(adj_probs)
        if total > 1e-6:
            adj_probs = adj_probs / total
        logs.append(f"[Bigram Prior] Konteks '{matching_key}' -> Prioritas kelanjutan logis: {', '.join(boosted_words)}")

    return adj_probs, logs


def evaluate_idle_gate(
    top_preds: List[Tuple[str, float]],
    motion_kinematics: Dict[str, float],
    min_conf: float = 0.40,
    entropy_margin: float = 0.06,
) -> Tuple[bool, str]:
    """
    Mengevaluasi apakah gerakan yang baru saja dilakukan adalah 'Gerakan Netral / Bukan Isyarat':
      1. Jika tangan tidak aktif (< 15% frame).
      2. Jika keyakinan tertinggi (Top-1) di bawah min_conf.
      3. Jika selisih Top-1 dan Top-2 sangat tipis (distribusi flat / model bingung).
    Return:
      (is_valid_sign, reason)
    """
    if not top_preds:
        return False, "Tidak ada prediksi"

    best_label, best_conf = top_preds[0]
    second_conf = top_preds[1][1] if len(top_preds) > 1 else 0.0

    if motion_kinematics.get("any_hand_ratio", 0.0) < 0.15:
        return False, "Tangan tidak terdeteksi aktif (Gerakan Netral)"

    if best_conf < min_conf:
        return False, f"Keyakinan model terlalu rendah ({best_conf*100:.1f}% < {min_conf*100:.0f}%) - Kemungkinan gerakan bebas / bukan isyarat"

    if (best_conf - second_conf) < entropy_margin and best_conf < 0.35:
        return False, f"Distribusi probabilitas ambigu/flat ({best_label} {best_conf*100:.1f}% vs {top_preds[1][0]} {second_conf*100:.1f}%)"

    return True, "Valid"
