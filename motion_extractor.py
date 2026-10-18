"""
motion_extractor.py - Ekstraksi 3D Skeleton & Hand Landmarks dari Video BISINDO.

Mengekstrak:
  - 33 Pose landmarks (dengan fokus pada Upper Body: Shoulders, Elbows, Wrists, Head).
  - 21 Left Hand landmarks (Wrist, Thumb, Index, Middle, Ring, Pinky).
  - 21 Right Hand landmarks.
  - Mempertahankan timestamp milidetik dan FPS asli dari video sumber.
  - Menyimpan data intermediate ke JSON.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import mediapipe as mp
import numpy as np


def extract_motion_from_video(
    video_path: str,
    out_json_path: Optional[str] = None,
    gloss: Optional[str] = None,
    smooth_window: int = 3,
) -> Dict:
    p = Path(video_path)
    if not p.exists():
        raise FileNotFoundError(f"Video tidak ditemukan: {p}")

    cap = cv2.VideoCapture(str(p))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 1.0:
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_ms = (total_frames / fps) * 1000.0 if fps > 0 else 0.0

    print(f"Mengekstrak gerakan dari: {p.name}")
    print(f"  Resolusi : {width}x{height} | FPS: {fps:.2f} | Total Frame: {total_frames}")

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames_data: List[Dict] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = (frame_idx / fps) * 1000.0
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(rgb)

        # 1. Pose Landmarks
        pose_pts = None
        pose_world_pts = None
        if res.pose_landmarks:
            pose_pts = [
                {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                for lm in res.pose_landmarks.landmark
            ]
        if res.pose_world_landmarks:
            pose_world_pts = [
                {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                for lm in res.pose_world_landmarks.landmark
            ]

        # 2. Left Hand
        lh_pts = None
        if res.left_hand_landmarks:
            lh_pts = [
                {"x": lm.x, "y": lm.y, "z": lm.z}
                for lm in res.left_hand_landmarks.landmark
            ]

        # 3. Right Hand
        rh_pts = None
        if res.right_hand_landmarks:
            rh_pts = [
                {"x": lm.x, "y": lm.y, "z": lm.z}
                for lm in res.right_hand_landmarks.landmark
            ]

        frames_data.append({
            "frameIndex": frame_idx,
            "timestampMs": round(timestamp_ms, 2),
            "pose": pose_pts,
            "poseWorld": pose_world_pts,
            "leftHand": lh_pts,
            "rightHand": rh_pts,
            "hasPose": pose_pts is not None,
            "hasLeftHand": lh_pts is not None,
            "hasRightHand": rh_pts is not None,
        })
        frame_idx += 1

    cap.release()
    holistic.close()

    print(f"  Berhasil mengekstrak {len(frames_data)} frame landmarks.")

    result = {
        "videoFile": p.name,
        "gloss": gloss or "Unknown",
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "totalFrames": len(frames_data),
        "durationMs": round(duration_ms, 2),
        "frames": frames_data,
    }

    if out_json_path:
        out_p = Path(out_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Data landmark tersimpan di: {out_p}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="datasets/WL-BISINDO/signer0_label0_sample1.mp4")
    parser.add_argument("--out", type=str, default="avatar/extracted_landmarks_air.json")
    parser.add_argument("--gloss", type=str, default="Air")
    args = parser.parse_args()

    extract_motion_from_video(args.video, args.out, gloss=args.gloss)
