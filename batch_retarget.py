"""Batch extraction + retargeting dari dataset rekaman manual."""

import argparse
import json
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path
from motion_extractor import extract_motion_from_video
from retargeter import AvatarRetargeter

DATASET_ROOT = Path("datasets/manual")
LABELS_PATH = DATASET_ROOT / "labeled.json"
BROWSER_VIDEO_ROOT = Path("avatar/videos_manual")


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")


def choose_representative_video(folder: Path) -> Path:
    videos = sorted(folder.glob("*.mp4"), key=lambda path: path.name.lower())
    if not videos:
        raise FileNotFoundError(f"Folder manual tidak memiliki video MP4: {folder}")

    # Rekaman bernama timestamp adalah take kamera utama yang konsisten. Ambil
    # take tengah agar tidak selalu memakai percobaan pertama/pemanasan.
    primary_takes = [path for path in videos if path.name[:1].isdigit()]
    candidates = primary_takes or videos
    return candidates[len(candidates) // 2]


def load_manual_signs():
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Label dataset manual tidak ditemukan: {LABELS_PATH}")
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    signs = []
    for item in labels:
        folder = DATASET_ROOT / item["folder"]
        video = choose_representative_video(folder)
        signs.append({
            "id": slugify(item["gloss"]),
            "gloss": item["gloss"],
            "english": item.get("english", item["gloss"]),
            "video": video,
            "label": item["id"],
        })
    return signs


def ensure_browser_video(source: Path, sign_id: str) -> Path:
    """Buat proxy H.264 karena rekaman manual HEVC tidak selalu didukung browser."""
    target = BROWSER_VIDEO_ROOT / f"{sign_id}.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg tidak ditemukan; diperlukan untuk membuat video H.264 browser.")

    print(f"[VIDEO] Membuat proxy H.264: {target}")
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source),
            "-map", "0:v:0", "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(target),
        ],
        check=True,
    )
    return target


def run_batch(retarget_only: bool = False):
    retargeter = AvatarRetargeter()
    manifest = []

    for item in load_manual_signs():
        sign_id = item["id"]
        gloss = item["gloss"]
        video_p = Path(item["video"])
        
        if not video_p.exists():
            print(f"[SKIP] Video tidak ditemukan: {video_p}")
            continue

        print(f"\n==========================================")
        print(f"Memproses Motion: {gloss} ({item['english']})")
        print(f"Source: {video_p}")
        print(f"==========================================")

        raw_json = f"avatar/extracted_landmarks_manual_{sign_id}.json"
        motion_json = f"avatar/motion_manual_{sign_id}.json"
        browser_video = BROWSER_VIDEO_ROOT / f"{sign_id}.mp4" if retarget_only else ensure_browser_video(video_p, sign_id)
        if retarget_only and not browser_video.exists():
            raise FileNotFoundError(f"Proxy video belum ada: {browser_video}")
        browser_video_rel = browser_video.relative_to(Path("avatar")).as_posix()

        # 1. Ekstraksi Landmarks
        if retarget_only:
            with open(raw_json, "r", encoding="utf-8") as f:
                extracted = json.load(f)
        else:
            extracted = extract_motion_from_video(
                video_path=str(video_p),
                out_json_path=raw_json,
                gloss=gloss,
            )

        # 2. Retargeting ke Avatar Skeleton
        motion_data = retargeter.process_frames(extracted)
        motion_data["gloss"] = gloss
        motion_data["english"] = item["english"]
        motion_data["videoRelPath"] = browser_video_rel
        motion_data["sourceVideo"] = video_p.as_posix()
        motion_data["sourceDataset"] = "manual"

        with open(motion_json, "w", encoding="utf-8") as f:
            json.dump(motion_data, f, indent=2)

        print(f"[OK] Motion {gloss} disimpan ke: {motion_json}")

        manifest.append({
            "id": sign_id,
            "gloss": gloss,
            "english": item["english"],
            "label": item["label"],
            "video": browser_video_rel,
            "sourceVideo": video_p.as_posix(),
            "motionFile": f"motion_manual_{sign_id}.json",
            "fps": motion_data["sourceFps"],
            "durationMs": motion_data["durationMs"],
            "totalFrames": motion_data["totalFrames"],
            "sourceDataset": "manual",
        })

    # Simpan manifest daftar motion untuk dibaca otomatis oleh GUI Viewer
    manifest_p = Path("avatar/motions_manifest.json")
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[DONE] Manifest motion berhasil disimpan ke: {manifest_p}")
    print(f"Total motion siap digunakan: {len(manifest)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retarget-only", action="store_true",
        help="Gunakan landmark dan video browser yang sudah ada; perbarui animasi saja.",
    )
    run_batch(retarget_only=parser.parse_args().retarget_only)
