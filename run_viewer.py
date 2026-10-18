"""
run_viewer.py - Menjalankan Local Web Server untuk 3D Avatar Motion Viewer.

Otomatis membuka browser ke:
  http://127.0.0.1:8000/avatar/avatar_viewer.html
"""

import http.server
import webbrowser
import os
import re
import sys
from pathlib import Path

PORT = 8000


class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Sajikan MP4 dengan byte ranges agar browser bisa seek/scrub video."""

    def send_head(self):
        self._range_remaining = None
        path = self.translate_path(self.path)
        if Path(path).suffix.lower() != ".mp4" or not Path(path).is_file():
            return super().send_head()

        video = open(path, "rb")
        size = os.fstat(video.fileno()).st_size
        request_range = self.headers.get("Range")
        if request_range:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", request_range.strip())
            if not match or (not match.group(1) and not match.group(2)):
                video.close()
                self.send_error(416, "Invalid byte range")
                return None

            if match.group(1):
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else size - 1
            else:
                count = int(match.group(2))
                start = max(0, size - count)
                end = size - 1

            if start >= size or end < start or size == 0:
                video.close()
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None

            end = min(end, size - 1)
            length = end - start + 1
            video.seek(start)
            self._range_remaining = length
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            length = size
            self.send_response(200)

        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        return video

    def copyfile(self, source, outputfile):
        remaining = self._range_remaining
        if remaining is None:
            return super().copyfile(source, outputfile)
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)

def run_server(open_browser=True):
    os.chdir(Path(__file__).parent.resolve())

    Handler = RangeRequestHandler
    Handler.extensions_map.update({
        '.glb': 'model/gltf-binary',
        '.gltf': 'model/gltf+json',
        '.json': 'application/json',
        '.mp4': 'video/mp4',
    })

    # Cari port yang tersedia jika 8000 dipakai
    port = PORT
    for p in range(PORT, PORT + 20):
        try:
            httpd = http.server.ThreadingHTTPServer(("", p), Handler)
            port = p
            break
        except OSError:
            continue

    studio_url = f"http://127.0.0.1:{port}/avatar/pose_studio.html"
    viewer_url = f"http://127.0.0.1:{port}/avatar/avatar_viewer.html"

    print("=" * 70)
    print("🚀 BISINDO 3D Avatar Web Server Aktif!")
    print(f"  • 🎨 3D Avatar Pose Studio (Custom Editor) : {studio_url}")
    print(f"  • 📹 3D Avatar Video Player (Comparison)  : {viewer_url}")
    print("=" * 70)
    print("Membuka browser otomatis ke Pose Studio... Tekan Ctrl+C untuk berhenti.\n")

    if open_browser:
        try:
            webbrowser.open(studio_url)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer dihentikan.")
        httpd.server_close()

if __name__ == "__main__":
    run_server(open_browser="--no-browser" not in sys.argv)
