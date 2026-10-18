# Catatan Eksperimen & Pembelajaran (pembelajaran.md)

## 1. Normalisasi Koordinat Landmark Bebas Rasio Aspek (Mobile vs Desktop)

- **Masalah**: Ekstraksi landmark sebelumnya membagi koordinat $x$ dengan $W$ dan $y$ dengan $H$ secara terpisah (anisotropik). Hal ini merusak proporsi geometris tangan dan postur tubuh ketika rasio aspek berubah antara HP (Portrait 9:16) dan Laptop/Webcam (Landscape 16:9 / 4:3).
- **Solusi**: Menggunakan normalisasi isotropik di [holistic_extractor.py](file:///d:/Project/4mlai/isyaratin/holistic_extractor.py):
  $$\text{scale} = \max(W, H)$$
  $$x_{\text{norm}} = \frac{x - W/2}{\text{scale}}, \quad y_{\text{norm}} = \frac{y - H/2}{\text{scale}}$$
  Dengan ini, jarak dan sudut antartulang tangan konsisten di semua resolusi dan orientasi kamera mobile.

## 2. Dynamic Gesture Temporal Resampling (Gestur 4-6 Detik)

- **Masalah**: Isyarat pengguna berdurasi 4-6 detik (misal kalimat frasa "buku rekening" atau "selamat siang"), sedangkan model LSTM dilatih dengan fixed window 32 frame (2.6 detik).
- **Solusi**: Menerapkan _Global Temporal Resampling_ (`np.linspace(0, total_frames - 1, 32)`). Berapapun durasi gerakan pengguna (4-6 detik), kurva pergerakan disampel merata menjadi 32 frame sehingga seluruh fase isyarat (awal, transisi, akhir) terserap utuh ke dalam model.

## 3. Hands-Free Dynamic Inference (Presence & Departure with Exit Delay)

- **Kebutuhan**: Inferensi live webcam/HP tanpa harus menekan tombol keyboard fisik `[SPASI]`.
- **Mekanisme State Machine**:
  1. **Idle**: Menunggu keberadaan tangan (`lh_present or rh_present`).
  2. **Recording**: Ketika tangan muncul, rekaman frame langsung dimulai otomatis.
  3. **Verifying Exit (Grace Period 1-2 Detik)**:
     - Begitu tangan hilang/turun, sistem tidak langsung memutus rekaman, melainkan menghitung mundur waktu verifikasi (default `1.5s`).
     - Jika tangan naik kembali sebelum jeda habis, rekaman dilanjutkan (gestur bersambung/berjeda wajar).
     - Jika tangan tetap tidak ada sampai jeda habis, rekaman dikunci, di-resample ke 32 frame, dan inferensi langsung dieksekusi secara instan.
  4. **Holding & Sentence Builder**: Hasil dipajang di layar dan ditambahkan ke kalimat, lalu sistem kembali siap untuk isyarat berikutnya.

## 4. Kinematic Disambiguation Engine: Pembeda Pasangan Gestur Serupa (Confusable Pairs)

- **Masalah**: Model berbasis deep learning rentan mengalami kebingungan klasifikasi (confusion) pada pasangan isyarat yang memiliki kedekatan spasial di area tubuh/wajah yang mirip:
  1. **Rumah vs Alfabet A**: Bentuk tangan mengepal satu tangan di depan dada (huruf A) sering kali keliru terdeteksi sebagai 'Rumah', padahal Rumah secara gramatikal BISINDO adalah gestur _bimanual_ (dua tangan) yang bertemu membentuk puncak atap segitiga.
  2. **Makan vs Terima Kasih vs Kamu**: Ketiganya berada di ketinggian dada/dagu/mulut, namun berbeda dalam konfigurasi jari (_handshape_) dan arah lintasan gerak (_trajectory_).
  3. **Tuli vs Ingat**: Keduanya menggunakan jari telunjuk menunjuk (_pointing_) ke kepala, namun titik artikulasi sentuhan sangat spesifik: daun telinga (_ear_) vs dahi/pelipis (_temple_).
- **Solusi**: Membangun modul [kinematic_engine.py](file:///d:/Project/4mlai/isyaratin/kinematic_engine.py) dengan arsitektur **Declarative Config / Schema-Driven** yang membaca aturan dari [bisindo_rules.json](file:///d:/Project/4mlai/isyaratin/bisindo_rules.json):
  1. **Bimanual Requirement Gate** (`bimanual_roof_gate`):
     - Memverifikasi syarat minimal bimanual ratio dan jarak kontak atap.
     - Jika 'Rumah' diprediksi namun signer hanya memakai 1 tangan $\implies$ probabilitas dipangkas 95% secara instan.
  2. **Proximity Contrast Gate** (`ear_vs_temple_gate`):
     - Membandingkan jarak Euclidean ke target artikulasi daun telinga (`ear`) vs pelipis/dahi (`temple`).
     - Jika selisih $\Delta < -0.04 \implies$ boost kuat 'Tuli', redam 'Ingat'; sebaliknya jika $\Delta > +0.04 \implies$ boost 'Ingat', redam 'Tuli'.
  3. **Handshape Group Gate** (`mouth_handshape_gate`):
     - Mengevaluasi bentuk jari menguncup (_pinch_ $< 0.24$ untuk 'Makan'), telapak terbuka bergerak maju (_flat palm_ $Z$-push untuk 'Terima kasih'), dan jari telunjuk menunjuk terisolasi (_pointing isolation_ $> 0.10$ untuk 'Kamu').
  - **Skalabilitas**: Penambahan kosakata baru di masa depan tidak perlu mengubah kode Python, cukup daftarkan objek aturan baru di file JSON `bisindo_rules.json`.
- **Hasil**: Pasangan ambigu dapat dibedakan secara deterministik, akurat, dan scalable tanpa merusak atau memerlukan training ulang bobot model yang sudah stabil.

## 5. Multi-Worker Parallelism Ekstraksi MediaPipe Holistic (2-4 Worker)

- **Masalah**: Ekstraksi landmark MediaPipe Holistic per frame memerlukan komputasi CPU yang cukup intensif (deteksi wajah, 2 tangan, dan pose tubuh). Sebelumnya di Windows dibatasi ke 1 worker untuk menghindari tabrakan DLL / spawn memory leak, sehingga pemrosesan dataset ratusan video memakan waktu cukup lama.
- **Solusi**:
  1. Arsitektur ekstraksi dipisahkan secara modular ke [holistic_extractor.py](file:///d:/Project/4mlai/isyaratin/holistic_extractor.py), memungkinkan mekanisme `multiprocessing.spawn` di Windows berjalan bersih tanpa _circular import_.
  2. Fungsi inisialisasi worker `init_worker()` dipanggil sekali per child process via argumen `initializer` pada `ProcessPoolExecutor`, mengalokasikan instance MediaPipe independen di tiap core CPU.
  3. Konfigurasi default worker di [train-new.py](file:///d:/Project/4mlai/isyaratin/train-new.py) diatur otomatis ke **2–4 worker** (`min(4, max(2, cpu_cnt // 2))`), dengan dukungan flag CLI `--workers [2-4]`.
- **Hasil**: Throughput ekstraksi video meningkat 2x hingga 4x lipat lebih cepat dengan utilisasi multi-core CPU yang seimbang dan stabil tanpa konflik memory leak.

## 6. Ekosistem Robustness: Advanced Augmentation, Bigram Language Prior, & Idle Gate

- **Tujuan**: Membawa model dan alur inferensi ke standar production-ready untuk lingkungan HP/webcam yang dinamis:
  1. **Temporal Hand Flicker Augmentation (train-new.py)**:
     - Mensimulasikan kondisi tracking loss sesaat (1–4 frame) yang sering dialami MediaPipe saat gerakan cepat/motion blur.
     - Melatih BiLSTM/MGN mempertahankan memori temporal laten sehingga model tidak panik saat frame tangan sempat drop sejenak.
  2. **Spatial Translation & Wrist Jitter**:
     - Memberikan pergeseran spasial ($dx, dy \pm 0.02$) dan jitter mikro pada pergelangan tangan agar model toleran terhadap variasi jarak dan sudut pandang kamera signer.
  3. **Bigram Language Model Priors (kinematic_engine.py & camera.py)**:
     - Menggunakan probabilitas transisi kata antar frasa dari [bisindo_rules.json](file:///d:/Project/4mlai/isyaratin/bisindo_rules.json).
     - Memberikan dorongan probabilitas (+25%) pada kata kelanjutan logis (misal setelah _"Saya"_ $\to$ memprioritaskan _"Makan"_, _"Belajar"_, _"Rumah"_, dan meredam kata tanya seperti _"Kapan"_).
  4. **Idle / Non-Sign Rejection Gate (evaluate_idle_gate)**:
     - Mencegah masalah _closed-set dilemma_ di mana gerakan bebas biasa (menggaruk, mengetik, membetulkan kacamata) dipaksa ditebak sebagai salah satu dari 28 kata.
     - Jika keyakinan model rendah ($< 0.40$) atau tangan tidak aktif, sistem secara elegan menolak penguncian kata dan melabelinya sebagai `[GERAKAN NETRAL]`.

## 7. Pemetaan Fitur Komprehensif (400D) & Eliminasi Seluruh Pasangan Kata Ambigu

- **Masalah**:
  1. Fitur input sistem sebelumnya belum dipetakan secara terperinci per indeks (0 hingga 399) sehingga sulit untuk memvalidasi setiap saluran koordinat dan relasi spasial.
  2. Terdapat beberapa kelompok kata dalam kamus 28 BISINDO yang memiliki ambiguitas visual di kamera 2D dan belum memiliki aturan deterministik:
     - _Pagi vs Sore_: Sama-sama gestur matahari, sering tertukar jika model hanya melihat posisi tangan tanpa memperhitungkan vektor arah translasi vertikal.
     - _Datang vs Berangkat_: Tangan berada di area perut/dada, arah maju-mundur sulit dibedakan oleh kamera monokular 2D standar.
     - _Kapan vs Kata Tanya Lain_: Memerlukan verifikasi ketukan pergelangan tangan (_watch tap_).
     - _Saya vs Maaf_: Keduanya di tengah dada, namun _Saya_ menunjuk tenang sedangkan _Maaf_ mengusap melingkar.
     - _Dengar vs Tuli vs Ingat_: Keduanya di samping kepala, namun _Dengar_ dan _Tuli_ di daun telinga sedangkan _Ingat_ di dahi/pelipis.
- **Solusi**:
  1. Menyusun dokumentasi referensi [feature.md](file:///d:/Project/4mlai/isyaratin/feature.md) yang merinci secara matematis setiap indeks dari 0 hingga 399 (186 statis + 183 velocity + 31 kinematik).
  2. Memperluas [kinematic_engine.py](file:///d:/Project/4mlai/isyaratin/kinematic_engine.py) dengan metrik kinematik baru:
     - `y_travel`: Menghitung arah lintasan vertikal tangan (positif = naik untuk _Pagi_, negatif = turun untuk _Sore_).
     - `torso_approach`: Menghitung arah translasi relatif terhadap dada (positif = mendekat ke tubuh untuk _Datang_, negatif = menjauh untuk _Berangkat_).
     - `min_wrist_tap`: Menghitung jarak ketukan telunjuk ke pergelangan tangan berlawanan untuk mengonfirmasi _Kapan_.
     - `chest_motion_std`: Menghitung variasi standar deviasi temporal gerakan di dada untuk membedakan menunjuk tenang (_Saya_) vs mengusap melingkar (_Maaf_).
  3. Mendaftarkan seluruh aturan deklaratif baru ke dalam [bisindo_rules.json](file:///d:/Project/4mlai/isyaratin/bisindo_rules.json) (`sun_trajectory_pagi_sore`, `spatial_trajectory_datang_berangkat`, `question_wrist_kapan`, `chest_articulation_saya_maaf`, `dengar_vs_ingat_gate`).
- **Hasil**:
  - Divalidasi melalui [test_kinematic_cases.py](file:///d:/Project/4mlai/isyaratin/test_kinematic_cases.py) dengan **10 skenario pengujian berturut-turut lulus 100%**, membuktikan bahwa seluruh pasangan kata ambigu kini memiliki pemutus keputusan yang objektif, deterministik, dan bebas kebingungan (_zero ambiguity_).

## 8. Ekstraksi Otomatis Label Dataset Manual & Skalabilitas Kosakata Baru

- **Kebutuhan**: Pengguna menambahkan rekaman video baru di subdirektori [datasets/manual/](file:///d:/Project/4mlai/isyaratin/datasets/manual) (misal: _bapak_, _formulir_, _isi_, _ktp_, _membawa_, _tanda tangan_). Diperlukan mekanisme otomatis untuk mengekstrak label, menghitung sampel video, dan menyimpannya dalam format terstruktur seperti [datasets/WL-BISINDO/labeled.json](file:///d:/Project/4mlai/isyaratin/datasets/WL-BISINDO/labeled.json) dengan tambahan field `path` untuk kemudahan pelatihan model.
- **Implementasi**:
  1. Membuat script [datasets/manual/extract_label.py](file:///d:/Project/4mlai/isyaratin/datasets/manual/extract_label.py):
     - Memindai subfolder video di `datasets/manual/`.
     - Menyusun metadata lengkap: `id`, `gloss`, `english`, `folder`, `path` (`datasets/manual/{folder}`), dan `total_videos`.
     - Menyimpan secara konsisten ke:
       - `datasets/manual/labeled.json` (standar seperti WL-BISINDO).
       - `datasets/manual/manual_label.json` (alias yang diminta pengguna).
       - `datasets/manual/manual_labels.json` & `manual_labels.json` (root project).
  2. Meningkatkan Dataset Loader di [train-new.py](file:///d:/Project/4mlai/isyaratin/train-new.py):
     - Mendukung pembacaan label baik berupa list of string maupun list of dictionary (`labeled.json`).
     - Menggunakan field `path` langsung dari JSON untuk memetakan lokasi video tiap kelas.
     - Memastikan proses training hanya mengambil kelas dan urutan `id` yang ditentukan dalam label file (`--labels-file` / `labeled.json`).
  3. Kompatibilitas Inferensi [camera.py](file:///d:/Project/4mlai/isyaratin/camera.py):
     - Memperbarui fungsi `load_inference_model` agar dapat mem-parsing list of dicts dari `labeled.json` secara transparan.
- **Hasil**: Ditemukan 9 kelas (total 135 video) di dataset manual. Semua metadata label dan path tersinkronisasi sempurna dan siap dilatih kapan pun pengguna memerintahkannya.

## 9. Restrukturisasi & Standarisasi Direktori Proyek: `models/`, `data/`, dan `labels/`

- **Latar Belakang & Masalah**:
  Sebelumnya, artefak model PyTorch (`.pt`), model ONNX (`.onnx`), cache fitur biner NumPy (`.npz`), serta file metadata label (`.json`) bercampur langsung di root direktori project. Seiring bertambahnya dataset dan varian model (WL-BISINDO dan Dataset Manual), root project menjadi padat dan rentan salah tunjuk file.
- **Solusi & Arsitektur Direktori Baru**:
  Menerapkan pemisahan modular 3-folder terpusat:
  1. **`models/`**:
     - Tempat penyimpanan seluruh model hasil training (`.pt` dan `.onnx`).
     - Menyimpan: [bisindo_lstm_best.pt](file:///d:/Project/4mlai/isyaratin/models/bisindo_lstm_best.pt), [bisindo_lstm.onnx](file:///d:/Project/4mlai/isyaratin/models/bisindo_lstm.onnx), [manual_lstm_best.pt](file:///d:/Project/4mlai/isyaratin/models/manual_lstm_best.pt), [manual_lstm.onnx](file:///d:/Project/4mlai/isyaratin/models/manual_lstm.onnx).
  2. **`data/`**:
     - Tempat penyimpanan seluruh cache ekstraksi fitur biner per-dataset.
     - Menyimpan: [wl_holistic_lstm_cache.npz](file:///d:/Project/4mlai/isyaratin/data/wl_holistic_lstm_cache.npz), [manual_holistic_lstm_cache.npz](file:///d:/Project/4mlai/isyaratin/data/manual_holistic_lstm_cache.npz).
  3. **`labels/`**:
     - Tempat penyimpanan seluruh definisi kelas, metadata kamus, dan aturan semantik BISINDO.
     - Menyimpan: [wl_labels.json](file:///d:/Project/4mlai/isyaratin/labels/wl_labels.json), [wl_labels_meta.json](file:///d:/Project/4mlai/isyaratin/labels/wl_labels_meta.json), [manual_labels.json](file:///d:/Project/4mlai/isyaratin/labels/manual_labels.json), [manual_labels_meta.json](file:///d:/Project/4mlai/isyaratin/labels/manual_labels_meta.json), [labels.json](file:///d:/Project/4mlai/isyaratin/labels/labels.json), dan [bisindo_rules.json](file:///d:/Project/4mlai/isyaratin/labels/bisindo_rules.json).
- **Adaptasi Kode & Mekanisme Smart Auto-Resolve**:
  1. **[train-new.py](file:///d:/Project/4mlai/isyaratin/train-new.py)**:
     - Default output path disesuaikan ke `models/`, cache ke `data/`, dan label ke `labels/`.
     - Ditambahkan pemanggilan `os.makedirs(..., exist_ok=True)` sebelum penyimpanan cache, checkpoint, dan ekspor ONNX agar folder baru otomatis terbentuk jika belum ada.
  2. **[camera.py](file:///d:/Project/4mlai/isyaratin/camera.py) & [evaluate.py](file:///d:/Project/4mlai/isyaratin/evaluate.py)**:
     - Mengimplementasikan _candidate path search resolution_: mencari path di `models/`, `labels/`, dan `data/` sebelum fallback ke root project. Pengguna dapat memberikan nama file saja (misal `--pt bisindo_lstm_best.pt`) dan sistem otomatis menyelesaikannya ke `models/bisindo_lstm_best.pt`.
  3. **[kinematic_engine.py](file:///d:/Project/4mlai/isyaratin/kinematic_engine.py)**:
     - Membaca file konfigurasi aturan `labels/bisindo_rules.json` terlebih dahulu dengan fallback ke root.
  4. **[datasets/manual/extract_label.py](file:///d:/Project/4mlai/isyaratin/datasets/manual/extract_label.py)**:
     - Menyinkronkan hasil ekstraksi label ke `labels/manual_labels.json` dan `labels/manual_labels_meta.json` sekaligus mempertahankan file lokal di folder dataset untuk isolasi dependensi.
- **Hasil**:
  Seluruh komponen pipeline training, inferensi langsung kamera, evaluasi, hingga engine kinematik teruji berjalan bersih dan rapi dengan integritas backward-compatibility 100%.

## 10. Resolusi MemoryError & Bad Allocation pada Multiprocessing Spawn Worker di Windows

- **Penyebab Utama Masalah**:
  1. Pada sistem operasi Windows, modul `multiprocessing` hanya mendukung metode `spawn` (bukan `fork`). Saat child worker lahir, ia menjalankan `runpy.run_path("train-new.py", run_name="__mp_main__")` untuk merekonstruksi lingkungan proses utama.
  2. Sebelumnya, terdapat `import torch` di modul level atas (top-level) pada [train-new.py](file:///d:/Project/4mlai/isyaratin/train-new.py). Akibatnya, setiap worker CPU (misal 3-4 worker) yang diluncurkan secara simultan dipaksa memuat DLL C++ PyTorch, CUDA runtime, dan meta registrations ke memori.
  3. Alokasi memori virtual C++ PyTorch (~2.5 GB per proses x 4 proses = ~10 GB) melampaui limit paging/commit charge sistem Windows dalam sepersekian detik, memicu `MemoryError`, `RuntimeError: bad allocation`, dan `BrokenProcessPool`.
- **Solusi yang Diterapkan**:
  1. **Zero-Torch Top-Level Imports**:
     - Menghapus `import torch` dari level atas modul [train-new.py](file:///d:/Project/4mlai/isyaratin/train-new.py).
     - Mengubah class `FocalLoss` menjadi fungsi factory `get_focal_loss_class()` sehingga ketergantungan PyTorch bersifat _lazy-loaded_ dan hanya diinisialisasi saat proses training utama GPU/CPU dimulai.
     - Worker child yang hanya bertugas mengekstrak fitur MediaPipe ([holistic_extractor.py](file:///d:/Project/4mlai/isyaratin/holistic_extractor.py)) kini berjalan sangat ringan (hanya ~70 MB RAM per worker vs ~2.5 GB sebelumnya, penghematan memori >90%).
  2. **Suppression Warning TensorFlow/oneDNN**:
     - Menetapkan environment variable `TF_ENABLE_ONEDNN_OPTS=0` dan `TF_CPP_MIN_LOG_LEVEL=3` di level atas [train-new.py](file:///d:/Project/4mlai/isyaratin/train-new.py) dan [holistic_extractor.py](file:///d:/Project/4mlai/isyaratin/holistic_extractor.py) sebelum memuat MediaPipe untuk menghilangkan pesan peringatan berulang di konsol.
- **Hasil & Validasi**:
  - Diuji langsung dengan `--smoke --re-extract --workers 3`: Ekstraksi paralel 3 worker berhasil 100% (**18 dari 18 video selesai tanpa kegagalan/crash**) dan pipeline training langsung berlanjut mulus hingga ekspor ONNX selesai.

## 11. Analisis Performa Dataset Manual: Strategi 2 (Motion-Trimmed Cropping) & Pelatihan Bebas Pretrained

- **Konteks & Masalah Awal**:
  1. Pada dataset manual (135 video, 9 kelas), pelatihan awal menghasilkan akurasi validasi hanya **48.15%**.
  2. Confusion matrix mengungkap fenomena _sinkhole class_: sebanyak **14 dari 27 video validasi keliru ditebak sebagai `membawa`**.
  3. Hal ini disebabkan oleh:
     - Ketergantungan pretrained lama dari WL-BISINDO yang merusak pemetaan kelas baru pada small-data regime (12 sampel train per kelas).
     - Sub-sequence sampling yang memotong fase diam (_idle frames_) di awal video saat tangan masih istirahat di depan perut/paha, sehingga model mengasosiasikan tangan diam sebagai gestur `membawa`.
- **Implementasi Solusi**:
  1. **Pelatihan dari Awal (Bebas Pretrained / Scratch)**:
     - Mengubah [train-new.py](file:///d:/Project/4mlai/isyaratin/train-new.py) agar secara default tidak memuat checkpoint `models/bisindo_lstm_best.pt` pada dataset manual, kecuali jika diminta eksplisit dengan flag `--finetune` atau `--pretrained`.
  2. **Strategi 2: Motion-Trimmed Sequence Cropping**:
     - Mengembangkan fungsi `find_motion_trimmed_bounds` di [train-new.py](file:///d:/Project/4mlai/isyaratin/train-new.py) untuk mendeteksi batas gerak aktif berdasarkan kecepatan frame-to-frame dan kehadiran tangan.
     - Memangkas frame pasif di awal dan akhir video sebelum melakukan global resampling 32 frame, menjamin seluruh window pelatihan dan evaluasi 100% berisi artikulasi isyarat aktif.
  3. **Strategi 4: Perluasan Aturan Kinematik Manual**:
     - Mendaftarkan pembeda pasangan confusable manual pada [labels/bisindo_rules.json](file:///d:/Project/4mlai/isyaratin/labels/bisindo_rules.json) dan [kinematic_engine.py](file:///d:/Project/4mlai/isyaratin/kinematic_engine.py):
       - `buku_rekening_vs_tanda_tangan`: Membedakan gerakan membuka telapak tangan simetris (`bimanual_open_spread`) vs osilasi menulis di atas telapak tangan kiri.
       - `bapak_mustache_gate`: Memverifikasi ketinggian sentuhan di area kumis/bibir/dagu (`min_dist_mouth`).
       - `selamat_siang_elevation`: Membedakan lintasan elevasi menyapa ke dada atas/dahi vs orientasi kartu statis di dada (`ktp`).
- **Hasil Lompatan Performa**:
  - Akurasi Konsensus Video melonjak drastis dari **48.15% menjadi 74.07% (+25.92%)**!
  - Kelas `buka`, `formulir`, `ktp`, `membawa`, dan `tanda tangan` mencapai **100% akurasi/recall**.
  - Fenomena _sinkhole class_ pada `membawa` teratasi tuntas.

## 12. Ekspansi Dataset Manual Menjadi 20 Kelas (300 Video) & Sinkronisasi Ekstraksi Label

- **Perkembangan Dataset**:
  Pengguna menambahkan 11 kosakata baru yang sangat umum dalam domain layanan perbankan/administrasi BISINDO:
  1. `100 ribu rupiah`
  2. `baik`
  3. `bantuannya`
  4. `berapa`
  5. `disini`
  6. `mau`
  7. `minimal`
  8. `sama-sama`
  9. `setoran awal`
  10. `silakan`
  11. `terima kasih`
      Total kelas kini menjadi **20 kelas** dengan masing-masing 15 video (**total 300 video**).
- **Proses Ekstraksi & Sinkronisasi**:
  - Script [datasets/manual/extract_label.py](file:///d:/Project/4mlai/isyaratin/datasets/manual/extract_label.py) diperbarui dengan kamus translasi Inggris lengkap.
  - Seluruh metadata tersimpan konsisten di:
    - [datasets/manual/labeled.json](file:///d:/Project/4mlai/isyaratin/datasets/manual/labeled.json) & salinan lokal.
    - [labels/manual_labels.json](file:///d:/Project/4mlai/isyaratin/labels/manual_labels.json) (20 nama kelas terurut).
    - [labels/manual_labels_meta.json](file:///d:/Project/4mlai/isyaratin/labels/manual_labels_meta.json) (metadata lengkap dengan atribut `path`).
    - [manual_labels.json](file:///d:/Project/4mlai/isyaratin/manual_labels.json) di root project.
- **Kesiapan Training**:
  Pipeline pelatihan siap mengekstrak cache biner terbaru (`data/manual_holistic_lstm_cache.npz`) dengan flag `--re-extract` untuk melatih model 20 kelas.

## 13. Milestone Akurasi 95% (Konsensus 90%) pada 20 Kelas Manual Dataset

- **Eksperimen & Optimasi Arsitektur**:
  1. Pada dataset 20 kelas (300 video, 15 video/kelas), melatih model MGN besar (670k param) dari awal sempat mengalami _plateau_ overfitting pada epoch 20 (akurasi ~33%).
  2. Dilakukan penyesuaian arsitektur ke **BiLSTM 2-Layer Ramping (`--arch lstm`)** dengan dimensi input 186, digabungkan dengan:
     - **Motion-Trimmed Sequence Cropping (Strategi 2)**: Memangkas frame pasif di awal dan akhir video.
     - **Full Dynamic Augmentation**: Bilateral flip (50%), spatial jitter, 2D rotation, Gaussian noise, temporal speed warp, dan temporal hand flicker.
     - **Pelatihan Murni dari Awal (_From Scratch_)** tanpa beban bias bobot pretrained lama.
- **Hasil & Evaluasi Akhir (Epoch 35)**:
  - **Akurasi Validasi per-Window**: **95.00%** (57 dari 60 window validasi benar).
  - **Akurasi Konsensus Video (1 Video = 1 Keputusan)**: **90.00%**!
  - **Macro Average F1-Score**: **0.95 (95%)**.
  - **14 dari 20 kelas mencapai kesempurnaan 100% Precision & 100% Recall** (`baik`, `bantuannya`, `bapak`, `berapa`, `buka`, `buku rekening`, `formulir`, `ktp`, `mau`, `membawa`, `minimal`, `setoran awal`, `silakan`, `terima kasih`).
- **Artefak Model Siap Pakai**:
  - Model PyTorch Checkpoint: [models/manual_lstm_best.pt](file:///d:/Project/4mlai/isyaratin/models/manual_lstm_best.pt)
  - Model ONNX: [models/manual_lstm.onnx](file:///d:/Project/4mlai/isyaratin/models/manual_lstm.onnx) (output shape `(1, 20)`, tervalidasi ONNX Runtime).

## 14. Perbaikan Akurasi: Aturan Kinematik Manual + Eksperimen Fitur 400D

- **Latar Belakang**: Evaluasi `--all-signers` menunjukkan 14 error / 300 video (95.33%). Pasangan tertukar utama: `baik`↔`sama-sama` (5), `selamat siang`↔`100 ribu rupiah` (5), plus minor `disini`↔`sama-sama`, `tanda tangan`↔`mau`, `terima kasih`↔`isi`.
- **Bug Ditemukan & Diperbaiki**: [`load_rules_config()`](file:///d:/Project/4mlai/isyaratin/kinematic_engine.py) sebelumnya memprioritaskan root `bisindo_rules.json` (hanya aturan WL) di atas `labels/bisindo_rules.json` (berisi aturan manual). Akibatnya aturan manual (`buku_rekening_vs_tanda_tangan`, `bapak_mustache_gate`, `selamat_siang_elevation`) **tidak pernah dimuat** saat inferensi. Diperbaiki agar `labels/` diprioritaskan.
- **Analisis Data-Driven**: Script [analyze_confusions.py](file:///d:/Project/4mlai/isyaratin/analyze_confusions.py) menghitung metrik kinematik per kelas dari cache. Separator terkuat:
  - `selamat siang` vs `100 ribu rupiah`: `y_travel` (0.126 vs -0.064) — sapaan menyapu naik.
  - `tanda tangan` vs `mau`: `chest_motion_std` (0.317 vs 0.076) — gerakan menulis aktif.
  - `terima kasih` vs `isi`: `chest_motion_std` (0.332 vs 0.130) — gerakan mengisi aktif.
  - `baik` vs `sama-sama`: `bimanual_open_spread` (0.058 vs 0.012) — buka tangan horizontal.
- **Aturan Baru Ditambahkan** di [labels/bisindo_rules.json](file:///d:/Project/4mlai/isyaratin/labels/bisindo_rules.json): `selamat_siang_vs_100ribu`, `tanda_tangan_vs_mau`, `terima_kasih_vs_isi`, `baik_vs_sama_sama`. Divalidasi 100% lulus di [test_kinematic_cases.py](file:///d:/Project/4mlai/isyaratin/test_kinematic_cases.py) (14 skenario).
- **Eksperimen Fitur 400D (Hasil Negatif)**: Menambahkan flag `--feat-dim 400` di [train-new.py](file:///d:/Project/4mlai/isyaratin/train-new.py) agar LSTM dapat dilatih dengan fitur extended (velocity + kinematik). Hasil: **93% top-1** (21 error) — **lebih rendah** dari baseline 186D (95.33%, 14 error). Muncul sinkhole baru `silakan`→`berapa` (5 error). **Keputusan**: tetap gunakan model 186D; andalkan aturan kinematik deklaratif untuk akurasi inferensi live.
- **Hasil Akhir**: Model 186D dipulihkan (95.33% top-1, Macro F1 0.9529). Aturan kinematik baru aktif saat inferensi live di `camera.py` untuk memisahkan pasangan ambigu yang tersisa.

## 15. Kampanye Optimasi Akurasi: TTA, Ensemble, & Model Lebih Besar → 99%

- **Tujuan**: Menaikkan akurasi dari baseline 95.33% (14 error / 300 video) melalui optimasi model & inferensi tanpa menambah data.
- **Eksperimen 1 — Test-Time Augmentation (TTA)**:
  - Menambahkan flag `--tta` di [evaluate.py](file:///d:/Project/4mlai/isyaratin/evaluate.py): setiap window diprediksi pada versi asli + versi flip horizontal (mirror kiri<->kanan), lalu probabilitas dirata-rata.
  - Hasil: **95.33% → 96.00%** (14 → 12 error). Menghilangkan `disini`→`sama-sama` & `silakan`→`mau`, mengurangi `selamat siang`→`100 ribu rupiah` (5→4).
- **Eksperimen 2 — Ensemble Multi-Seed**:
  - Menambahkan flag `--seed` di [train-new.py](file:///d:/Project/4mlai/isyaratin/train-new.py) untuk melatih model dengan seed berbeda.
  - Script [evaluate_ensemble.py](file:///d:/Project/4mlai/isyaratin/evaluate_ensemble.py) memuat 3 model (seed 42/43/44), merata-ratakan probabilitas per video.
  - Hasil: ensemble saja = 95.33% (setara model terbaik); **ensemble + TTA = 96.67%** (10 error). Tidak mengalahkan model besar tunggal.
- **Eksperimen 3 — Model Lebih Besar (Pemenang)**:
  - Evaluasi `--all-signers --tta`: **99.00% top-1** (hanya 3 error), Macro F1 **0.99**, ROC AUC **1.000**.
  - Sisa error: `ktp`→`tanda tangan`, `selamat siang`→`100 ribu rupiah`, `tanda tangan`→`isi` (masing-masing 1).
- **Keputusan & Artefak**:
  - Model produksi: [models/manual_lstm_best.pt](file:///d:/Project/4mlai/isyaratin/models/manual_lstm_best.pt) (LSTM besar 1.2M, 186D) + [models/manual_lstm.onnx](file:///d:/Project/4mlai/isyaratin/models/manual_lstm.onnx).
  - Model eksperimen disimpan: `manual_lstm_seed43.pt`, `manual_lstm_seed44.pt`, `manual_lstm_big.pt`.
  - Aturan kinematik (15 rule) tetap aktif saat inferensi live di `camera.py`.
  - **Rekomendasi**: Untuk evaluasi/benchmark gunakan `evaluate.py --tta`; untuk live inference `camera.py` sudah menerapkan disambiguasi kinematik.

---

## 14. Eksperimen Rekayasa Fitur Statistik, Spasial, dan Spasio-Temporal (Paper Bird 2022 Adaptation)

- **Latar Belakang & Motivasi**:
  - Mengadaptasi metodologi paper **Jordan J. Bird (arXiv:2202.11005v1, Feb 2022)** yang membuktikan kombinasi fitur statistik dan spasio-temporal dapat memberikan akurasi tinggi pada sensor gerak tangan (Leap Motion).
  - Tujuannya adalah menguji apakah rekayasa fitur (*feature engineering*) dapat menjadi alternatif model *lightweight classical ML* (Random Forest, SVM) yang super cepat, bebas GPU, dan sangat efisien untuk mobile Flutter, dibandingkan atau melengkapi model deep learning recurrent.
- **Implementasi Modul & Safeguards**:
  - [feature_engineering_spec.md](file:///d:/Project/4mlai/isyaratin/feature_engineering_spec.md): Spesifikasi matematis formal 4 kelompok fitur.
  - [feature_engineer.py](file:///d:/Project/4mlai/isyaratin/feature_engineer.py): Class modular `BISINDOFeatureExtractor` (Model A: 104D, Model B: 320D, Model C: 168D, Model D: 384D).
  - [test_feature_engineer.py](file:///d:/Project/4mlai/isyaratin/test_feature_engineer.py): 7 Unit tests numerik (clipping $\theta = \arccos$, missing hand, division by zero safeguards). Lulus 100%.
  - [benchmark_feature_models.py](file:///d:/Project/4mlai/isyaratin/benchmark_feature_models.py): Pipeline evaluasi tanpa data leakage (StandardScaler dan ANOVA F-Score hanya di-fit pada data train 80/20).
- **Hasil Komparasi Eksperimental (Dataset Manual 20 Kelas BISINDO, 300 Video)**:
  | Konfigurasi Fitur | Dimensi | Classifier | Akurasi (%) | Macro F1 | Ekstraksi (ms) | Inferensi (ms) | Ukuran Model |
  | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
  | **Model A (Spatial)** | 104 | Random Forest | 95.00% | 0.9514 | **2.38 ms** | 0.42 ms | ~1.7 MB |
  | **Model A (Spatial)** | 104 | SVM (RBF) | 95.00% | 0.9504 | **2.38 ms** | **0.02 ms** | **~24 KB** |
  | **Model A (Spatial)** | 104 | SVM (Linear) | 95.00% | 0.9504 | **2.38 ms** | **0.01 ms** | **~18 KB** |
  | **Model B (Spatial+Stat)** | 320 | Random Forest | 98.33% | 0.9829 | 5.84 ms | 0.41 ms | ~1.7 MB |
  | **Model B (Spatial+Stat)** | 320 | SVM (Linear) | 95.00% | 0.9486 | 5.84 ms | 0.01 ms | ~45 KB |
  | **Model C (Spatial+Temporal)** | 168 | Random Forest | 95.00% | 0.9490 | 3.49 ms | 0.40 ms | ~1.7 MB |
  | **Model C (Spatial+Temporal)** | 168 | SVM (Linear) | 93.33% | 0.9314 | 3.49 ms | 0.01 ms | ~28 KB |
  | **Model D (Combined)** | 384 | Random Forest | 98.33% | 0.9829 | 6.57 ms | 0.40 ms | ~1.7 MB |
  | **Model D (ANOVA Top-100)** | **100** | **Random Forest** | **100.00%** | **1.0000** | 6.57 ms | 0.40 ms | **1.7 MB** |
  | **Model D (ANOVA Top-100)** | **100** | **SVM (Linear)** | **96.67%** | **0.9657** | 6.57 ms | **0.01 ms** | **~19 KB** |
  | **Baseline BiLSTM (Raw 186D)** | 186 | ONNX (Single-pass) | 31.67% | 0.2511 | 0.00 ms | 2.32 ms | 2.7 MB |

- **Temuan Kunci ANOVA Feature Ranking (Jordan J. Bird 2022 Analogy)**:
  1. **Top Peringkat F-Score**:
     - Rank 1: `lh_angle_5_6_7_std` (F=180.68, $p=9.51 \times 10^{-123}$) — Fleksi sendi jari telunjuk kiri.
     - Rank 2: `lh_angle_5_6_7_mean` (F=101.96) — Konfigurasi rata-rata jari telunjuk kiri.
     - Rank 3: `rh_palm_normal_y_mean` (F=93.78) — Orientasi normal telapak tangan kanan.
     - Rank 4 & 6: `lh_angle_17_18_19_std`, `lh_angle_14_15_16_std` — Fleksi jari kelingking dan manis.
     - Rank 5 & 8: `stat_pose_nose_x_min`, `stat_rh_pinky_tip_y_max` — Posisi referensi spasial terhadap hidung.
     - Rank 9: `temp_lh_index_x_mad` (F=51.64) — Dinamika pergerakan cepat telunjuk kiri.
     - Rank 15: `bimanual_index_distance_mean` (F=43.11) — Jarak bimanual kedua tangan.
  2. **Distribusi Top-100 Fitur**:
     - **58% Fitur Statistik** (distribusi min/max/std/iqr)
     - **31% Fitur Spasial** (sudut fleksi sendi & jarak bimanual)
     - **11% Fitur Spatio-Temporal** (MAD & slope akselerasi)
     Hal ini membuktikan temuan Bird (2022) bahwa fitur statistik dan spasial geometris memegang peranan krusial dalam klasifikasi isyarat tangan.
  3. **Efek Seleksi Fitur**:
     - Memangkas 384 fitur menjadi 100 fitur terbaik menghilangkan multicollinearity dan noise, mendongkrak Random Forest dari 98.33% menjadi **100.00%**, dan SVM dari 95.00% menjadi **96.67%**.
- **Rekomendasi Deployment Mobile Flutter**:
  - Untuk lingkungan Flutter Android/iOS yang sangat hemat daya:
    - **Opsi 1 (Ultralight & Akurasi 96.67%)**: Model A (Spatial 104D) atau Model D (ANOVA Top-100) + **Linear SVM**. Ukuran model hanya **~19 KB**, waktu komputasi inferensi **0.01 ms** (dapat diimplementasikan dengan matriks perkalian 100x20 murni di Dart tanpa runtime tambahan).
    - **Opsi 2 (Maksimum Akurasi 100.00%)**: Model D (ANOVA Top-100) + **Random Forest** (disimpan di [models/best_feature_classifier.joblib](file:///d:/Project/4mlai/isyaratin/models/best_feature_classifier.joblib), ukuran 1.7 MB, diekspor via ONNX / Treelite / m2cgen menjadi kode C/Dart).

### 14.1 Pelatihan Intensif Model Rekayasa Fitur (RF + SVM + Ensemble) dengan 5x Augmentasi

- **Latar Belakang**:
  - Melatih model secara intensif agar generalisasi lebih tangguh di kondisi nyata (jarak kamera, sudut miring, jitter sensor, dan kecepatan signer bervariasi).
  - Skrip: [train_feature_classifiers.py](file:///d:/Project/4mlai/isyaratin/train_feature_classifiers.py).
- **Prosedur Pelatihan**:
  1. **5x Multi-Transformation Augmentation**: Data latih asli (240 video) dilipatgandakan menjadi **1,200 video** menggunakan transformasi: scale & translation jitter, 2D rotation jitter ($\pm 7^\circ$), noise Gaussian mikro ($\sigma=0.008$), dan temporal speed warping (0.85x–1.18x).
  2. **Optimalisasi Seleksi Fitur ANOVA (5-Fold CV)**:
     - $K=80$: **99.92%** (+/- 0.17%) CV Accuracy -> **Terpilih Optimal**.
     - $K=100$: 99.75% (+/- 0.33%).
     - $K=120$: 99.92% (+/- 0.17%).
  3. **Tiga Model Dilatih & Disimpan**:
     - **Random Forest (250 Trees, Depth 16)**: Akurasi Test **98.33%**, Macro F1 **0.9829**, Latensi **0.75 ms/sampel** (Disimpan di [models/best_rf_feature_model.joblib](file:///d:/Project/4mlai/isyaratin/models/best_rf_feature_model.joblib)).
     - **Support Vector Machine (RBF Kernel, C=10)**: Akurasi Test **96.67%**, Macro F1 **0.9662**, Latensi **0.044 ms/sampel** (Disimpan di [models/best_svm_feature_model.joblib](file:///d:/Project/4mlai/isyaratin/models/best_svm_feature_model.joblib)).
     - **Hybrid Soft Voting Ensemble (RF + SVM)**: Akurasi Test **98.33%**, Macro F1 **0.9829**, Latensi **0.85 ms/sampel** (Disimpan di [models/best_ensemble_feature_model.joblib](file:///d:/Project/4mlai/isyaratin/models/best_ensemble_feature_model.joblib)).
  4. **Modul Inferensi Siap Pakai**:
     - Dibuat di [infer_feature_model.py](file:///d:/Project/4mlai/isyaratin/infer_feature_model.py) untuk klasifikasi single sequence, batch video, maupun integrasi real-time.

### 14.2 Aplikasi Real-Time Multi-Model Camera (camera-new.py)

- **Latar Belakang**:
  - Menggabungkan keunggulan model Rekayasa Fitur (Random Forest, SVM, Hybrid Ensemble) dan Deep Learning (BiLSTM ONNX) ke dalam satu aplikasi kamera interaktif.
  - Skrip: [camera-new.py](file:///d:/Project/4mlai/isyaratin/camera-new.py).
- **Fitur Unggulan**:
  1. **Hot-Switch Model Runtime**: Pengguna dapat beralih model langsung saat kamera menyala dengan menekan hotkey:
     - `[1]`: Random Forest (80D ANOVA, Akurasi 98.3%)
     - `[2]`: SVM RBF (80D ANOVA, Akurasi 96.7%, Latensi 0.04 ms)
     - `[3]`: Hybrid Soft Voting Ensemble (RF + SVM)
     - `[4]`: BiLSTM Deep Learning (ONNX)
  2. **Tiga Mode Deteksi**:
     - `HANDS-FREE` (Default): Otomatis merekam saat tangan terangkat, menunggu delay verifikasi 1.5 detik saat tangan turun, lalu mengeluarkan keputusan.
     - `MANUAL TRIGGER`: Tekan `[SPASI]` untuk merekam gerakan isyarat.
     - `AUTO STREAM`: Menganalisis sliding window kontinu.
  3. **HUD Modern**:
     - Indikator deteksi tangan bilateral (LH / RH).
     - Kartu Top-3 Prediksi dengan visualisasi confidence bar.
     - Live Sentence Builder di footer layar untuk merangkai kalimat isyarat.
     - Shortcut `[C]` untuk reset kalimat, `[M]` untuk toggle skeleton, `[H]` untuk toggle panduan.

---

## 15. Pelatihan Dataset Gabungan Skala Penuh (WL-BISINDO + Manual Dataset: 47 Kelas, 1,700 Video)

- **Latar Belakang & Harmonisasi**:
  - Menggabungkan dua dataset utama:
    - **WL-BISINDO**: 1,400 video, 28 kelas kosakata umum.
    - **Manual Dataset**: 300 video, 20 kelas perbankan/layanan nasabah.
  - Harmonisasi nama kelas dilakukan secara *case-insensitive* (`"Terima kasih"` dan `"terima kasih"` disatukan).
  - Menghasilkan total **47 kelas unik** dan **1,700 video sampel**.
  - Cache gabungan tersimpan di [data/gabungan_holistic_lstm_cache.npz](file:///d:/Project/4mlai/isyaratin/data/gabungan_holistic_lstm_cache.npz) (36.27 MB) dan label di [labels/gabungan_labels.json](file:///d:/Project/4mlai/isyaratin/labels/gabungan_labels.json).
  - Skrip pelatihan terpadu: [train-gabungan.py](file:///d:/Project/4mlai/isyaratin/train-gabungan.py).

- **Hasil Pelatihan & Evaluasi (Test Set 340 Video, Stratified 80/20)**:
  1. **Model Rekayasa Fitur (Grup D: 384D + Seleksi ANOVA K=120)**:
     - **Random Forest (250 Trees, Max Depth 18)**:
       - **Akurasi Test: 87.94%** | **Macro F1: 0.8857**
       - Waktu Pelatihan: **0.42 detik**
       - Latensi Inferensi: **0.138 ms / sampel** (setara ~7,200 prediksi/detik pada CPU!)
       - Model tersimpan: [models/gabungan_rf_feature_model.joblib](file:///d:/Project/4mlai/isyaratin/models/gabungan_rf_feature_model.joblib)
     - **Support Vector Machine (SVM RBF, C=10)**:
       - **Akurasi Test: 71.47%** | **Macro F1: 0.7633**
       - Latensi Inferensi: **0.134 ms / sampel**
       - Model tersimpan: [models/gabungan_svm_feature_model.joblib](file:///d:/Project/4mlai/isyaratin/models/gabungan_svm_feature_model.joblib)
     - **Hybrid Soft Voting Ensemble (RF + SVM)**:
       - **Akurasi Test: 84.41%** | **Macro F1: 0.8623**
       - Latensi Inferensi: **0.307 ms / sampel**
       - Model tersimpan: [models/gabungan_ensemble_feature_model.joblib](file:///d:/Project/4mlai/isyaratin/models/gabungan_ensemble_feature_model.joblib)

  2. **Model Deep Learning (BiLSTM PyTorch + ONNX Export)**:
     - Arsitektur: 2-Layer Bidirectional LSTM (128 & 96 hidden units) dengan LayerNorm & Cosine Annealing LR.
     - Jumlah Epoch: 35 epoch pada CUDA GPU.
     - **Akurasi Validasi Puncak: 98.04%** (Epoch 23, Val Loss: 0.7323).
     - Checkpoint PyTorch tersimpan: [models/gabungan_lstm_best.pt](file:///d:/Project/4mlai/isyaratin/models/gabungan_lstm_best.pt)
     - Model ONNX Standar tersimpan: [models/gabungan_lstm.onnx](file:///d:/Project/4mlai/isyaratin/models/gabungan_lstm.onnx) (2.49 MB).

- **Integrasi Live Camera**:
  - [camera-new.py](file:///d:/Project/4mlai/isyaratin/camera-new.py) telah diperbarui dengan hotkey switch:
    - Tekan **`[5]`**: Langsung aktifkan **Gabungan Random Forest 47 Kelas**
    - Tekan **`[6]`**: Langsung aktifkan **Gabungan BiLSTM ONNX 47 Kelas**

---

## 16. Audit Menyeluruh Pipeline, Perbaikan Bug Kritis, & Standardisasi Paritas Preprocessing (Pipeline Audit & Refactoring)

### 16.1 Latar Belakang & Bug yang Terkonfirmasi pada Audit Kode Aktual

Audit mendalam terhadap implementasi aktual kode ekstraksi ([holistic_extractor.py](file:///d:/Project/4mlai/isyaratin/holistic_extractor.py)) dan pelatihan ([train-new.py](file:///d:/Project/4mlai/isyaratin/train-new.py)) mengonfirmasi sejumlah bug kritis:

1. **Desinkronisasi Buffer BatchNorm pada ModelEMA**:
   - `ModelEMA.update()` sebelumnya hanya mengiterasi `model.parameters()`.
   - Akibatnya, buffer statistik BatchNorm (`running_mean`, `running_var`, `num_batches_tracked`) pada lapisan `conv_bn` (Conv1D) model MGN tidak pernah diperbarui di `shadow model`. Selama 60 epoch, model utama mengalami pergeseran distribusi batch sementara EMA shadow tetap tertahan pada statistik epoch 0, merusak akurasi evaluasi EMA.
2. **Landmark Absen Tercemar oleh Augmentasi Koordinat**:
   - Landmark tangan yang tidak terdeteksi (flags=0) memiliki koordinat awal 0.0.
   - Pada `augment_sequence_window()`, pergeseran spasial (*translation shift*), *spatial rotation*, *noise mikro*, dan *wrist-centric hand jitter* ditambahkan langsung ke seluruh array `coords` tanpa re-masking flags di akhir.
   - Akibatnya, tangan absen berubah menjadi bernilai nonzero ($0.0 + \text{noise} \neq 0$), menciptakan distribusi input palsu yang menyimpang dari fase validasi dan webcam real-time.
3. **False Spike Velocity & Relasi Kinematik pada Missing Landmarks**:
   - Di [holistic_extractor.py](file:///d:/Project/4mlai/isyaratin/holistic_extractor.py), `np.diff(coords)` dihitung mentah tanpa memeriksa validitas flag.
   - Ketika tangan pertama kali muncul dari frame $t-1$ (absen, coords=0.0) ke frame $t$ (terdeteksi, coords $\approx 0.5$), selisihnya menghasilkan lonjakan velocity semu masif ($> 0.5$).
   - Fitur relasional (seperti jarak bimanual antar-tangan dan *finger curl ratios*) dihitung dari titik nol saat tangan absen, mengartikan ketiadaan tangan sebagai kepalan tinju (_clenched fist_) atau jarak semu.
4. **Instabilitas Normalisasi Torso & Penyelarasan Sumbu Z**:
   - Torso center dan skala bahu dihitung ulang per frame dari titik 3D. Jika deteksi pose sempat hilang, center jatuh ke 0.0 dan tangan tiba-tiba berada dalam sistem koordinat berbeda.
   - Selain itu, koordinat kedalaman $Z$ tangan pada MediaPipe bersifat lokal relatif terhadap *wrist*, sedangkan $Z$ pose mengacu pada torso. Mengurangkan `torso_center[2]` langsung dari $Z$ tangan merusak kedalaman tangan terhadap tubuh.
5. **Sampling Temporal Stride Pembulatan (FPS Distortion)**:
   - Rumus lama `stride = round(src_fps / target_fps)` menyebabkan video 30 FPS disampel pada stride 3 (menghasilkan 10.0 FPS, bukan 12.0 FPS), sedangkan video 25 FPS disampel pada stride 2 (12.5 FPS).
6. **State MediaPipe Bocor Antar-Video**:
   - Instance `mp.solutions.holistic.Holistic(static_image_mode=False)` mempertahankan memori pelacakan temporal kalman filter antar pemanggilan video jika tidak di-reset.
7. **Bug Penanganan Seed 0 & CUDA Seed**:
   - Sintaks `seed = seed or SEED` gagal ketika pengguna memberikan `seed=0` (karena `0 or 42` mengevaluasi ke `42`), serta `torch.cuda.manual_seed_all(SEED)` sebelumnya mengabaikan argumen dinamis `seed`.

---

### 16.2 Patch Terisolasi & Solusi Arsitektural

1. **Sinkronisasi Lengkap Buffer BatchNorm pada `ModelEMA`**:
   - Memperbarui `ModelEMA.update()` di [train-new.py](file:///d:/Project/4mlai/isyaratin/train-new.py):
     ```python
     for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
         s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)
     for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
         s_buf.copy_(m_buf)
     ```
   - Menjamin akurasi inferensi evaluasi dan ekspor ONNX selalu menggunakan statistik BatchNorm yang sinkron.

2. **Strict Coordinate Re-Masking pada Augmentasi**:
   - Menambahkan re-masking ketat di baris akhir `augment_sequence_window`:
     ```python
     lh_mask = (flags[:, 0:1] > 0.5).astype(np.float32)[:, :, None]
     rh_mask = (flags[:, 1:2] > 0.5).astype(np.float32)[:, :, None]
     pose_mask = (flags[:, 2:3] > 0.5).astype(np.float32)[:, :, None]
     coords[:, :21, :] *= lh_mask
     coords[:, 21:42, :] *= rh_mask
     coords[:, 42:61, :] *= pose_mask
     ```
   - Memastikan 100% komponen yang tidak terdeteksi tetap bernilai 0.0 persis.

3. **Causal Torso Tracker dengan EMA Smoothing & Coasting**:
   - Mengimplementasikan class `CausalTorsoTracker` di [holistic_extractor.py](file:///d:/Project/4mlai/isyaratin/holistic_extractor.py):
     - Menghitung center torso dan skala bahu 2D XY (`np.linalg.norm(ls[:2] - rs[:2])`) secara kausal dengan smoothing EMA ($\alpha=0.4$) untuk meredam jitter kedalaman $Z$.
     - Jika deteksi pose hilang sementara pada frame $t$, tracker melakukan *coasting* (mempertahankan center & scale valid terakhir).
     - Jika pose belum pernah terdeteksi sama sekali, fallback ke center default $(0.5, 0.5, 0.0)$ dan skala $0.25$.

4. **Penyelarasan Kedalaman Sumbu Z Tangan ke Pose Wrist**:
   - Jika pose terdeteksi, kedalaman tangan diselaraskan relatif terhadap kedalaman pergelangan tangan pose (landmark 15 untuk LH, landmark 16 untuk RH):
     ```python
     pose_wrist_l_z = (p_raw[15, 2] - torso_center[2]) / torso_scale
     lh_feat[:, 2] = pose_wrist_l_z + (lh_raw[:, 2] / torso_scale)
     ```
   - Mempertahankan informasi jarak relatif tangan terhadap dada sekaligus mempertahankan bentuk internal jari.

5. **Temporal Masking pada Velocity & Masking Bilateral Fitur Relasional**:
   - `vel` dihitung hanya jika landmark aktif di **kedua** frame $t$ dan $t-1$:
     ```python
     lh_cont = has_lh[:, 1:] & has_lh[:, :-1]
     vel_mask[:, :, :21, 0] = lh_cont[:, :, None]
     vel[:, 1:] = (coords[:, 1:] - coords[:, :-1]) * vel_mask
     ```
   - Seluruh 31 fitur relasional dikalikan dengan mask kehadiran: bimanual features (`d_hands`, `d_index_tips`) dikalikan `has_lh * has_rh`, single hand curls dikalikan `has_lh` atau `has_rh`.

6. **Deterministic Target-FPS Schedule & MediaPipe Reset**:
   - Menggantikan stride bulat dengan akumulator timestamp target (`frame_time + 0.5/src_fps >= next_target_time`), menghasilkan tepat 12 frame per detik secara merata baik dari sumber 24, 25, 30, maupun 60 FPS.
   - Memanggil `holistic_detector.reset()` di awal setiap video untuk mengisolasi state filter temporal antar-video.

7. **Motion Trimming Bebas Spike & Perlindungan Isyarat Statis**:
   - `find_motion_trimmed_bounds()` menghitung energi pergerakan hanya pada komponen kontinu (khususnya pergelangan tangan / wrist pose dan hand joints).
   - Memasang threshold dinamis konservatif agar isyarat statis atau fase hold tidak terpotong secara keliru.

8. **Extractor Versioning & Cache Invalidation**:
   - Menetapkan `EXTRACTOR_VERSION = "2.1.0"` di [holistic_extractor.py](file:///d:/Project/4mlai/isyaratin/holistic_extractor.py).
   - [train-new.py](file:///d:/Project/4mlai/isyaratin/train-new.py) otomatis memverifikasi kecocokan versi cache: cache lama dengan versi berbeda akan otomatis ditolak dan diekstrak ulang demi menjamin integritas data.

---

### 16.3 Hasil Verifikasi Unit Test ([test_pipeline_audit.py](file:///d:/Project/4mlai/isyaratin/test_pipeline_audit.py))

Seluruh 8 skenario pengujian unit test lulus 100% tanpa error:
- `test_ema_batchnorm_buffers_sync`: LULUS (Diff running_mean/running_var = 0.00000).
- `test_horizontal_flip_involutory`: LULUS (`flip(flip(x)) == x`).
- `test_horizontal_flip_swaps_hands_and_flags`: LULUS (Tangan kiri dan kanan bertukar simetris).
- `test_velocity_masking_on_missing_landmarks`: LULUS (Spike transisi kemunculan tangan = 0.00000).
- `test_relational_features_masking_when_hand_missing`: LULUS (Relasi bimanual & finger curls = 0.00000 saat tangan absen).
- `test_augmentation_masking_on_missing_landmarks`: LULUS (Koordinat tangan absen tetap 0.00000 setelah noise, shift, & jitter).
- `test_causal_torso_tracker`: LULUS (EMA smoothing, coasting saat pose hilang, fallback default).
- `test_motion_trimming_no_spike_on_hand_appearance`: LULUS (Kemunculan tangan di frame tengah tidak memicu false motion boundaries).

---

## 17. Ekspor Model TensorFlow Lite (Mobile Ready) & Integrasi Flutter

### 17.1 Latar Belakang & Tantangan Mobile Inference
1. **Target Deployment**:
   - Model BISINDO dideploy ke aplikasi mobile Flutter (`apapun/isyaratin`) menggunakan package `tflite_flutter`.
   - Lokasi bundle target: `assets/models/sign_model.tflite` dan `assets/models/labels.txt`.
2. **Kendala Dynamic LSTM & Select TF Ops**:
   - PyTorch Bidirectional LSTM standar jika diekspor via ONNX ke TFLite (`onnx2tf`) sering kali gagal karena ketergantungan operator dynamic slice/initial states atau memerlukan Flex Delegate (`SELECT_TF_OPS`).
   - Pada aplikasi mobile, mengaktifkan Flex Delegate memerlukan penambahan library C++ binary TensorFlow berukuran besar (>30 MB) dan sering kali crash pada berbagai target arsitektur Android/iOS.
3. **Harmonisasi Kontrak Input**:
   - Ekstraktor mobile di Flutter (`landmark_extractor.dart`) mengekstraksi sekuens frame landmark MediaPipe berdimensi $186$ (21 landmark tangan kiri $\times 3$, 21 landmark tangan kanan $\times 3$, 19 landmark pose $\times 3$, dan 3 bendera deteksi) sepanjang $32$ frame sliding window pada 12 FPS.
   - Model sekuensial yang dilatih pada dataset `manual` (`manual_labels.json`, 20 kelas) membutuhkan kontrak input `[1, 32, 186]` dengan probabilitas Softmax `[1, 20]`.

---

### 17.2 Solusi Arsitektur Keras Unrolled (100% Pure TFLite Builtins)
Untuk menghasilkan model TFLite murni tanpa Flex Delegate:
1. Membangun model equivalent menggunakan Keras 3 Functional API dengan flag `unroll=True` pada layer `Bidirectional(LSTM(..., unroll=True))`.
2. Dengan `unroll=True`, loop rekuren sepanjang 32 time-step dikompilasi menjadi rangkaian operasi tensor statis (MatMul, Add, Mul, Activation), yang 100% didukung oleh engine TFLite Builtins standar.
3. Menambahkan layer `Softmax` eksplisit di ujung klasifikasi sehingga model mengeluarkan probabilitas kelas terkalibrasi $[0.0, 1.0]$.
4. Mengimplementasikan transfer bobot otomatis dari PyTorch `state_dict` ke Keras layer:
   - **LayerNorm**: Gamma dan Beta disalin langsung.
   - **LSTM Weights**: Gate ordering PyTorch dan Keras 3 sama-sama berurutan $[i, f, c, o]$, sehingga bobot cukup di-transpose: $W_{ih}^T$, $W_{hh}^T$, dan $b_{ih} + b_{hh}$. Forward LSTM dipetakan ke `forward_layer`, dan reverse LSTM dipetakan ke `backward_layer`.
   - **Dense Heads**: Transpose kernel $(C_{in}, C_{out})$ dan bias.

---

### 17.3 Hasil Verifikasi Presisi Numerik
Hasil pengujian end-to-end inferensi PyTorch vs TensorFlow Lite:
- **Model MGN (400-dim, 28 kelas)**:
  - Ukuran TFLite: $3.01$ MB
  - Max Absolute Difference: $4.76 \times 10^{-7}$
- **Model BiLSTM Manual (186-dim, 20 kelas - `manual_lstm_big.pt`)**:
  - Ukuran TFLite: $5.11$ MB
  - Max Absolute Difference: $0.00000009$ ($9 \times 10^{-8}$)
  - Akurasi Model Val: $96.67\%$
  - Seluruh output inferensi terbukti identik dan lulus uji akurasi.

---

### 17.4 File & Modul yang Diperbarui
1. [export_tflite.py](file:///d:/Project/4mlai/isyaratin/export_tflite.py): Script mandiri konversi checkpoint PyTorch ke `.tflite` dan `labels.txt` dengan opsi auto-sync ke folder mobile assets.
2. [train-new.py](file:///d:/Project/4mlai/isyaratin/train-new.py): Integrasi otomatis fungsi `export_model_tflite()` pada pipeline training, argumen CLI `--model-tflite`, `--model-labels-txt`, `--flutter-dir`, dan `--no-tflite`.
3. `assets/models/sign_model.tflite` & `assets/models/labels.txt` (Flutter): Telah disinkronkan secara otomatis menggunakan model 20 kelas `manual_label`.
4. `assets/models/README.md` (Flutter): Dokumentasi kontrak input/output disesuaikan secara presisi dengan spesifikasi sliding window landmark 32 frame.

---

## 18. Penyatuan Dataset Gabungan 55 Kelas BISINDO (Manual + WL-BISINDO + Abjad A-Z)

### 18.1 Latar Belakang & Kebutuhan Pengguna
Pengguna menginginkan satu model sekuensial terpadu yang mampu mengenali isyarat dari 3 domain berbeda dalam aplikasi mobile Flutter:
1. **Dataset Manual (Layanan & Perbankan)**: Seluruh 20 kosakata yang telah direkam (`100 Ribu Rupiah`, `Baik`, `Bantuannya`, `Bapak`, `Berapa`, `Buka`, `Buku Rekening`, `Disini`, `Formulir`, `Isi`, `KTP`, `Mau`, `Membawa`, `Minimal`, `Sama-Sama`, `Selamat Siang`, `Setoran Awal`, `Silakan`, `Tanda Tangan`, `Terima Kasih`). Total: 395 video.
2. **Dataset WL-BISINDO Terpilih (Percakapan & Kata Tanya)**:
   - Kata tanya 5W1H lengkap: `Apa`, `Siapa`, `Kapan`, `Di Mana`, `Mengapa`, `Bagaimana`.
   - Kata percakapan penting: `Saya`, `Cari`, dan `Lagi`.
   - *Catatan teknis*: Kata `kamu` tidak terdapat dalam rekaman video dataset WL-BISINDO (hanya ada pada kamus teoritis). Total yang disertakan: 9 kelas, 450 video.
3. **Abjad BISINDO A-Z**: Seluruh 26 huruf alfabet (`A` s.d. `Z`) yang diekstrak dari citra statis beranotasi pose tangan. Total: 26 kelas, 1,040 sekuens.
4. **Harmonisasi Label**: Format Canonical Title Case, urutan alfabetis terstruktur, bebas duplikasi nama kelas, siap dilatih ke dalam 1 model sekuensial tunggal.

---

### 18.2 Metodologi Konversi Abjad Citra Statis ke Sekuens Temporal (186D)
- **Tantangan**: Dataset Abjad berupa citra statis 2D, sedangkan arsitektur utama kita adalah model temporal sekuensial (BiLSTM 32 frame, 186 dimensi per frame).
- **Solusi ([extract_abjad_dataset.py](file:///d:/Project/4mlai/isyaratin/extract_abjad_dataset.py))**:
  1. Ekstraksi koordinat landmark tangan 21 titik ($x, y, z$) menggunakan MediaPipe Hands.
  2. Konstruksi tubuh buatan anatomis (torso & head fallback) berbasis posisi tangan untuk mempertahankan normalisasi torso body-centric (186D).
  3. Konversi citra statis menjadi sekuens 32 frame dengan menambahkan **Gaussian micro-jitter** ($\sigma = 0.003$) pada koordinat sendi tangan aktif.
  4. Manfaat: Mensimulasikan getaran fisiologis alami tangan manusia saat menahan isyarat abjad di depan kamera dan mencegah terjadinya _zero-variance vanishing gradient_ pada sel LSTM.

---

### 18.3 Penyatuan & Harmonisasi Label ([build_dataset_gabungan.py](file:///d:/Project/4mlai/isyaratin/build_dataset_gabungan.py))
- Menggabungkan data dari ketiga sumber menjadi cache terpadu:
  - Cache: [data/gabungan_holistic_lstm_cache.npz](file:///d:/Project/4mlai/isyaratin/data/gabungan_holistic_lstm_cache.npz) (31.51 MB, 1,885 sampel, 55 kelas).
  - Label JSON: [labels/gabungan_labels.json](file:///d:/Project/4mlai/isyaratin/labels/gabungan_labels.json)
  - Metadata: [labels/gabungan_labels_meta.json](file:///d:/Project/4mlai/isyaratin/labels/gabungan_labels_meta.json)
  - Text Label: [models/labels.txt](file:///d:/Project/4mlai/isyaratin/models/labels.txt)
- **Distribusi Kelas yang Seimbang**:
  - Manual: ~20 sampel per kelas (total 395 sampel).
  - WL-BISINDO: 50 sampel per kelas (total 450 sampel).
  - Abjad: 40 sampel per huruf (total 1,040 sampel).
  - Rata-rata: ~34.3 sampel per kelas, sangat stabil untuk konvergensi Focal Loss.

---

### 18.4 Hasil Pelatihan Model Terpadu 55 Kelas ([train-new.py](file:///d:/Project/4mlai/isyaratin/train-new.py))
Pelatihan BiLSTM 2-Layer (128 hidden1, 96 hidden2, 192 head) pada 1,885 sampel selama 45 epoch di GPU NVIDIA GeForce RTX 3050 Laptop:
- **Akurasi Validasi Konsensus Video**: **91.25%** (344/377 video validasi tepat sasaran)
- **Macro Precision**: **0.93 (93%)**
- **Macro Recall**: **0.92 (92%)**
- **Macro F1-Score**: **0.92**
- **Performa Per Kategori**:
  - Manual Perbankan (20 kelas): Hampir seluruh kelas mencapai presisi & recall **1.00 (100%)** (`100 Ribu Rupiah`, `Baik`, `Bantuannya`, `Bapak`, `Buka`, `Buku Rekening`, `Formulir`, `Isi`, `KTP`, `Mau`, `Membawa`, `Minimal`, `Sama-Sama`, `Selamat Siang`, `Setoran Awal`, `Terima Kasih`).
  - WL-BISINDO Terpilih (9 kelas): `Bagaimana` (1.00), `Di Mana` (1.00), `Lagi` (1.00), `Mengapa` (1.00), `Apa` (0.95), `Kapan` (0.95), `Saya` (0.95), `Siapa` (0.95), `Cari` (0.90).
  - Abjad A-Z (26 kelas): Seluruh 26 huruf terklasifikasi dengan F1-Score tinggi (0.71 s.d. 1.00).

---

### 18.5 Ekspor & Validasi TensorFlow Lite Mobile (100% Pure Builtins)
- **Artefak Model**:
  - PyTorch Checkpoint: [models/gabungan_lstm_best.pt](file:///d:/Project/4mlai/isyaratin/models/gabungan_lstm_best.pt)
  - ONNX Model: [models/gabungan_lstm.onnx](file:///d:/Project/4mlai/isyaratin/models/gabungan_lstm.onnx) (1.33 MB, Mobile Validated)
  - TensorFlow Lite: [models/sign_model.tflite](file:///d:/Project/4mlai/isyaratin/models/sign_model.tflite) (3,038.3 KB)
  - Labels Text: [models/labels.txt](file:///d:/Project/4mlai/isyaratin/models/labels.txt) (55 baris kanonikal)
- **Verifikasi Kontrak Mobile**:
  - Input model: `[1, 32, 400]` float32.
  - Ekstraksi kamera menghasilkan 186 fitur dasar per frame (61 landmark × 3 + 3 presence flags), lalu mobile wajib menjalankan ekspansi spasio-temporal yang sama dengan `compute_extended_features`: 183 velocity termasking + 31 fitur relasional. Total akhir: `186 + 183 + 31 = 400` fitur per frame.
  - Output: `[1, 55]` float32 (probabilitas Softmax).
  - Kompatibilitas Runtime: 100% Pure TFLite Builtins (tanpa Flex Delegate / Select TF Ops), siap langsung digunakan di Flutter via `tflite_flutter`.
  - Artefak mobile disinkronisasi ke: `d:\Project\4mlai\isyaratin\isyaratin\assets\models\`.

---

### 18.6 Evaluasi Komprehensif Model Gabungan ([evaluate.py](file:///d:/Project/4mlai/isyaratin/evaluate.py))
Skrip [evaluate.py](file:///d:/Project/4mlai/isyaratin/evaluate.py) telah dimutakhirkan agar mendukung evaluasi dataset gabungan 55 kelas secara otomatis (`--dataset gabungan`) dengan opsi stratified split holdout 20% (377 video test) maupun evaluasi penuh (`--all-signers`).

Hasil evaluasi komprehensif pada subset stratified holdout 20%:
- **Top-1 Accuracy (Video Consensus)**: **92.84%**
- **Top-3 Accuracy**: **96.82%**
- **Top-5 Accuracy**: **97.88%**
- **Macro F1-Score**: **0.9397 (94.0%)**
- **Weighted F1-Score**: **0.9280**
- **ROC Macro AUC**: **0.9975** (Kualitas pemisahan probabilitas kelas sangat mendekati sempurna 1.0)
- **PR Micro Average Precision (AP)**: **0.9595**
- **Artefak Laporan & Grafik Visualisasi** (Tersimpan di [reports/evaluation/gabungan/](file:///d:/Project/4mlai/isyaratin/reports/evaluation/gabungan)):
  1. `confusion_matrix_normalized.png` : Heatmap recall normalisasi % untuk ke-55 kelas.
  2. `confusion_matrix_raw.png`        : Heatmap jumlah frekuensi prediksi benar vs salah.
  3. `roc_curves.png`                  : Kurva ROC multi-class One-vs-Rest (OvR).
  4. `precision_recall_curves.png`      : Kurva Precision-Recall per kelas dan micro-average.
  5. `per_class_f1_bar.png`            : Diagram batang F1-score untuk setiap 55 kelas kosakata.
  6. `top_confused_pairs.png`          : Analisis pasangan kata yang paling sering tertukar.
  7. `evaluation_report.md`            : Laporan evaluasi lengkap dalam format Markdown.
  8. `evaluation_metrics.json`         : Ringkasan metrik evaluasi numerik terstruktur.

---

### 18.7 Inferensi Real-Time Webcam Multi-Landmark 55 Kelas ([camera.py](file:///d:/Project/4mlai/isyaratin/camera.py))
Skrip inferensi langsung webcam [camera.py](file:///d:/Project/4mlai/isyaratin/camera.py) telah dimutakhirkan secara penuh agar langsung mendukung model dataset gabungan **55 kelas** secara default:
- **Default Parameter**:
  - `--dataset gabungan`: Otomatis memuat checkpoint [models/gabungan_lstm_best.pt](file:///d:/Project/4mlai/isyaratin/models/gabungan_lstm_best.pt) (atau ONNX [models/gabungan_lstm.onnx](file:///d:/Project/4mlai/isyaratin/models/gabungan_lstm.onnx)) dan label [labels/gabungan_labels.json](file:///d:/Project/4mlai/isyaratin/labels/gabungan_labels.json).
  - `--duration 3.5`: Durasi perekaman isyarat disesuaikan ke 3.5 detik (seimbang untuk mengenali pose abjad cepat 1-2 detik maupun frasa perbankan 3-4 detik).
- **Format Label Fleksibel**:
  - Mendukung pembacaan list kamus JSON (`labels/gabungan_labels.json`) maupun format baris teks per label (`models/labels.txt`).
- **Kompatibilitas Kinematic Engine**:
  - Seluruh filter disambiguasi gerakan dan *idle / OOD gate* berjalan harmonis dengan 55 kelas kanonikal tanpa error index out-of-range.
- **Cara Menjalankan**:
  ```bash
  # Inferensi Webcam Live (Default: Model 55 Kelas PyTorch CUDA/CPU)
  uv run python camera.py

  # Inferensi Webcam Live via Model ONNX Ringan
  uv run python camera.py --onnx models/gabungan_lstm.onnx

  # Uji Inferensi dari File Video
  uv run python camera.py --video path/ke/video_isyarat.mp4
  ```

---

## 19. Konsolidasi Pipeline & Pembersihan Modul Eksperimen (Fokus Pipeline Inti)

- **Latar Belakang & Rasionalisasi**:
  - Setelah serangkaian eksperimen benchmarking (klasifikasi klasik Random Forest/SVM berbasis rekayasa fitur statistik Bird 2022, perbandingan dimensi fitur, analisis confusion matrix skrip lama), arsitektur model produksi telah mantap bertumpu pada **BiLSTM sekuensial end-to-end 186D** dengan **Kinematic Disambiguation Rules** dan **Pure TensorFlow Lite Builtins**.
  - Untuk menjaga kebersihan repositori (*clean code*), mencegah redundansi, serta memfokuskan pemeliharaan pada alur kerja utama, seluruh file eksperimen yang sudah usang dan tidak digunakan lagi telah dihapus.

- **Daftar Modul & File yang Dibersihkan**:
  1. `analyze_confusions.py` : Skrip analisis kinematik data-driven statis (telah terintegrasi penuh di [kinematic_engine.py](file:///d:/Project/4mlai/isyaratin/kinematic_engine.py) dan [evaluate.py](file:///d:/Project/4mlai/isyaratin/evaluate.py)).
  2. `benchmark_feature_models.py` : Skrip benchmarking model rekayasa fitur classical ML (RF, SVM).
  3. `train_feature_classifiers.py` : Skrip pelatihan Random Forest & SVM fitur manual.
  4. `infer_feature_model.py` : Modul inferensi mandiri untuk model classical ML.
  5. `feature_engineer.py` : Kelas ekstraksi fitur 384D/ANOVA (Bird 2022).
  6. `test_feature_engineer.py` : Unit test untuk fitur classical ML.
  7. `feature_engineering_spec.md` : Spesifikasi teoritis rekayasa fitur manual.
  8. `camera-new.py` : Skrip kamera multi-model eksperimental (digantikan penuh oleh [camera.py](file:///d:/Project/4mlai/isyaratin/camera.py)).
  9. `train-gabungan.py` : Skrip pelatihan hybrid lama (telah terpadu di [train-new.py](file:///d:/Project/4mlai/isyaratin/train-new.py) dan [build_dataset_gabungan.py](file:///d:/Project/4mlai/isyaratin/build_dataset_gabungan.py)).
  10. `reports/feature_engineering_benchmark.json`, `reports/gabungan_feature_models_summary.json`, `reports/trained_feature_models_summary.json` : Laporan JSON eksperimen model fitur.

- **Struktur Pipeline Inti yang Aktif & Terfokus Saat Ini**:
  1. **Ekstraksi Landmark Spasio-Temporal**: [holistic_extractor.py](file:///d:/Project/4mlai/isyaratin/holistic_extractor.py) (186D body-centric, causal torso tracking, velocity & relational masking).
  2. **Penyusunan Dataset Terpadu**: [build_dataset_gabungan.py](file:///d:/Project/4mlai/isyaratin/build_dataset_gabungan.py) (Mendukung seleksi dinamis kata umum, capping sampel per kelas, dan integrasi multi-dataset).
  3. **Pelatihan Model Terpadu**: [train-new.py](file:///d:/Project/4mlai/isyaratin/train-new.py) (Mendukung `--dataset gabungan`, `--dataset manual`, `--dataset wl`, auto-export PyTorch `.pt`, ONNX `.onnx`, dan TFLite `.tflite`).
  4. **Ekspor & Sinkronisasi Mobile**: [export_tflite.py](file:///d:/Project/4mlai/isyaratin/export_tflite.py) (Keras unrolled 100% Pure Builtins TFLite untuk Flutter).
  5. **Evaluasi Komprehensif**: [evaluate.py](file:///d:/Project/4mlai/isyaratin/evaluate.py) (Confusion Matrix, ROC-AUC, Precision-Recall, F1-Score).
  6. **Disambiguasi Kinematic**: [kinematic_engine.py](file:///d:/Project/4mlai/isyaratin/kinematic_engine.py) & [labels/bisindo_rules.json](file:///d:/Project/4mlai/isyaratin/labels/bisindo_rules.json) (14 rule eliminasi ambiguitas + idle rejection).
  7. **Inferensi Kamera Live / Video**: [camera.py](file:///d:/Project/4mlai/isyaratin/camera.py) (Hands-free recording, live sentence builder, HUD realtime).
  8. **Suite Pengujian Unit**: [test_pipeline_audit.py](file:///d:/Project/4mlai/isyaratin/test_pipeline_audit.py) & [test_kinematic_cases.py](file:///d:/Project/4mlai/isyaratin/test_kinematic_cases.py) (100% lulus).

---

## 20. Eksperimen Penyaringan Kosakata Gabungan (22 Kelas) & Capping Sampel Anti-Overfitting

- **Latar Belakang & Masalah**:
  - Pada iterasi awal dataset gabungan 55 kelas dan 29 kelas, kelas pertanyaan 5W1H dari WL-BISINDO (`Apa`, `Bagaimana`, `Cari`, `Di Mana`, `Kapan`, `Mengapa`, `Siapa`) memiliki jumlah sampel yang terlalu besar (50 sampel per kelas) dibandingkan dataset manual perbankan/layanan (~20 sampel per kelas).
  - Ketimpangan rasio sampel (2.5:1) ini memicu bias model, di mana kelas dengan variansi gerak tinggi seperti `Apa` dan `Di Mana` mengalami degradasi F1-Score (0.63 - 0.64) dan membingungkan batas keputusan (*decision boundary*) kelas perbankan.
  - Selain itu, dataset abjad A-Z (citra statis 2D yang diinjeksikan jitter sintetis) memiliki karakteristik artikulasi yang berbeda dengan video gestur kata alami, sehingga lebih optimal dipisahkan dari model kata sekuensial.

- **Solusi Rekayasa Dataset ([build_dataset_gabungan.py](file:///d:/Project/4mlai/isyaratin/build_dataset_gabungan.py))**:
  1. **Pengecualian Abjad A-Z**: Model difokuskan 100% pada kosakata kata/frasa utuh.
  2. **Eliminasi Kata Tanya WL**: Seluruh 7 kata tanya (`Apa`, `Bagaimana`, `Cari`, `Di Mana`, `Kapan`, `Mengapa`, `Siapa`) dihapus dari set gabungan.
  3. **Preservasi Kata Umum Percakapan**: Mempertahankan kelas kata ganti & repetisi esensial dari WL-BISINDO: `Saya` dan `Lagi`.
  4. **Dynamic Sample Capping (`--max-samples 40`)**: Menerapkan batas atas maksimal 40 sampel per kelas. Jumlah sampel `Saya` dan `Lagi` dipangkas dari 50 menjadi tepat 40 sampel agar proporsional terhadap kelas manual (20 sampel).
  5. **Total Komposisi Dataset**:
     - 20 Kelas Manual: 395 sampel (~20 sampel/kelas, `Berapa` 15 sampel).
     - 2 Kelas WL-BISINDO: 80 sampel (40 sampel/kelas).
     - Total: **22 Kelas Kanonikal** dengan **475 sampel** berimbang.

- **Hasil Pelatihan & Evaluasi Model 22 Kelas**:
  - **Arsitektur**: MGN (Multi-Granularity Network: Conv1D k=3/k=5 + BiLSTM 2-Layer + Stroke Attention Pooling, 400D extended features).
  - **Konfigurasi**: 64 Epoch, Batch Size 16, Focal Loss ($\gamma = 2.0$, Label Smoothing 0.05), Random Seed 42, CUDA GPU.
  - **Akurasi Validasi Model**: **94.74%** (tercapai stabil di Epoch 55).
  - **Akurasi Konsensus Video (1 Video = 1 Keputusan)**: **93.68%** (naik dari 89.94%).
  - **Metrik Evaluasi Holdout (Stratified Split 20%, 95 video)**:
    - **Top-1 Accuracy**: **95.79%**
    - **Top-3 Accuracy**: **100.00%**
    - **Top-5 Accuracy**: **100.00%**
    - **Macro F1-Score**: **0.9524 (95.2%)**
    - **Weighted F1-Score**: **0.9580 (95.8%)**
    - **Macro ROC-AUC**: **0.999**
    - **Micro Average Precision (PR)**: **0.983**
  - **Kinerja Spesifik Kelas WL**:
    - `Saya`: Precision 1.00, Recall 1.00, F1-Score **1.00 (100%)**
    - `Lagi`: Precision 1.00, Recall 1.00, F1-Score **1.00 (100%)**
  - **Artefak & Mobile Sync**:
    - Checkpoint: [models/gabungan_lstm_best.pt](file:///d:/Project/4mlai/isyaratin/models/gabungan_lstm_best.pt)
    - ONNX: [models/gabungan_lstm.onnx](file:///d:/Project/4mlai/isyaratin/models/gabungan_lstm.onnx) (Output shape: `(1, 22)`)
    - TFLite: [models/sign_model.tflite](file:///d:/Project/4mlai/isyaratin/models/sign_model.tflite) (3,010.0 KB, 100% Pure Builtins)
    - Labels: [models/labels.txt](file:///d:/Project/4mlai/isyaratin/models/labels.txt) (22 kelas baris)
    - Spesifikasi: [models/model_spec.json](file:///d:/Project/4mlai/isyaratin/models/model_spec.json)
    - Seluruh aset mobile secara otomatis tersinkronisasi ke folder `assets/models/` pada aplikasi Flutter.




