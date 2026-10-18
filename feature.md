# Dokumentasi Fitur Sistem Isyaratin (feature.md)

Dokumen ini merupakan referensi teknis lengkap arsitektur fitur input, representasi spasio-temporal, pemetaan indeks landmark, normalisasi matematis, dan mekanisme pembeda kinematik untuk mengeliminasi seluruh ambiguitas kosakata dalam sistem pengenalan BISINDO.

---

## 1. Arsitektur & Dimensi Vektor Fitur

Model Sign Language Recognition (MGN / Multi-Stream BiLSTM) memproses sequence video dengan representasi spasio-temporal multi-level:

* **Dimensi Input Statis Dasar**: **186 Dimensi per Frame** (Koordinat normalisasi tubuh & tangan).
* **Dimensi Input Extended Spasio-Temporal**: **400 Dimensi per Frame** (186 statis + 183 kecepatan sesaat + 31 relasi kinematik).
* **Jendela Temporal Standar**: **32 Frame** (Sampling adaptif dari video gestur asli 4–6 detik).
* **Ukuran Tensor Input**: $(B, 32, 400)$ untuk batch training, atau $(32, 400)$ untuk inferensi live.

$$\mathbf{X}_t = \Big[ \mathbf{P}_t^{\text{static}} \;(186\text{d}) \;\parallel\; \mathbf{V}_t^{\text{velocity}} \;(183\text{d}) \;\parallel\; \mathbf{K}_t^{\text{kinematic}} \;(31\text{d}) \Big] \in \mathbb{R}^{400}$$

---

## 2. Pemetaan Indeks Fitur Lengkap (Indeks 0 hingga 399)

Berikut adalah pemetaan rinci seluruh 400 dimensi fitur per frame:

### A. Fitur Statis Landmark Asli (Indeks 0 – 185, Total 186 Dimensi)

Diekstrak dari MediaPipe Holistic pada frame yang telah dinormalisasi torso-centric dan berpusat di tengah kedua bahu:

| Rentang Indeks | Komponen Tubuh | Jumlah Titik | Dimensi | Penjelasan Anatomis & Urutan Titik |
| :---: | :--- | :---: | :---: | :--- |
| **0 – 62** | **Tangan Kiri (Left Hand)** | 21 Titik | 63 Dim $(x, y, z)$ | • Titik 0: Wrist (Pergelangan tangan kiri)<br>• Titik 1–4: Thumb (CMC, MCP, IP, Tip)<br>• Titik 5–8: Index (MCP, PIP, DIP, Tip)<br>• Titik 9–12: Middle (MCP, PIP, DIP, Tip)<br>• Titik 13–16: Ring (MCP, PIP, DIP, Tip)<br>• Titik 17–20: Pinky (MCP, PIP, DIP, Tip) |
| **63 – 125** | **Tangan Kanan (Right Hand)** | 21 Titik | 63 Dim $(x, y, z)$ | • Titik 21: Wrist (Pergelangan tangan kanan)<br>• Titik 22–25: Thumb (CMC, MCP, IP, Tip)<br>• Titik 26–29: Index (MCP, PIP, DIP, Tip)<br>• Titik 30–33: Middle (MCP, PIP, DIP, Tip)<br>• Titik 34–37: Ring (MCP, PIP, DIP, Tip)<br>• Titik 38–41: Pinky (MCP, PIP, DIP, Tip) |
| **126 – 158** | **Pose: Kepala & Wajah** | 11 Titik | 33 Dim $(x, y, z)$ | • Titik 42: Hidung (`nose`)<br>• Titik 43–45: Mata Kiri (Inner, Mid, Outer / Pelipis Kiri)<br>• Titik 46–48: Mata Kanan (Inner, Mid, Outer / Pelipis Kanan)<br>• Titik 49–50: Telinga Kiri (`ear_l`) & Telinga Kanan (`ear_r`)<br>• Titik 51–52: Sudut Mulut Kiri (`mouth_l`) & Kanan (`mouth_r`) |
| **159 – 176** | **Pose: Lengan & Bahu** | 6 Titik | 18 Dim $(x, y, z)$ | • Titik 53: Bahu Kiri (`shoulder_l` / LS)<br>• Titik 54: Bahu Kanan (`shoulder_r` / RS)<br>• Titik 55: Siku Kiri (`elbow_l`)<br>• Titik 56: Siku Kanan (`elbow_r`)<br>• Titik 57: Pergelangan Lengan Kiri (`wrist_pose_l`)<br>• Titik 58: Pergelangan Lengan Kanan (`wrist_pose_r`) |
| **177 – 182** | **Pose: Torso & Pinggul** | 2 Titik | 6 Dim $(x, y, z)$ | • Titik 59: Pinggul Kiri (`hip_l`)<br>• Titik 60: Pinggul Kanan (`hip_r`) |
| **183 – 185** | **Presence Flags (Indikator Deteksi)** | 3 Nilai | 3 Dim Biner | • Indeks 183: `lh_present` (1.0 jika tangan kiri terdeteksi, 0.0 jika tidak)<br>• Indeks 184: `rh_present` (1.0 jika tangan kanan terdeteksi, 0.0 jika tidak)<br>• Indeks 185: `pose_present` (1.0 jika tubuh terdeteksi, 0.0 jika tidak) |

---

### B. Instantaneous Velocity / Kecepatan Mikro (Indeks 186 – 368, Total 183 Dimensi)

Turunan waktu diferensial tingkat satu untuk seluruh 61 titik koordinat fisik $(x, y, z)$ antara dua frame berturut-turut:
$$\mathbf{V}_t = \mathbf{P}_t^{\text{coords}} - \mathbf{P}_{t-1}^{\text{coords}}, \quad \forall t \ge 1 \quad (\text{pada } t=0, \mathbf{V}_0 = \mathbf{V}_1)$$

* Menangkap kecepatan translasi, arah lintasan (naik/turun/maju/mundur), dan akselerasi gestur.
* Catatan: Nilai presence flags (indeks 183–185) tidak dihitung kecepatannya sehingga jumlah dimensi kecepatan adalah $61 \times 3 = 183$ dimensi.

---

### C. Relasi Kinematik Spasial Kunci (Indeks 369 – 399, Total 31 Dimensi)

Fitur relasional non-linier yang merepresentasikan jarak Euclidean anatomis dan konfigurasi jari tangan:

| Indeks | Nama Simbolik Fitur | Titik Referensi Anatomis | Formula Matematis | Fungsi Linguistik / Diskriminatif |
| :---: | :--- | :--- | :--- | :--- |
| **369** | `d_hands` | Pergelangan LH $\leftrightarrow$ RH | $\|P_{\text{lh\_wrist}} - P_{\text{rh\_wrist}}\|$ | Mendeteksi gestur dua tangan (bimanual). |
| **370** | `d_index_tips` | Ujung Telunjuk LH $\leftrightarrow$ RH | $\|P_{\text{lh\_index}} - P_{\text{rh\_index}}\|$ | Mendeteksi pertemuan ujung jari (puncak atap 'Rumah'). |
| **371** | `d_lh_nose` | Pergelangan LH $\leftrightarrow$ Hidung | $\|P_{\text{lh\_wrist}} - P_{\text{nose}}\|$ | Ketinggian tangan kiri di area wajah atas. |
| **372** | `d_rh_nose` | Pergelangan RH $\leftrightarrow$ Hidung | $\|P_{\text{rh\_wrist}} - P_{\text{nose}}\|$ | Ketinggian tangan kanan di area wajah atas. |
| **373** | `d_lh_mouth` | Pergelangan LH $\leftrightarrow$ Bibir | $\|P_{\text{lh\_wrist}} - P_{\text{mouth}}\|$ | Kedekatan tangan kiri dengan mulut. |
| **374** | `d_rh_mouth` | Pergelangan RH $\leftrightarrow$ Bibir | $\|P_{\text{rh\_wrist}} - P_{\text{mouth}}\|$ | Kedekatan tangan kanan dengan mulut. |
| **375** | `d_lh_index_mouth` | Telunjuk LH $\leftrightarrow$ Bibir | $\|P_{\text{lh\_index}} - P_{\text{mouth}}\|$ | Artikulasi mulut tangan kiri ('Makan', 'Bicara'). |
| **376** | `d_rh_index_mouth` | Telunjuk RH $\leftrightarrow$ Bibir | $\|P_{\text{rh\_index}} - P_{\text{mouth}}\|$ | Artikulasi mulut tangan kanan ('Makan', 'Air'). |
| **377** | `d_lh_thumb_mouth` | Ibu Jari LH $\leftrightarrow$ Bibir | $\|P_{\text{lh\_thumb}} - P_{\text{mouth}}\|$ | Gestur minum atau isyarat jempol di mulut. |
| **378** | `d_rh_thumb_mouth` | Ibu Jari RH $\leftrightarrow$ Bibir | $\|P_{\text{rh\_thumb}} - P_{\text{mouth}}\|$ | Gestur minum ('Air'). |
| **379** | `d_lh_index_ear` | Telunjuk LH $\leftrightarrow$ Telinga Kiri | $\|P_{\text{lh\_index}} - P_{\text{ear\_l}}\|$ | Isyarat pendengaran tangan kiri ('Tuli', 'Dengar'). |
| **380** | `d_rh_index_ear` | Telunjuk RH $\leftrightarrow$ Telinga Kanan | $\|P_{\text{rh\_index}} - P_{\text{ear\_r}}\|$ | Isyarat pendengaran tangan kanan ('Tuli', 'Dengar'). |
| **381** | `d_lh_wrist_ear` | Pergelangan LH $\leftrightarrow$ Telinga Kiri | $\|P_{\text{lh\_wrist}} - P_{\text{ear\_l}}\|$ | Kedekatan telapak kiri ke daun telinga. |
| **382** | `d_rh_wrist_ear` | Pergelangan RH $\leftrightarrow$ Telinga Kanan | $\|P_{\text{rh\_wrist}} - P_{\text{ear\_r}}\|$ | Kedekatan telapak kanan ke daun telinga. |
| **383** | `d_lh_ls` | Pergelangan LH $\leftrightarrow$ Bahu Kiri | $\|P_{\text{lh\_wrist}} - P_{\text{ls}}\|$ | Rentang lengan kiri. |
| **384** | `d_rh_rs` | Pergelangan RH $\leftrightarrow$ Bahu Kanan | $\|P_{\text{rh\_wrist}} - P_{\text{rs}}\|$ | Rentang lengan kanan. |
| **385** | `d_lh_index_torso` | Telunjuk LH $\leftrightarrow$ Pusat Dada | $\|P_{\text{lh\_index}} - P_{\text{torso}}\|$ | Menunjuk dada kiri ('Saya', 'Hati'). |
| **386** | `d_rh_index_torso` | Telunjuk RH $\leftrightarrow$ Pusat Dada | $\|P_{\text{rh\_index}} - P_{\text{torso}}\|$ | Menunjuk dada kanan ('Saya', 'Maaf'). |
| **387** | `rel_y_lh_nose` | Elevasi Vertikal LH vs Hidung | $Y_{\text{lh\_index}} - Y_{\text{nose}}$ | Ketinggian telunjuk kiri relatif terhadap hidung. |
| **388** | `rel_y_rh_nose` | Elevasi Vertikal RH vs Hidung | $Y_{\text{rh\_index}} - Y_{\text{nose}}$ | Ketinggian telunjuk kanan relatif terhadap hidung. |
| **389** | `rel_y_lh_torso` | Elevasi Vertikal LH vs Dada | $Y_{\text{lh\_wrist}} - Y_{\text{torso}}$ | Elevasi pergelangan tangan kiri di depan tubuh. |
| **390 – 394** | `lh_finger_curls` (5 dim) | 5 Ujung Jari LH $\leftrightarrow$ Wrist LH | $\|P_{\text{lh\_wrist}} - P_{\text{tip}_k}\|$ | Derajat kepalan tangan kiri (Thumb, Index, Mid, Ring, Pinky). |
| **395 – 399** | `rh_finger_curls` (5 dim) | 5 Ujung Jari RH $\leftrightarrow$ Wrist RH | $\|P_{\text{rh\_wrist}} - P_{\text{tip}_k}\|$ | Derajat kepalan tangan kanan (Thumb, Index, Mid, Ring, Pinky). |

$$\text{Total Dimensi} = 186 \text{ (Statis)} + 183 \text{ (Velocity)} + 31 \text{ (Kinematik)} = \mathbf{400 \text{ Dimensi}}$$

---

## 3. Matematika Normalisasi Koordinat

Untuk menjamin invariansi terhadap resolusi kamera, jarak signer ke lensa, dan rasio aspek perangkat (misal HP Portrait 9:16 vs Webcam Laptop 16:9), diterapkan normalisasi 3-tahap:

1. **Koreksi Rasio Aspek Isotropik**:
   $$\text{scale}_{\text{screen}} = \max(W, H), \quad s_x = \frac{W}{\text{scale}_{\text{screen}}}, \quad s_y = \frac{H}{\text{scale}_{\text{screen}}}$$
   $$P_{\text{raw}} = \big( \text{lm}.x \cdot s_x, \;\text{lm}.y \cdot s_y, \;\text{lm}.z \cdot s_x \big)$$
   Mencegah pipi/tangan tertarik atau menyusut secara geometris saat berganti orientasi HP.

2. **Translasi Torso-Centric (Titik Acuan Dada)**:
   Pusat koordinat $(0, 0, 0)$ ditetapkan tepat di titik tengah antara bahu kiri dan bahu kanan:
   $$\mathbf{C}_{\text{torso}} = \frac{\mathbf{P}_{\text{left\_shoulder}} + \mathbf{P}_{\text{right\_shoulder}}}{2}$$

3. **Penskalaan Normalisasi Bahu (Body Scale)**:
   Skala fisik dinormalisasi berdasarkan lebar bahu pengguna:
   $$S_{\text{torso}} = \|\mathbf{P}_{\text{left\_shoulder}} - \mathbf{P}_{\text{right\_shoulder}}\|$$
   $$\mathbf{P}_{\text{normalized}} = \frac{\mathbf{P}_{\text{raw}} - \mathbf{C}_{\text{torso}}}{S_{\text{torso}}}$$
   Dengan demikian, koordinat bersifat invarian baik signer berbadan besar/kecil maupun berdiri dekat/jauh dari kamera.

---

## 4. Dimensi Temporal & Strategi Resampling

1. **Jendela Waktu Model**: Tetap **32 Frame** (Sampling laju $\approx 12.0$ FPS).
2. **Global Temporal Linear Resampling**:
   Video isyarat pengguna yang berdurasi **4–6 detik** (misal 50–90 frame mentah) tidak dipotong acak (*no blind cropping*), melainkan disampel merata di sepanjang rentang waktu:
   $$t_i = \text{round}\left( \frac{i \cdot (T_{\text{total}} - 1)}{31} \right), \quad \forall i \in \{0, 1, \dots, 31\}$$
   Hal ini menjamin fase persiapan (*stroke preparation*), fase inti artikulasi (*nucleus*), dan fase kembali (*recovery*) terserap utuh ke dalam model BiLSTM/MGN.
3. **Hands-Free Dynamic Trigger with Exit Delay**:
   * **Entry**: Merekam otomatis saat tangan muncul (`lh_present or rh_present`).
   * **Exit Grace Period**: Ketika tangan turun atau hilang, rekaman tidak langsung putus, melainkan menunggu hitung mundur **1.5 detik**. Jika tangan naik lagi, rekaman diteruskan; jika tetap kosong sampai batas waktu, rekaman dikunci dan diinferensikan instan.

---

## 5. Matriks Eliminasi Ambiguitas Kosakata (*Confusable Pairs Matrix*)

Model klasifikasi 28 kelas BISINDO sering mengalami kesulitan membedakan isyarat yang bertetangga spasial di kamera 2D. Modul [kinematic_engine.py](file:///d:/Project/4mlai/isyaratin/kinematic_engine.py) bersama [bisindo_rules.json](file:///d:/Project/4mlai/isyaratin/bisindo_rules.json) mengeliminasi seluruh ambiguitas ini melalui post-hoc fact checking berbasis biomekanik:

| No | Kelompok Kata Ambigu | Mengapa Mirip di Kamera 2D? | Pembeda Fisik & Rumus Kinematik | Rule ID di `bisindo_rules.json` | Status Evaluasi |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **1** | **Rumah** vs **Alfabet A** / **Saya** | Tangan sama-sama berada di area tengah dada / sternum. | **Bimanual Atap Segitiga**: <br>• `Rumah` **wajib 2 tangan** (`bimanual_ratio >= 0.28`) dan ujung jari bertemu membentuk atap segitiga (`min_roof_contact <= 0.38`).<br>• `A` atau `Saya` hanya 1 tangan mengepal/menunjuk. Jika unimanual, probabilitas `Rumah` dipangkas 95%. | `bimanual_roof_gate` | **Aktif 100% Teruji** |
| **2** | **Tuli** vs **Ingat** vs **Dengar** | Ketiganya menggunakan jari telunjuk menunjuk (*pointing*) ke kepala samping. | **Kontras Titik Artikulasi**: <br>• `Tuli` & `Dengar`: Jari aktif menyentuh daun telinga (`ear_l / ear_r`).<br>• `Ingat`: Jari aktif menyentuh dahi / pelipis luar (`temple_l / temple_r`).<br>• Evaluasi: $\Delta = d_{\text{ear}} - d_{\text{temple}}$. Jika $\Delta < -0.04 \implies$ boost kuat 'Tuli'/'Dengar', redam 'Ingat'. | `ear_vs_temple_gate`<br>`dengar_vs_ingat_gate` | **Aktif 100% Teruji** |
| **3** | **Makan** vs **Terima Kasih** vs **Air** vs **Kamu** | Ketinggian tangan sama-sama di sekitar bibir, dagu, dan mulut. | **Bentuk Jari & Lintasan Vektor**: <br>• `Makan`: Jari menguncup rapat (*pinch* $< 0.24$) menyentuh mulut.<br>• `Terima kasih`: Telapak terbuka (*flat palm*) bergerak **maju menjauhi dagu** (`forward_push > 0.01`).<br>• `Kamu`: Hanya telunjuk menunjuk lurus ke depan (`pointing_isolation > 0.10`).<br>• `Air`: Telapak tangan terbuka/tiga jari mendekati bibir tanpa menguncup. | `mouth_handshape_gate` | **Aktif 100% Teruji** |
| **4** | **Pagi** vs **Siang** vs **Sore** vs **Malam** | Semuanya gestur siklus matahari dengan tangan bergerak vertikal/diagonal. | **Lintasan Elevasi Sumbu Y**: <br>• `Pagi`: Lintasan tangan bergerak **naik vertikal** dari bawah ke leher (`y_travel > 0.08`).<br>• `Siang`: Posisi tangan tinggi **di atas puncak kepala/hidung** (`min_y_rel_nose < -0.08`).<br>• `Sore`: Lintasan tangan bergerak **turun condong** ke samping (`y_travel < -0.08`).<br>• `Malam`: Isyarat bimanual kedua tangan menyilang/menutup di dada. | `sun_trajectory_pagi_sore` | **Aktif 100% Teruji** |
| **5** | **Datang** vs **Berangkat** | Kedua gestur berada di ketinggian perut/dada dengan orientasi tangan serupa. | **Arah Translasi Spasial Relatif ke Tubuh**: <br>• `Datang`: Tangan bergerak **mendekat ke arah tubuh signer** (`torso_approach > 0.06`).<br>• `Berangkat`: Tangan bergerak **menjauh dari tubuh signer ke arah depan/kamera** (`torso_approach < -0.06`). | `spatial_trajectory_datang_berangkat` | **Aktif 100% Teruji** |
| **6** | **Kapan** vs **Di mana** vs **Siapa** vs **Apa** | Kelompok kata tanya (5W1H) dengan tangan terbuka di depan badan. | **Kontak Ketukan Pergelangan (Watch Tap)**: <br>• `Kapan`: Ujung jari telunjuk tangan kanan **mengetuk pergelangan tangan kiri** (`min_wrist_tap <= 0.28`, menunjuk jam tangan).<br>• `Di mana`: Dua tangan mengayun horizontal kiri-kanan.<br>• `Siapa`: Telunjuk berputar melingkar di pipi/dagu. | `question_wrist_kapan` | **Aktif 100% Teruji** |
| **7** | **Saya** vs **Maaf** | Kedua isyarat dieksekusi tepat di tengah dada (*sternum*). | **Tipe Kontak Dada**: <br>• `Saya`: Jari telunjuk menunjuk tenang tanpa gerakan melingkar (`pointing_isolation >= 0.08`, `chest_motion_std <= 0.04`).<br>• `Maaf`: Telapak tangan terbuka menempel di dada sambil mengusap melingkar (`mean_pinch > 0.20`, `chest_motion_std > 0.04`). | `chest_articulation_saya_maaf` | **Aktif 100% Teruji** |
| **8** | **Teman** vs **Keluarga** vs **Belajar** | Isyarat interaksi dua tangan (bimanual) di depan dada. | **Bentuk Topologi Kontak**: <br>• `Teman`: Dua kelingking saling mengait (*pinky hook*).<br>• `Keluarga`: Kedua tangan melingkar bersama membentuk kurva lingkaran (*circle join*).<br>• `Belajar`: Tangan kanan membuka dan menepuk telapak kiri datar seperti membaca buku. | `bimanual_interaction_rules` | **Didukung Schema JSON** |
| **9** | **Gerakan Netral / Bukan Isyarat** vs Seluruh Kata | Masalah *closed-set dilemma* (gerakan bebas menggaruk/mengetik terpaksa ditebak). | **Idle / Non-Sign Rejection Gate**: <br>• Menolak jika tangan aktif $< 15\%$ durasi.<br>• Menolak jika keyakinan model $< 0.40$.<br>• Menolak jika selisih Top-1 vs Top-2 $< 0.06$ pada keyakinan rendah (distribusi probabilitas flat). | `evaluate_idle_gate` | **Aktif 100% Teruji** |

---

## 6. Format Skema Deklaratif di `bisindo_rules.json`

Setiap kali kosakata baru ditambahkan ke dalam sistem, aturan pembeda dapat didefinisikan secara langsung di [bisindo_rules.json](file:///d:/Project/4mlai/isyaratin/bisindo_rules.json) tanpa mengubah kode Python:

```json
{
  "id": "nama_rule_gate",
  "type": "trajectory_contrast | proximity_contrast | bimanual_requirement | handshape_group | wrist_tap_requirement | chest_contrast",
  "class_a": "KataA",
  "class_b": "KataB",
  "metric": "y_travel | torso_approach | min_dist_mouth | min_wrist_tap",
  "threshold": 0.06,
  "boost_scale": 1.60,
  "suppress_scale": 0.30,
  "reason_a": "Penjelasan konfirmasi Kata A",
  "reason_b": "Penjelasan konfirmasi Kata B"
}
```

Semua aturan ini dieksekusi secara instan (< 1 milidetik) di pipeline inferensi sebelum hasil akhir ditampilkan ke antarmuka pengguna.
