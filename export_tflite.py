"""
export_tflite.py - Konversi Model PyTorch (MGN / BiLSTM) BISINDO ke TensorFlow Lite (.tflite).

Modul ini menyediakan transfer bobot otomatis dari PyTorch state_dict ke arsitektur Keras
yang di-unroll, lalu dikonversi menjadi file TFLite murni (Pure TFLITE_BUILTINS) tanpa
ketergantungan Select TF Ops / Flex Delegate, sehingga kompatibel 100% dengan tflite_flutter.

Keluaran:
  - sign_model.tflite : Model TFLite input [1, 32, feat_dim] -> output [1, num_classes] (softmax)
  - labels.txt        : Daftar nama kelas isyarat, 1 per baris
  - model_spec.json   : Metadata spesifikasi input/output untuk mobile developer
"""

import argparse
import copy
import json
import os
import shutil
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def detect_model_metadata(state_dict: Dict[str, torch.Tensor], meta: Optional[dict] = None) -> dict:
    """
    Mendeteksi arsitektur, dimensi input, dan ukuran layer dari state_dict PyTorch.
    """
    detected = {}
    if meta:
        detected.update(meta)

    # Deteksi arsitektur
    if "arch" not in detected:
        if "conv_in.weight" in state_dict:
            detected["arch"] = "mgn"
        else:
            detected["arch"] = "lstm"

    # Deteksi input feature dimension
    if "feat_dim" not in detected:
        if "input_norm.weight" in state_dict:
            detected["feat_dim"] = state_dict["input_norm.weight"].shape[0]
        else:
            detected["feat_dim"] = 400 if detected["arch"] == "mgn" else 186

    # Deteksi jumlah kelas output
    if "num_classes" not in detected:
        head_out_keys = ["head.4.weight", "head.3.weight", "head.weight"]
        for k in head_out_keys:
            if k in state_dict:
                detected["num_classes"] = state_dict[k].shape[0]
                break
        if "num_classes" not in detected:
            detected["num_classes"] = 28

    # Deteksi layer sizes
    if detected["arch"] == "mgn":
        if "conv_in.weight" in state_dict:
            detected["hidden1"] = state_dict["conv_in.weight"].shape[0]  # conv_dim
        else:
            detected["hidden1"] = 128
        if "lstm.weight_hh_l0" in state_dict:
            detected["hidden2"] = state_dict["lstm.weight_hh_l0"].shape[1]  # lstm_hidden
        else:
            detected["hidden2"] = 96
        if "head.0.weight" in state_dict:
            detected["head_dim"] = state_dict["head.0.weight"].shape[0]
        else:
            detected["head_dim"] = 192
    else:
        if "lstm1.weight_hh_l0" in state_dict:
            detected["hidden1"] = state_dict["lstm1.weight_hh_l0"].shape[1]
        else:
            detected["hidden1"] = 128
        if "lstm2.weight_hh_l0" in state_dict:
            detected["hidden2"] = state_dict["lstm2.weight_hh_l0"].shape[1]
        else:
            detected["hidden2"] = 96
        if "head.0.weight" in state_dict:
            detected["head_dim"] = state_dict["head.0.weight"].shape[0]
        else:
            detected["head_dim"] = 192

    if "seq_len" not in detected:
        detected["seq_len"] = 32

    return detected


def build_keras_mgn_model(
    feat_dim: int,
    seq_len: int,
    num_classes: int,
    conv_dim: int,
    lstm_hidden: int,
    head_dim: int,
):
    """
    Membangun model Keras MGN equivalent dengan unroll=True agar menghasilkan
    pure TFLite builtin ops (tanpa Flex Delegate).
    """
    import keras
    from keras import layers, ops

    # Layer definitions
    l_norm = layers.LayerNormalization(epsilon=1e-5, name="input_norm")
    l_cin = layers.Conv1D(conv_dim, 1, padding="same", name="conv_in")
    l_c3 = layers.Conv1D(conv_dim, 3, padding="same", name="conv3")
    l_c5 = layers.Conv1D(conv_dim, 5, padding="same", name="conv5")
    l_bn = layers.BatchNormalization(epsilon=1e-5, momentum=0.1, name="conv_bn")
    l_gelu = layers.Activation("gelu", name="conv_act")

    l_lstm1 = layers.Bidirectional(
        layers.LSTM(lstm_hidden, return_sequences=True, unroll=True),
        name="lstm_l0",
    )
    l_lstm2 = layers.Bidirectional(
        layers.LSTM(lstm_hidden, return_sequences=True, unroll=True),
        name="lstm_l1",
    )

    l_att1 = layers.Dense(64, activation="tanh", name="attn_dense1")
    l_att2 = layers.Dense(1, name="attn_dense2")

    l_h1 = layers.Dense(head_dim, activation="gelu", name="head_dense1")
    l_hnorm = layers.LayerNormalization(epsilon=1e-5, name="head_norm")
    l_out = layers.Dense(num_classes, name="head_logits")
    l_softmax = layers.Softmax(name="probabilities")

    # Computational Graph
    inp = layers.Input(shape=(seq_len, feat_dim), batch_size=1, name="input_landmarks")
    norm = l_norm(inp)
    cin = l_cin(norm)
    c3 = l_c3(cin)
    c5 = l_c5(cin)
    bn = l_bn(c3 + c5)
    micro = l_gelu(bn + cin)

    lstm1 = l_lstm1(micro)
    lstm2 = l_lstm2(lstm1)

    att1 = l_att1(lstm2)
    att2 = l_att2(att1)
    att_w = ops.softmax(att2, axis=1)
    attn_pool = ops.sum(lstm2 * att_w, axis=1)
    max_pool = ops.max(lstm2, axis=1)
    pooled = ops.concatenate([attn_pool, max_pool], axis=-1)

    h1 = l_h1(pooled)
    hnorm = l_hnorm(h1)
    logits = l_out(hnorm)
    probs = l_softmax(logits)

    model = keras.Model(inp, probs, name="BisindoMGNClassifier")
    layers_dict = {
        "norm": l_norm,
        "cin": l_cin,
        "c3": l_c3,
        "c5": l_c5,
        "bn": l_bn,
        "lstm1": l_lstm1,
        "lstm2": l_lstm2,
        "att1": l_att1,
        "att2": l_att2,
        "h1": l_h1,
        "hnorm": l_hnorm,
        "out": l_out,
    }
    return model, layers_dict


def build_keras_lstm_model(
    feat_dim: int,
    seq_len: int,
    num_classes: int,
    hidden1: int,
    hidden2: int,
    head_dim: int,
):
    """
    Membangun model Keras BiLSTM 2-Layer equivalent dengan unroll=True.
    """
    import keras
    from keras import layers, ops

    l_norm = layers.LayerNormalization(epsilon=1e-5, name="input_norm")
    l_lstm1 = layers.Bidirectional(
        layers.LSTM(hidden1, return_sequences=True, unroll=True),
        name="lstm1",
    )
    l_lstm2 = layers.Bidirectional(
        layers.LSTM(hidden2, return_sequences=True, unroll=True),
        name="lstm2",
    )

    l_h1 = layers.Dense(head_dim, activation="gelu", name="head_dense1")
    l_hnorm = layers.LayerNormalization(epsilon=1e-5, name="head_norm")
    l_out = layers.Dense(num_classes, name="head_logits")
    l_softmax = layers.Softmax(name="probabilities")

    inp = layers.Input(shape=(seq_len, feat_dim), batch_size=1, name="input_landmarks")
    norm = l_norm(inp)
    h1 = l_lstm1(norm)
    h2 = l_lstm2(h1)

    mean_pool = ops.mean(h2, axis=1)
    max_pool = ops.max(h2, axis=1)
    pooled = ops.concatenate([mean_pool, max_pool], axis=-1)

    dense1 = l_h1(pooled)
    hnorm = l_hnorm(dense1)
    logits = l_out(hnorm)
    probs = l_softmax(logits)

    model = keras.Model(inp, probs, name="BisindoLSTMClassifier")
    layers_dict = {
        "norm": l_norm,
        "lstm1": l_lstm1,
        "lstm2": l_lstm2,
        "h1": l_h1,
        "hnorm": l_hnorm,
        "out": l_out,
    }
    return model, layers_dict


def transfer_weights(arch: str, state_dict: Dict[str, torch.Tensor], layers_dict: dict):
    """
    Menyalin bobot tensor dari PyTorch ke Keras Layer dengan konversi layout dimensi.
    """
    sd = state_dict

    # 1. Input LayerNorm
    layers_dict["norm"].set_weights([
        sd["input_norm.weight"].cpu().numpy(),
        sd["input_norm.bias"].cpu().numpy(),
    ])

    if arch == "mgn":
        # 2. Convolutions & BatchNorm
        layers_dict["cin"].set_weights([
            sd["conv_in.weight"].permute(2, 1, 0).cpu().numpy(),
            sd["conv_in.bias"].cpu().numpy(),
        ])
        layers_dict["c3"].set_weights([
            sd["conv3.weight"].permute(2, 1, 0).cpu().numpy(),
            sd["conv3.bias"].cpu().numpy(),
        ])
        layers_dict["c5"].set_weights([
            sd["conv5.weight"].permute(2, 1, 0).cpu().numpy(),
            sd["conv5.bias"].cpu().numpy(),
        ])
        layers_dict["bn"].set_weights([
            sd["conv_bn.weight"].cpu().numpy(),
            sd["conv_bn.bias"].cpu().numpy(),
            sd["conv_bn.running_mean"].cpu().numpy(),
            sd["conv_bn.running_var"].cpu().numpy(),
        ])

        # 3. 2-Layer BiLSTM
        def get_lstm_w(l_idx):
            f_w = [
                sd[f"lstm.weight_ih_l{l_idx}"].cpu().numpy().T,
                sd[f"lstm.weight_hh_l{l_idx}"].cpu().numpy().T,
                (sd[f"lstm.bias_ih_l{l_idx}"] + sd[f"lstm.bias_hh_l{l_idx}"]).cpu().numpy(),
            ]
            b_w = [
                sd[f"lstm.weight_ih_l{l_idx}_reverse"].cpu().numpy().T,
                sd[f"lstm.weight_hh_l{l_idx}_reverse"].cpu().numpy().T,
                (sd[f"lstm.bias_ih_l{l_idx}_reverse"] + sd[f"lstm.bias_hh_l{l_idx}_reverse"]).cpu().numpy(),
            ]
            return f_w, b_w

        f0, b0 = get_lstm_w(0)
        layers_dict["lstm1"].forward_layer.set_weights(f0)
        layers_dict["lstm1"].backward_layer.set_weights(b0)

        f1, b1 = get_lstm_w(1)
        layers_dict["lstm2"].forward_layer.set_weights(f1)
        layers_dict["lstm2"].backward_layer.set_weights(b1)

        # 4. Temporal Attention
        layers_dict["att1"].set_weights([
            sd["attn_pool.attn.0.weight"].cpu().numpy().T,
            sd["attn_pool.attn.0.bias"].cpu().numpy(),
        ])
        layers_dict["att2"].set_weights([
            sd["attn_pool.attn.2.weight"].cpu().numpy().T,
            sd["attn_pool.attn.2.bias"].cpu().numpy(),
        ])

    else:
        # LSTM Architecture
        def get_lstm_w(prefix):
            f_w = [
                sd[f"{prefix}.weight_ih_l0"].cpu().numpy().T,
                sd[f"{prefix}.weight_hh_l0"].cpu().numpy().T,
                (sd[f"{prefix}.bias_ih_l0"] + sd[f"{prefix}.bias_hh_l0"]).cpu().numpy(),
            ]
            b_w = [
                sd[f"{prefix}.weight_ih_l0_reverse"].cpu().numpy().T,
                sd[f"{prefix}.weight_hh_l0_reverse"].cpu().numpy().T,
                (sd[f"{prefix}.bias_ih_l0_reverse"] + sd[f"{prefix}.bias_hh_l0_reverse"]).cpu().numpy(),
            ]
            return f_w, b_w

        f1, b1 = get_lstm_w("lstm1")
        layers_dict["lstm1"].forward_layer.set_weights(f1)
        layers_dict["lstm1"].backward_layer.set_weights(b1)

        f2, b2 = get_lstm_w("lstm2")
        layers_dict["lstm2"].forward_layer.set_weights(f2)
        layers_dict["lstm2"].backward_layer.set_weights(b2)

    # 5. MLP Classification Head
    layers_dict["h1"].set_weights([
        sd["head.0.weight"].cpu().numpy().T,
        sd["head.0.bias"].cpu().numpy(),
    ])
    layers_dict["hnorm"].set_weights([
        sd["head.3.weight"].cpu().numpy(),
        sd["head.3.bias"].cpu().numpy(),
    ])
    layers_dict["out"].set_weights([
        sd["head.4.weight"].cpu().numpy().T,
        sd["head.4.bias"].cpu().numpy(),
    ])


def export_checkpoint_to_tflite(
    checkpoint_path: str,
    output_tflite_path: str,
    output_labels_path: str,
    output_spec_path: Optional[str] = None,
    flutter_assets_dir: Optional[str] = None,
    labels_file: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Fungsi utama untuk membaca PyTorch checkpoint .pt dan menghasilkan model .tflite + labels.txt.
    """
    import tensorflow as tf

    print(f"\n=========================================================")
    print(f" Memulai Konversi PyTorch -> TensorFlow Lite (Mobile Ready)")
    print(f" Checkpoint Sumber : {checkpoint_path}")
    print(f" Target TFLite     : {output_tflite_path}")
    print(f" Target Labels     : {output_labels_path}")
    print(f"=========================================================\n")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint tidak ditemukan di: {checkpoint_path}")

    # Muat PyTorch Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("ema_state_dict", checkpoint.get("model_state_dict", checkpoint))
    meta = checkpoint.get("meta", {})

    cfg = detect_model_metadata(state_dict, meta)
    arch = cfg["arch"]
    feat_dim = cfg["feat_dim"]
    seq_len = cfg["seq_len"]
    num_classes = cfg["num_classes"]
    hidden1 = cfg["hidden1"]
    hidden2 = cfg["hidden2"]
    head_dim = cfg["head_dim"]

    print(f"[1/5] Metadata Terdeteksi:")
    print(f"  - Arsitektur   : {arch.upper()}")
    print(f"  - Feature Dim  : {feat_dim}")
    print(f"  - Sequence Len : {seq_len} frames")
    print(f"  - Jumlah Kelas : {num_classes}")
    print(f"  - Hidden Dims  : h1={hidden1}, h2={hidden2}, head={head_dim}")

    # Dapatkan Nama Label Kelas
    class_names = cfg.get("class_names")
    if not class_names and labels_file and os.path.exists(labels_file):
        with open(labels_file, "r", encoding="utf-8") as f:
            class_names = json.load(f)

    if not class_names:
        # Fallback candidates
        for cand in ["labels/wl_labels.json", "wl_labels.json", "labels/manual_labels.json", "manual_labels.json"]:
            if os.path.exists(cand):
                try:
                    with open(cand, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                        if len(loaded) == num_classes:
                            class_names = loaded
                            break
                except Exception:
                    pass

    if not class_names or len(class_names) != num_classes:
        print(f"  [WARN] Daftar nama label tidak ditemukan/sesuai. Menggunakan indeks numerik Isyarat 0..{num_classes-1}")
        class_names = [f"Isyarat {i}" for i in range(num_classes)]
    else:
        print(f"  [OK] Berhasil memetakan {len(class_names)} nama kelas isyarat.")

    # Bangun Model Keras
    print(f"\n[2/5] Membangun Arsitektur Keras Equivalent (unroll=True)...")
    if arch == "mgn":
        keras_model, layers_dict = build_keras_mgn_model(
            feat_dim=feat_dim,
            seq_len=seq_len,
            num_classes=num_classes,
            conv_dim=hidden1,
            lstm_hidden=hidden2,
            head_dim=head_dim,
        )
    else:
        keras_model, layers_dict = build_keras_lstm_model(
            feat_dim=feat_dim,
            seq_len=seq_len,
            num_classes=num_classes,
            hidden1=hidden1,
            hidden2=hidden2,
            head_dim=head_dim,
        )

    # Transfer Bobot
    print(f"[3/5] Mentransfer Bobot StateDict PyTorch ke Keras...")
    transfer_weights(arch, state_dict, layers_dict)

    # Konversi ke TFLite
    print(f"[4/5] Mengonversi Keras Model ke Pure TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_bytes = converter.convert()

    os.makedirs(os.path.dirname(os.path.abspath(output_tflite_path)), exist_ok=True)
    with open(output_tflite_path, "wb") as f:
        f.write(tflite_bytes)
    tflite_size_kb = len(tflite_bytes) / 1024
    print(f"  [OK] Model TFLite tersimpan di: {output_tflite_path} ({tflite_size_kb:.2f} KB)")

    # Simpan Labels.txt
    os.makedirs(os.path.dirname(os.path.abspath(output_labels_path)), exist_ok=True)
    with open(output_labels_path, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"  [OK] Daftar label tersimpan di: {output_labels_path} ({len(class_names)} baris)")

    # Validasi Inference dengan TFLite Interpreter
    print(f"\n[5/5] Memvalidasi Inferensi TFLite Runtime...")
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    test_input = np.random.randn(1, seq_len, feat_dim).astype(np.float32)
    interp.set_tensor(in_det["index"], test_input)
    interp.invoke()
    tflite_out = interp.get_tensor(out_det["index"])

    keras_out = keras_model.predict(test_input, verbose=0)
    diff = float(np.max(np.abs(keras_out - tflite_out)))
    print(f"  - Input shape  : {in_det['shape'].tolist()} ({in_det['dtype'].__name__})")
    print(f"  - Output shape : {out_det['shape'].tolist()} ({out_det['dtype'].__name__})")
    print(f"  - Maximum diff : {diff:.8f}")
    assert diff < 1e-4, f"Akurasi inferensi TFLite menyimpang terlalu jauh: {diff}"
    print(f"  [OK] Inferensi TFLite 100% IDENTIK dengan Model Keras!")

    # Buat File Spesifikasi Model (model_spec.json)
    spec_data = {
        "model_name": "BISINDO Sign Recognition",
        "arch": arch,
        "input": {
            "name": in_det["name"],
            "shape": in_det["shape"].tolist(),
            "dtype": "float32",
            "description": f"Sequence of {seq_len} frames, each containing {feat_dim} normalized landmarks/kinematic features.",
            "target_fps": 12.0,
        },
        "output": {
            "name": out_det["name"],
            "shape": out_det["shape"].tolist(),
            "dtype": "float32",
            "activation": "softmax (probabilities [0, 1])",
        },
        "num_classes": num_classes,
        "class_names": class_names,
    }
    if output_spec_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_spec_path)), exist_ok=True)
        with open(output_spec_path, "w", encoding="utf-8") as f:
            json.dump(spec_data, f, indent=2, ensure_ascii=False)
        print(f"  [OK] Metadata spesifikasi tersimpan di: {output_spec_path}")

    # Salin otomatis ke Flutter Assets jika diarahkan
    candidate_flutter_dirs = [
        flutter_assets_dir,
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "isyaratin", "assets", "models"),
        r"d:\Project\4mlai\isyaratin\isyaratin\assets\models",
        r"d:\Project\4mlai\apapun\isyaratin\assets\models",
    ]
    for f_dir in set(candidate_flutter_dirs):
        if f_dir and os.path.exists(f_dir):
            dest_tflite = os.path.join(f_dir, "sign_model.tflite")
            dest_labels = os.path.join(f_dir, "labels.txt")
            shutil.copyfile(output_tflite_path, dest_tflite)
            shutil.copyfile(output_labels_path, dest_labels)
            print(f"\n[FLUTTER SYNC] Berhasil menyalin model dan label ke aset Flutter:")
            print(f"  -> {dest_tflite}")
            print(f"  -> {dest_labels}")

    return output_tflite_path, output_labels_path


def parse_args():
    parser = argparse.ArgumentParser(description="Ekspor Model BISINDO PyTorch (.pt) ke TensorFlow Lite (.tflite)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/bisindo_lstm_best.pt",
        help="Path ke checkpoint PyTorch (.pt)",
    )
    parser.add_argument(
        "--out-tflite",
        type=str,
        default="models/sign_model.tflite",
        help="Path output file .tflite (default: models/sign_model.tflite)",
    )
    parser.add_argument(
        "--out-labels",
        type=str,
        default="models/labels.txt",
        help="Path output file labels.txt (default: models/labels.txt)",
    )
    parser.add_argument(
        "--labels-file",
        type=str,
        default=None,
        help="Path opsional ke file JSON label (misal: labels/wl_labels.json)",
    )
    parser.add_argument(
        "--flutter-dir",
        type=str,
        default=r"d:\Project\4mlai\apapun\isyaratin\assets\models",
        help="Folder assets/models Flutter untuk auto-sync (default: d:/Project/4mlai/apapun/isyaratin/assets/models)",
    )
    parser.add_argument(
        "--no-flutter-copy",
        action="store_true",
        help="Jangan salin otomatis ke folder Flutter",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    flutter_assets = None if args.no_flutter_copy else args.flutter_dir
    spec_path = os.path.join(os.path.dirname(os.path.abspath(args.out_tflite)), "model_spec.json")

    export_checkpoint_to_tflite(
        checkpoint_path=args.checkpoint,
        output_tflite_path=args.out_tflite,
        output_labels_path=args.out_labels,
        output_spec_path=spec_path,
        flutter_assets_dir=flutter_assets,
        labels_file=args.labels_file,
    )
