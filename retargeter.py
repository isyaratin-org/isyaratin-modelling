"""
retargeter.py - Kinematic Motion Retargeting dari MediaPipe Landmarks ke Skeleton 3D Avatar (man-rigged.glb).

Fitur:
  1. Transformasi sistem koordinat (MediaPipe Screen-Space -> Avatar World/Local Space).
  2. Perhitungan rotasi Upper Body (Shoulder, Arm, ForeArm, Wrist).
  3. Artikulasi jari 5 jari (Thumb, Index, Middle, Ring, Pinky) x 3 ruas per tangan.
  4. Perhitungan rotasi lokal per-tulang relatif terhadap parent bone (Hierarchical Quaternions).
  5. Penghalusan temporal (Spherical Linear Interpolation / Slerp filter).
  6. Ekspor animasi ke format JSON Motion Data terstruktur.
"""

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


def normalize_vector(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return v / norm


def quaternion_from_two_vectors(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Menghitung quaternion rotasi terpendek dari vektor u ke vektor v: [x, y, z, w]."""
    u = normalize_vector(u)
    v = normalize_vector(v)
    dot = np.dot(u, v)

    if dot > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    elif dot < -0.999999:
        # 180 derajat: cari aksis ortogonal
        axis = np.cross(np.array([1.0, 0.0, 0.0]), u)
        if np.linalg.norm(axis) < 0.01:
            axis = np.cross(np.array([0.0, 1.0, 0.0]), u)
        axis = normalize_vector(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0], dtype=np.float32)

    cross = np.cross(u, v)
    w = 1.0 + dot
    q = np.array([cross[0], cross[1], cross[2], w], dtype=np.float32)
    return normalize_vector(q)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Mengalikan dua quaternion: q1 * q2 [x, y, z, w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w], dtype=np.float32)


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    result = np.asarray(q, dtype=np.float32) / norm
    # q dan -q menyatakan orientasi yang sama. W positif membuat serialisasi
    # stabil dan menghindari lompatan tanda antarkeyframe.
    if result[3] < 0:
        result = -result
    return result


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    norm_sq = float(np.dot(q, q))
    if norm_sq < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return quaternion_conjugate(q) / norm_sq


def quaternion_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = normalize_vector(np.asarray(axis, dtype=np.float32))
    half = angle_rad * 0.5
    sin_half = math.sin(half)
    return normalize_quaternion(np.array([
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half,
        math.cos(half),
    ], dtype=np.float32))


def quaternion_rotate_vector(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(q)
    vector_q = np.array([vector[0], vector[1], vector[2], 0.0], dtype=np.float32)
    return quaternion_multiply(
        quaternion_multiply(q, vector_q), quaternion_inverse(q)
    )[:3]


def quaternion_slerp(q1: np.ndarray, q2: np.ndarray, alpha: float) -> np.ndarray:
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)
    dot = float(np.dot(q1, q2))
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        return normalize_quaternion(q1 + (q2 - q1) * alpha)
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    return normalize_quaternion(
        q1 * (math.sin(theta_0 - theta) / sin_theta_0)
        + q2 * (math.sin(theta) / sin_theta_0)
    )


def quaternion_angle_degrees(q1: np.ndarray, q2: np.ndarray) -> float:
    """Jarak sudut terpendek dua orientasi quaternion."""
    dot = abs(float(np.dot(normalize_quaternion(q1), normalize_quaternion(q2))))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Konversi matriks rotasi 3x3 menjadi quaternion [x, y, z, w]."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
            0.25 * s,
        ])
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q = np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
        elif i == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q = np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q = np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s])
    return normalize_quaternion(q.astype(np.float32))


def quaternion_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
    """Konversi quaternion ke sudut Euler (Roll, Pitch, Yaw)."""
    x, y, z, w = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def euler_to_quaternion(x: float, y: float, z: float, order: str = "XYZ") -> np.ndarray:
    """Setara dengan THREE.Quaternion.setFromEuler untuk order yang dipakai studio."""
    c1, c2, c3 = math.cos(x * 0.5), math.cos(y * 0.5), math.cos(z * 0.5)
    s1, s2, s3 = math.sin(x * 0.5), math.sin(y * 0.5), math.sin(z * 0.5)

    if order == "XYZ":
        q = np.array([
            s1 * c2 * c3 + c1 * s2 * s3,
            c1 * s2 * c3 - s1 * c2 * s3,
            c1 * c2 * s3 + s1 * s2 * c3,
            c1 * c2 * c3 - s1 * s2 * s3,
        ], dtype=np.float32)
    elif order == "YXZ":
        q = np.array([
            s1 * c2 * c3 + c1 * s2 * s3,
            c1 * s2 * c3 - s1 * c2 * s3,
            c1 * c2 * s3 - s1 * s2 * c3,
            c1 * c2 * c3 + s1 * s2 * s3,
        ], dtype=np.float32)
    else:
        raise ValueError(f"Euler order tidak didukung: {order}")
    return normalize_quaternion(q)


class AvatarRetargeter:
    """
    Mesin Retargeting Kinematik yang memetakan landmark pose & tangan
    ke rotasi tulang avatar 3D 'man-rigged.glb'.
    """

    def __init__(
        self,
        inspection_report_path: str = "avatar_inspection_report.json",
        glb_path: str = "avatar/man-rigged.glb",
    ):
        p = Path(inspection_report_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                self.report = json.load(f)
        else:
            self.report = {}

        (
            self.rest_rotations,
            self.node_translations,
            self.node_parents,
        ) = self._read_glb_rig(Path(glb_path))
        self.rest_world_rotations = self._build_rest_world_rotations()

        # Definisi tulang lengan
        self.right_arm_bones = ["RightArm", "RightForeArm", "RightHand"]
        self.left_arm_bones = ["LeftArm", "LeftForeArm", "LeftHand"]

        # Pemetaan sendi MediaPipe Hand (21 points) ke 3 ruas tulang avatar (5 jari)
        # MediaPipe Hand index:
        # Thumb: 1, 2, 3, 4
        # Index: 5, 6, 7, 8
        # Middle: 9, 10, 11, 12
        # Ring: 13, 14, 15, 16
        # Pinky: 17, 18, 19, 20
        self.finger_map = {
            "Thumb":  {"bones": ["Thumb1", "Thumb2", "Thumb3"],   "joints": [1, 2, 3, 4]},
            "Index":  {"bones": ["Index1", "Index2", "Index3"],   "joints": [5, 6, 7, 8]},
            "Middle": {"bones": ["Middle1", "Middle2", "Middle3"], "joints": [9, 10, 11, 12]},
            "Ring":   {"bones": ["Ring1", "Ring2", "Ring3"],     "joints": [13, 14, 15, 16]},
            "Pinky":  {"bones": ["Pinky1", "Pinky2", "Pinky3"],   "joints": [17, 18, 19, 20]},
        }

        # Kalibrasi parent-space dari pembelajaran.md untuk rig ini.
        self.knuckle_hinge = {
            "Right": normalize_vector(np.array([0.824, -0.116, 0.555], dtype=np.float32)),
            "Left": normalize_vector(np.array([-0.824, -0.116, 0.555], dtype=np.float32)),
        }
        self.palm_normal = {
            "Right": normalize_vector(np.array([0.566, 0.107, -0.818], dtype=np.float32)),
            "Left": normalize_vector(np.array([-0.566, 0.107, -0.818], dtype=np.float32)),
        }
        self.thumb_opposition = {
            "Right": normalize_vector(np.array([0.707, 0.150, -0.690], dtype=np.float32)),
            "Left": normalize_vector(np.array([-0.707, 0.150, -0.690], dtype=np.float32)),
        }

    @staticmethod
    def _read_glb_rig(glb_path: Path):
        """Baca quaternion, translation, dan hierarchy langsung dari GLB."""
        if not glb_path.exists():
            raise FileNotFoundError(f"Model GLB tidak ditemukan: {glb_path}")
        with glb_path.open("rb") as handle:
            magic, version, _ = struct.unpack("<4sII", handle.read(12))
            if magic != b"glTF" or version != 2:
                raise ValueError(f"GLB tidak valid atau bukan glTF 2.0: {glb_path}")
            chunk_length, chunk_type = struct.unpack("<II", handle.read(8))
            if chunk_type != 0x4E4F534A:  # JSON
                raise ValueError(f"Chunk pertama GLB bukan JSON: {glb_path}")
            document = json.loads(handle.read(chunk_length).decode("utf-8").rstrip("\x00 "))

        nodes = document.get("nodes", [])
        parent_indices = {
            child_index: parent_index
            for parent_index, node in enumerate(nodes)
            for child_index in node.get("children", [])
        }
        rotations: Dict[str, np.ndarray] = {}
        translations: Dict[str, np.ndarray] = {}
        parents: Dict[str, Optional[str]] = {}
        for index, node in enumerate(nodes):
            name = node.get("name")
            if name:
                rotations[name] = normalize_quaternion(np.array(
                    node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32
                ))
                translations[name] = np.array(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
                parent_index = parent_indices.get(index)
                parents[name] = nodes[parent_index].get("name") if parent_index is not None else None
        return rotations, translations, parents

    def _build_rest_world_rotations(self) -> Dict[str, np.ndarray]:
        result: Dict[str, np.ndarray] = {}

        def resolve(name: str) -> np.ndarray:
            if name in result:
                return result[name]
            local = self.rest_rotations.get(name, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
            parent = self.node_parents.get(name)
            result[name] = (
                normalize_quaternion(quaternion_multiply(resolve(parent), local))
                if parent else normalize_quaternion(local)
            )
            return result[name]

        for bone_name in self.rest_rotations:
            resolve(bone_name)
        return result

    def convert_mp_to_3d_point(self, pt: Dict, origin: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Mengonversi koordinat MediaPipe ke sistem koordinat avatar (Y-up, right-handed):
          X_avatar = +(X_mp - origin_X)
          Y_avatar = -(Y_mp - origin_Y)
          Z_avatar = -(Z_mp - origin_Z)
        """
        x = float(pt.get("x", 0.0))
        y = float(pt.get("y", 0.0))
        z = float(pt.get("z", 0.0))

        if origin is not None:
            x -= origin[0]
            y -= origin[1]
            z -= origin[2]

        return np.array([x, -y, -z], dtype=np.float32)

    def retarget_arm(
        self,
        shoulder_pt: np.ndarray,
        elbow_pt: np.ndarray,
        wrist_pt: np.ndarray,
        is_right: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Menghitung rotasi UpperArm dan ForeArm berdasarkan arah vektor.
        Avatar rest-pose:
          RightArm mengarah diagonal bawah: [0.5, -0.866, 0.0]
          LeftArm mengarah diagonal bawah: [-0.5, -0.866, 0.0]
        """
        side_sign = 1.0 if is_right else -1.0
        rest_arm_dir = normalize_vector(np.array([0.5 * side_sign, -0.866, 0.0]))
        rest_forearm_dir = rest_arm_dir

        target_arm_dir = normalize_vector(elbow_pt - shoulder_pt)
        target_forearm_dir = normalize_vector(wrist_pt - elbow_pt)

        # Rotasi World Upper Arm
        q_arm_world = quaternion_from_two_vectors(rest_arm_dir, target_arm_dir)

        # Rotasi ForeArm relatif terhadap Upper Arm
        # Putar rest_forearm_dir menggunakan q_arm_world, lalu cari koreksinya ke target_forearm_dir
        q_forearm_world = quaternion_from_two_vectors(rest_forearm_dir, target_forearm_dir)
        q_forearm_local = quaternion_multiply(quaternion_conjugate(q_arm_world), q_forearm_world)

        return q_arm_world, q_forearm_local

    def retarget_fingers(
        self,
        hand_pts: List[Dict],
        is_right: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Menghitung rotasi per ruas jari (3 ruas x 5 jari = 15 tulang).
        Menggunakan sudut tekukan (flexion angle) dan abduksi antar ruas.
        """
        rotations = {}
        prefix = "RightHand" if is_right else "LeftHand"

        # Wrist point
        p_wrist = np.array([hand_pts[0]["x"], -hand_pts[0]["y"], -hand_pts[0]["z"]])

        for finger_name, f_data in self.finger_map.items():
            bones = f_data["bones"]
            j_indices = f_data["joints"]

            pts = [
                np.array([hand_pts[idx]["x"], -hand_pts[idx]["y"], -hand_pts[idx]["z"]])
                for idx in j_indices
            ]

            # Vektor segmen tulang
            v0 = pts[0] - p_wrist       # Wrist -> MCP
            v1 = pts[1] - pts[0]        # MCP -> PIP (Bone 1)
            v2 = pts[2] - pts[1]        # PIP -> DIP (Bone 2)
            v3 = pts[3] - pts[2]        # DIP -> Tip (Bone 3)

            # Hitung sudut tekukan (flexion angle)
            # Bone 1: tekukan relatif terhadap telapak tangan
            def get_curl_rot(u, v, curl_axis=(1.0, 0.0, 0.0)):
                u_n = normalize_vector(u)
                v_n = normalize_vector(v)
                cos_theta = np.clip(np.dot(u_n, v_n), -1.0, 1.0)
                angle = math.acos(cos_theta)
                # Batasi rentang sudut realistis jari (0 s/d 90 derajat per sendi)
                angle = max(0.0, min(angle, 1.57))
                # Rotasi tekukan pada sumbu engsel (hinge axis X)
                half_angle = angle * 0.5
                return np.array([math.sin(half_angle), 0.0, 0.0, math.cos(half_angle)], dtype=np.float32)

            q_b1 = get_curl_rot(v0, v1)
            q_b2 = get_curl_rot(v1, v2)
            q_b3 = get_curl_rot(v2, v3)

            rotations[f"{prefix}{bones[0]}"] = [round(float(x), 4) for x in q_b1]
            rotations[f"{prefix}{bones[1]}"] = [round(float(x), 4) for x in q_b2]
            rotations[f"{prefix}{bones[2]}"] = [round(float(x), 4) for x in q_b3]

        return rotations

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, float(value)))

    @staticmethod
    def _point(pt: Dict) -> np.ndarray:
        return np.array([
            float(pt.get("x", 0.0)),
            float(pt.get("y", 0.0)),
            float(pt.get("z", 0.0)),
        ], dtype=np.float32)

    @staticmethod
    def _smooth(state: Dict[str, float], key: str, target: float, alpha: float) -> None:
        state[key] = state.get(key, 0.0) * (1.0 - alpha) + float(target) * alpha

    def _extract_arm_controls(
        self,
        pose: List[Dict],
        hand: Optional[List[Dict]],
        side: str,
        state: Dict[str, float],
    ) -> None:
        """Port solver MediaPipe Pose Studio agar offline dan webcam memakai semantik sama."""
        is_right = side == "Right"
        shoulder_idx, elbow_idx, wrist_idx = ((12, 14, 16) if is_right else (11, 13, 15))
        shoulder = self._point(pose[shoulder_idx])
        elbow = self._point(pose[elbow_idx])
        wrist = self._point(pose[wrist_idx])

        upper = elbow - shoulder
        lower = wrist - elbow
        len_upper = max(float(np.linalg.norm(upper)), 1e-6)
        len_lower = max(float(np.linalg.norm(lower)), 1e-6)

        bend_dot = float(np.dot(upper, lower) / (len_upper * len_lower))
        bend = math.degrees(math.acos(self._clamp(bend_dot, -1.0, 1.0)))
        bend = self._clamp(bend, 0.0, 135.0)

        outward_elbow = shoulder[0] - elbow[0] if is_right else elbow[0] - shoulder[0]
        outward_wrist = shoulder[0] - wrist[0] if is_right else wrist[0] - shoulder[0]
        effective_outward = max(float(outward_elbow), float(outward_wrist) * 0.75)
        spread = self._clamp((effective_outward - 0.04) * 155.0, -10.0, 80.0)

        wrist_elevation = float(shoulder[1] - wrist[1])
        elbow_elevation = float(shoulder[1] - elbow[1])
        forward_reach = max(0.0, float(-upper[2]) * 1.4) + max(
            0.0, float(-(wrist[2] - shoulder[2])) * 0.8
        )
        arm_up = 35.0 + wrist_elevation * 135.0 + max(0.0, elbow_elevation) * 55.0 + forward_reach * 45.0
        arm_up = self._clamp(arm_up, 0.0, 115.0)

        forearm_up = max(0.0, float(elbow[1] - wrist[1])) / len_lower
        arm_roll = self._clamp(forearm_up * 42.0 + max(0.0, spread) * 0.35, -15.0, 65.0)

        forearm_twist = wrist_pitch = wrist_yaw = wrist_roll = 0.0
        if hand and len(hand) == 21:
            p0, p5, p9, p17 = (self._point(hand[i]) for i in (0, 5, 9, 17))
            palm_vector = p9 - p0
            palm_length = max(float(np.linalg.norm(palm_vector)), 1e-6)
            knuckle_vector = (p5 - p17) if is_right else (p17 - p5)
            palm_normal = np.cross(palm_vector, knuckle_vector)
            normal_length = max(float(np.linalg.norm(palm_normal)), 1e-6)
            normal_z = float(palm_normal[2] / normal_length)

            forearm_twist = 12.0 + normal_z * 22.0 if normal_z >= 0 else -15.0 + normal_z * 35.0
            forearm_twist = self._clamp(forearm_twist, -65.0, 65.0)

            forearm_dir = lower / len_lower
            palm_dir = palm_vector / palm_length
            cross_pitch = float(forearm_dir[0] * palm_dir[1] - forearm_dir[1] * palm_dir[0])
            wrist_pitch = self._clamp(
                math.degrees(math.asin(self._clamp(cross_pitch, -1.0, 1.0))) * 1.1,
                -45.0,
                45.0,
            )
            yaw_delta = (
                float(palm_dir[0] - forearm_dir[0])
                if is_right
                else float(forearm_dir[0] - palm_dir[0])
            )
            wrist_yaw = self._clamp(yaw_delta * 45.0, -35.0, 35.0)
            wrist_roll = self._clamp(normal_z * -25.0, -35.0, 35.0)

        for suffix, value in (
            ("Arm_up", arm_up),
            ("Arm_spread", spread),
            ("Arm_roll", arm_roll),
            ("ForeArm_bend", bend),
            ("ForeArm_twist", forearm_twist),
            ("Hand_pitch", wrist_pitch),
            ("Hand_yaw", wrist_yaw),
            ("Hand_roll", wrist_roll),
        ):
            self._smooth(state, f"{side}{suffix}", value, 0.42)

    def _measure_hand_controls(
        self,
        hand: Optional[List[Dict]],
        side: str,
    ) -> Dict[str, float]:
        if not hand or len(hand) != 21:
            return {}

        points = [self._point(pt) for pt in hand]
        palm_size = max(float(np.linalg.norm(points[9] - points[0])), 1e-6)
        controls = {}
        fingers = (
            ("Index", 8, 6, 5),
            ("Middle", 12, 10, 9),
            ("Ring", 16, 14, 13),
            ("Pinky", 20, 18, 17),
        )
        for name, tip, pip, mcp in fingers:
            tip_mcp = float(np.linalg.norm(points[tip] - points[mcp]))
            pip_mcp = float(np.linalg.norm(points[pip] - points[mcp]))
            tip_pip = float(np.linalg.norm(points[tip] - points[pip]))
            full_length = max(pip_mcp + tip_pip, 1e-6)
            extension = self._clamp((tip_mcp / full_length - 0.40) / 0.52, 0.0, 1.0)
            controls[f"{side}_{name}"] = (1.0 - extension) * 85.0

        thumb_pinky = float(np.linalg.norm(points[4] - points[17])) / palm_size
        thumb_extension = self._clamp((thumb_pinky - 0.46) / 0.48, 0.0, 1.0)
        controls[f"{side}_Thumb"] = (1.0 - thumb_extension) * 75.0

        index_pinky = float(np.linalg.norm(points[8][:2] - points[20][:2])) / palm_size
        spread = self._clamp((index_pinky - 0.75) * 40.0, -10.0, 30.0)
        controls[f"{side}_Spread"] = spread
        return controls

    @staticmethod
    def _stabilize_pose_depth(frames: List[Dict]) -> None:
        """Redam spike satu-frame pada Z lengan tanpa mengubah lintasan X/Y."""
        original = [frame.get("poseWorld") for frame in frames]
        for frame_index in range(1, len(frames) - 1):
            if not all(
                isinstance(original[index], list) and len(original[index]) > 16
                for index in (frame_index - 1, frame_index, frame_index + 1)
            ):
                continue
            stable_pose = [dict(point) for point in original[frame_index]]
            for joint in (11, 12, 13, 14, 15, 16):
                neighbors = [original[index][joint] for index in
                             (frame_index - 1, frame_index, frame_index + 1)]
                if min(float(point.get("visibility", 1.0)) for point in neighbors) < 0.45:
                    continue
                stable_pose[joint]["z"] = float(np.median([
                    float(point.get("z", 0.0)) for point in neighbors
                ]))
            frames[frame_index]["poseWorld"] = stable_pose

    @staticmethod
    def _palm_footprint(
        hand: Optional[List[Dict]], width: float, height: float
    ) -> Optional[Tuple[float, float]]:
        """Ukuran dan aspek telapak 2D; jari tidak dipakai agar curl tidak mengubah skala."""
        if not hand or len(hand) != 21:
            return None

        def distance(a: int, b: int) -> float:
            return math.hypot(
                (float(hand[a]["x"]) - float(hand[b]["x"])) * width,
                (float(hand[a]["y"]) - float(hand[b]["y"])) * height,
            )

        length = distance(0, 9)       # pergelangan -> pangkal jari tengah
        breadth = distance(5, 17)     # pangkal telunjuk -> kelingking
        if length < 1.0 or breadth < 1.0:
            return None
        return math.log(math.sqrt(length * breadth)), math.log(breadth / length)

    def _apply_palm_size_depth_prior(
        self, frames: List[Dict], width: float, height: float
    ) -> Dict[str, Dict]:
        """Kalibrasi per-video: ukuran telapak hanya petunjuk kedalaman relatif.

        Pose-world tetap sumber utama. Koreksi dibatasi dan dimatikan jika
        hubungan ukuran-vs-Z tidak konsisten (mis. telapak berputar/samping).
        """
        diagnostics: Dict[str, Dict] = {}
        for side, hand_key, wrist_index in (
            ("Right", "rightHand", 16), ("Left", "leftHand", 15)
        ):
            observations = []
            for index, frame in enumerate(frames):
                pose = frame.get("poseWorld")
                footprint = self._palm_footprint(frame.get(hand_key), width, height)
                if not pose or len(pose) <= wrist_index or footprint is None:
                    continue
                wrist = pose[wrist_index]
                if float(wrist.get("visibility", 1.0)) < 0.55:
                    continue
                observations.append((index, footprint[0], footprint[1], float(wrist["z"])))

            if len(observations) < 12:
                diagnostics[side] = {"usedFrames": 0, "reason": "insufficient_landmarks"}
                continue

            typical_aspect = float(np.median([sample[2] for sample in observations]))
            stable = [sample for sample in observations
                      if abs(sample[2] - typical_aspect) <= 0.30]
            if len(stable) < 12:
                diagnostics[side] = {"usedFrames": 0, "reason": "changing_palm_orientation"}
                continue

            sizes = np.array([sample[1] for sample in stable], dtype=np.float32)
            depths = np.array([sample[3] for sample in stable], dtype=np.float32)
            size_center = float(np.mean(sizes))
            depth_center = float(np.mean(depths))
            size_delta = sizes - size_center
            depth_delta = depths - depth_center
            size_variance = float(np.dot(size_delta, size_delta))
            depth_variance = float(np.dot(depth_delta, depth_delta))
            if size_variance < 0.02 or depth_variance < 1e-5:
                diagnostics[side] = {"usedFrames": 0, "reason": "insufficient_size_change"}
                continue

            correlation = float(np.dot(size_delta, depth_delta) /
                                math.sqrt(size_variance * depth_variance))
            if correlation > -0.45:
                diagnostics[side] = {
                    "usedFrames": 0, "correlation": round(correlation, 3),
                    "reason": "unreliable_size_depth_relation",
                }
                continue

            slope = self._clamp(
                float(np.dot(size_delta, depth_delta) / size_variance), -0.5, -0.05
            )
            used = 0
            for frame in frames:
                pose = frame.get("poseWorld")
                footprint = self._palm_footprint(frame.get(hand_key), width, height)
                if not pose or len(pose) <= wrist_index or footprint is None:
                    continue
                if abs(footprint[1] - typical_aspect) > 0.30:
                    continue
                wrist = pose[wrist_index]
                current_z = float(wrist["z"])
                size_z = depth_center + slope * (footprint[0] - size_center)
                correction = self._clamp(size_z - current_z, -0.075, 0.075)
                confidence = 0.5 if float(wrist.get("visibility", 1.0)) < 0.6 else 0.35
                wrist["z"] = current_z + confidence * correction
                used += 1
            diagnostics[side] = {
                "usedFrames": used, "correlation": round(correlation, 3),
                "source": "relative_palm_size_prior",
            }
        return diagnostics

    def _parent_delta_to_local(self, bone_name: str, parent_delta: np.ndarray) -> np.ndarray:
        rest = self.rest_rotations.get(bone_name)
        if rest is None:
            return normalize_quaternion(parent_delta)
        return normalize_quaternion(quaternion_multiply(
            quaternion_multiply(quaternion_inverse(rest), parent_delta),
            rest,
        ))

    @staticmethod
    def _degrees_quaternion(x: float, y: float, z: float, order: str = "XYZ") -> np.ndarray:
        return euler_to_quaternion(math.radians(x), math.radians(y), math.radians(z), order)

    def _fill_short_landmark_gaps(
        self,
        frames: List[Dict],
        key: str,
        max_gap: int = 8,
    ) -> None:
        """Interpolasi gap deteksi singkat tanpa mengarang data di awal/akhir video."""
        detected = [
            index for index, frame in enumerate(frames)
            if isinstance(frame.get(key), list) and len(frame[key]) > 0
        ]
        for start, end in zip(detected, detected[1:]):
            gap = end - start - 1
            if gap <= 0 or gap > max_gap:
                continue
            before = frames[start][key]
            after = frames[end][key]
            if len(before) != len(after):
                continue
            for index in range(start + 1, end):
                alpha = (index - start) / (end - start)
                interpolated = []
                for point_a, point_b in zip(before, after):
                    point = {}
                    for component in ("x", "y", "z", "visibility"):
                        if component in point_a or component in point_b:
                            value_a = float(point_a.get(component, point_b.get(component, 0.0)))
                            value_b = float(point_b.get(component, point_a.get(component, 0.0)))
                            point[component] = value_a + (value_b - value_a) * alpha
                    interpolated.append(point)
                frames[index][key] = interpolated

    @staticmethod
    def _pose_world_point(point: Dict) -> np.ndarray:
        # MediaPipe world: X ke kanan citra, Y ke bawah, Z menjauhi kamera.
        # Avatar: X ke kanan dunia, Y ke atas, +Z menghadap kamera.
        return np.array([
            float(point.get("x", 0.0)),
            -float(point.get("y", 0.0)),
            -float(point.get("z", 0.0)),
        ], dtype=np.float32)

    def _solve_bone_direction(
        self,
        bone_name: str,
        child_name: str,
        target_world_direction: np.ndarray,
        parent_world_rotation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Arahkan sumbu bone ke target world dan kembalikan (delta, world rotation)."""
        rest = self.rest_rotations[bone_name]
        child_axis_local = normalize_vector(self.node_translations[child_name])
        rest_direction_parent = quaternion_rotate_vector(rest, child_axis_local)
        target_direction_parent = quaternion_rotate_vector(
            quaternion_inverse(parent_world_rotation),
            normalize_vector(target_world_direction),
        )
        swing_parent = quaternion_from_two_vectors(rest_direction_parent, target_direction_parent)
        actual_local = normalize_quaternion(quaternion_multiply(swing_parent, rest))
        delta_local = normalize_quaternion(quaternion_multiply(quaternion_inverse(rest), actual_local))
        actual_world = normalize_quaternion(quaternion_multiply(parent_world_rotation, actual_local))
        return delta_local, actual_world

    @staticmethod
    def _orientation_basis(primary: np.ndarray, lateral: np.ndarray) -> np.ndarray:
        y_axis = normalize_vector(primary)
        x_axis = lateral - y_axis * float(np.dot(lateral, y_axis))
        if float(np.linalg.norm(x_axis)) < 1e-5:
            x_axis = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float32), y_axis)
        x_axis = normalize_vector(x_axis)
        z_axis = normalize_vector(np.cross(x_axis, y_axis))
        x_axis = normalize_vector(np.cross(y_axis, z_axis))
        return np.column_stack((x_axis, y_axis, z_axis))

    def _solve_hand_orientation(
        self,
        side: str,
        hand: List[Dict],
        forearm_world_rotation: np.ndarray,
    ) -> np.ndarray:
        hand_name = f"{side}Hand"
        local_primary = self.node_translations[f"{side}HandMiddle1"]
        local_lateral = (
            self.node_translations[f"{side}HandIndex1"]
            - self.node_translations[f"{side}HandPinky1"]
        )
        local_basis = self._orientation_basis(local_primary, local_lateral)

        points = [self._point(point) for point in hand]
        # Selisih landmark tangan memakai koordinat relatif pergelangan; mapping
        # sumbunya sama dengan pose world, tetapi tidak mencampurkan skala pose.
        target_primary_raw = points[9] - points[0]
        target_lateral_raw = points[5] - points[17]
        target_primary = np.array([
            target_primary_raw[0], -target_primary_raw[1], -target_primary_raw[2]
        ], dtype=np.float32)
        target_lateral = np.array([
            target_lateral_raw[0], -target_lateral_raw[1], -target_lateral_raw[2]
        ], dtype=np.float32)
        target_basis = self._orientation_basis(target_primary, target_lateral)

        target_world_rotation = quaternion_from_matrix(target_basis @ local_basis.T)
        actual_local = normalize_quaternion(quaternion_multiply(
            quaternion_inverse(forearm_world_rotation), target_world_rotation
        ))
        return normalize_quaternion(quaternion_multiply(
            quaternion_inverse(self.rest_rotations[hand_name]), actual_local
        ))

    def _solve_upper_body_ik(
        self,
        pose_world: List[Dict],
        right_hand: Optional[List[Dict]],
        left_hand: Optional[List[Dict]],
        previous: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        solved: Dict[str, np.ndarray] = {}
        for side, indices, hand in (
            ("Right", (12, 14, 16), right_hand),
            ("Left", (11, 13, 15), left_hand),
        ):
            shoulder = self._pose_world_point(pose_world[indices[0]])
            elbow = self._pose_world_point(pose_world[indices[1]])
            wrist = self._pose_world_point(pose_world[indices[2]])

            shoulder_name = f"{side}Shoulder"
            arm_name = f"{side}Arm"
            forearm_name = f"{side}ForeArm"
            hand_name = f"{side}Hand"
            solved[shoulder_name] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

            shoulder_world = self.rest_world_rotations[shoulder_name]
            arm_delta, arm_world = self._solve_bone_direction(
                arm_name, forearm_name, elbow - shoulder, shoulder_world
            )
            forearm_delta, forearm_world = self._solve_bone_direction(
                forearm_name, hand_name, wrist - elbow, arm_world
            )
            solved[arm_name] = arm_delta
            solved[forearm_name] = forearm_delta
            if hand and len(hand) == 21:
                hand_delta = self._solve_hand_orientation(side, hand, forearm_world)
                previous_hand = previous.get(hand_name)
                # Z landmark tangan MediaPipe dapat sesaat membalik basis telapak
                # ketika kedua tangan bertumpuk. Gerakan >85 derajat per frame
                # tidak mungkin secara fisik pada video 30 FPS, jadi tahan frame
                # sebelumnya alih-alih membiarkan pergelangan terpelintir.
                if (
                    previous_hand is not None
                    and quaternion_angle_degrees(previous_hand, hand_delta) > 85.0
                ):
                    hand_delta = previous_hand
                solved[hand_name] = hand_delta
            else:
                solved[hand_name] = previous.get(
                    hand_name, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                )
        return solved

    def _controls_to_rotations(self, state: Dict[str, float]) -> Dict[str, List[float]]:
        rotations: Dict[str, np.ndarray] = {}

        for side in ("Right", "Left"):
            sign = 1.0 if side == "Right" else -1.0
            arm_up = state.get(f"{side}Arm_up", 0.0)
            arm_spread = state.get(f"{side}Arm_spread", 0.0)
            arm_roll = state.get(f"{side}Arm_roll", 0.0)
            forearm_bend = state.get(f"{side}ForeArm_bend", 0.0)
            forearm_twist = state.get(f"{side}ForeArm_twist", 0.0)
            wrist_pitch = state.get(f"{side}Hand_pitch", 0.0)
            wrist_yaw = state.get(f"{side}Hand_yaw", 0.0)
            wrist_roll = state.get(f"{side}Hand_roll", 0.0)

            if arm_up > 25.0 and forearm_bend > 70.0:
                proximity = min(1.0, (arm_up - 25.0) / 35.0) * min(1.0, (forearm_bend - 70.0) / 35.0)
                max_safe_bend = 100.0 - proximity * 12.0
                if forearm_bend > max_safe_bend:
                    overflow = forearm_bend - max_safe_bend
                    forearm_bend = max_safe_bend
                    wrist_pitch += overflow * 0.22
                if arm_spread < 10.0:
                    arm_spread += (10.0 - arm_spread) * proximity * 0.8
                arm_roll += sign * 10.0 * proximity

            shoulder_elevation = max(0.0, arm_up - 20.0) * 0.14
            shoulder_forward = max(0.0, arm_up - 20.0) * 0.08
            shoulder_parent = self._degrees_quaternion(
                -shoulder_forward, 0.0, sign * shoulder_elevation, "YXZ"
            )
            rotations[f"{side}Shoulder"] = self._parent_delta_to_local(
                f"{side}Shoulder", shoulder_parent
            )
            rotations[f"{side}Arm"] = self._degrees_quaternion(
                -arm_up, sign * arm_roll, sign * arm_spread
            )
            rotations[f"{side}ForeArm"] = self._degrees_quaternion(
                -forearm_bend, sign * forearm_twist, 0.0
            )
            rotations[f"{side}Hand"] = self._degrees_quaternion(
                -wrist_pitch, sign * wrist_roll, sign * wrist_yaw
            )

            spread_value = state.get(f"{side}_Spread", 0.0)
            finger_spread = {"Index": -0.4, "Middle": 0.0, "Ring": 0.35, "Pinky": 0.8}
            for finger, spread_factor in finger_spread.items():
                curl = state.get(f"{side}_{finger}", 0.0)
                q_spread = quaternion_from_axis_angle(
                    self.palm_normal[side], math.radians(spread_value * sign * spread_factor)
                )
                q_curl = quaternion_from_axis_angle(
                    self.knuckle_hinge[side], math.radians(-curl * 0.45)
                )
                parent_delta = normalize_quaternion(quaternion_multiply(q_spread, q_curl))
                bone_prefix = f"{side}Hand{finger}"
                rotations[f"{bone_prefix}1"] = self._parent_delta_to_local(
                    f"{bone_prefix}1", parent_delta
                )
                rotations[f"{bone_prefix}2"] = self._degrees_quaternion(-curl * 0.35, 0.0, 0.0)
                rotations[f"{bone_prefix}3"] = self._degrees_quaternion(-curl * 0.20, 0.0, 0.0)

            thumb = state.get(f"{side}_Thumb", 0.0)
            thumb_parent = quaternion_from_axis_angle(
                self.thumb_opposition[side], math.radians(thumb * 0.55)
            )
            rotations[f"{side}HandThumb1"] = self._parent_delta_to_local(
                f"{side}HandThumb1", thumb_parent
            )
            rotations[f"{side}HandThumb2"] = self._degrees_quaternion(-thumb * 0.35, 0.0, 0.0)
            rotations[f"{side}HandThumb3"] = self._degrees_quaternion(-thumb * 0.20, 0.0, 0.0)

        rotations["Head"] = self._degrees_quaternion(
            -state.get("Head_tilt", 0.0), state.get("Head_turn", 0.0), 0.0
        )
        rotations["Spine"] = self._degrees_quaternion(-state.get("Spine_bend", 0.0), 0.0, 0.0)

        return {
            name: [round(float(component), 5) for component in normalize_quaternion(quaternion)]
            for name, quaternion in rotations.items()
        }

    def process_frames(self, extracted_data: Dict) -> Dict:
        """
        Memproses seluruh frame landmark video dan menghasilkan data animasi retargeted.
        """
        raw_frames = extracted_data.get("frames", [])
        fps = extracted_data.get("fps", 25.0)
        total_frames = len(raw_frames)
        duration_ms = extracted_data.get("durationMs", 0.0)

        detection_coverage = {
            key: round(sum(bool(frame.get(key)) for frame in raw_frames) / max(1, total_frames), 4)
            for key in ("pose", "poseWorld", "leftHand", "rightHand")
        }
        prepared_frames = [dict(frame) for frame in raw_frames]
        self._fill_short_landmark_gaps(prepared_frames, "leftHand")
        self._fill_short_landmark_gaps(prepared_frames, "rightHand")
        self._stabilize_pose_depth(prepared_frames)
        depth_diagnostics = self._apply_palm_size_depth_prior(
            prepared_frames,
            float(extracted_data.get("width", 1)),
            float(extracted_data.get("height", 1)),
        )
        hand_measurements = [
            {
                **self._measure_hand_controls(frame.get("rightHand"), "Right"),
                **self._measure_hand_controls(frame.get("leftHand"), "Left"),
            }
            for frame in prepared_frames
        ]

        print(f"Memulai retargeting kinematik full upper-body pada {total_frames} frame...")
        animation_frames = []
        state: Dict[str, float] = {}
        previous_rotations: Dict[str, np.ndarray] = {}

        for frame_number, f in enumerate(prepared_frames):
            frame_idx = f["frameIndex"]
            timestamp_ms = f["timestampMs"]

            pose = f.get("pose")
            pose_world = f.get("poseWorld")
            lh = f.get("leftHand")
            rh = f.get("rightHand")

            if pose and len(pose) > 8:
                nose_x = float(pose[0].get("x", 0.0))
                ear_mid_x = (float(pose[7].get("x", 0.0)) + float(pose[8].get("x", 0.0))) * 0.5
                head_turn = self._clamp((nose_x - ear_mid_x) * 150.0, -40.0, 40.0)
                self._smooth(state, "Head_turn", head_turn, 0.30)

            # Median terpusat menghapus satu frame jari yang salah terdeteksi,
            # lalu smoothing biasa tetap mengikuti bentuk isyarat yang berubah.
            for key, measured in hand_measurements[frame_number].items():
                neighbors = [
                    hand_measurements[index][key]
                    for index in range(max(0, frame_number - 1),
                                       min(total_frames, frame_number + 2))
                    if key in hand_measurements[index]
                ]
                self._smooth(state, key, float(np.median(neighbors)), 0.42)
            target_rotations = {
                name: np.array(value, dtype=np.float32)
                for name, value in self._controls_to_rotations(state).items()
            }
            if pose_world and len(pose_world) > 16:
                target_rotations.update(self._solve_upper_body_ik(
                    pose_world, rh, lh, previous_rotations
                ))
            elif pose and len(pose) > 16:
                # Kompatibilitas data ekstraksi lama tanpa pose-world.
                self._extract_arm_controls(pose, rh, "Right", state)
                self._extract_arm_controls(pose, lh, "Left", state)
                target_rotations = {
                    name: np.array(value, dtype=np.float32)
                    for name, value in self._controls_to_rotations(state).items()
                }

            smoothed_rotations: Dict[str, np.ndarray] = {}
            for name, target in target_rotations.items():
                previous = previous_rotations.get(name)
                if previous is None:
                    smoothed_rotations[name] = normalize_quaternion(target)
                    continue

                finger_bone = "Hand" in name and name[-1:].isdigit()
                alpha = 0.48 if finger_bone else 0.58
                # Batasi kecepatan satu-frame agar noise landmark tidak tampak
                # sebagai sendi yang patah. Pergelangan perlu batas lebih ketat.
                max_step = (8.0 if finger_bone else
                            18.0 if name in ("RightHand", "LeftHand") else 25.0)
                target_angle = quaternion_angle_degrees(previous, target)
                if target_angle > max_step:
                    alpha = min(alpha, max_step / target_angle)
                smoothed_rotations[name] = quaternion_slerp(previous, target, alpha)
            previous_rotations = smoothed_rotations
            rotations = {
                name: [round(float(component), 5) for component in quaternion]
                for name, quaternion in smoothed_rotations.items()
            }

            animation_frames.append({
                "frame": frame_idx,
                "timestampMs": timestamp_ms,
                "rotations": rotations,
            })

        print(f"Retargeting selesai. Dihasilkan {len(animation_frames)} frame animasi.")

        output_motion = {
            "signId": extracted_data.get("videoFile", "unknown_sign").replace(".mp4", ""),
            "gloss": extracted_data.get("gloss", "Unknown"),
            "sourceFps": fps,
            "durationMs": duration_ms,
            "coordinateSpace": "avatar_local_quaternion",
            "skeleton": "Armature",
            "retargeterVersion": 5,
            "depthEstimate": {
                "type": "monocular_relative_heuristic_not_metric_depth",
                "palmSizePrior": depth_diagnostics,
            },
            "drivenBones": sorted(animation_frames[0]["rotations"].keys()) if animation_frames else [],
            "landmarkCoverage": detection_coverage,
            "totalFrames": len(animation_frames),
            "frames": animation_frames,
        }
        return output_motion


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="avatar/extracted_landmarks_air.json")
    parser.add_argument("--out", type=str, default="avatar/motion_air.json")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        extracted = json.load(f)

    retargeter = AvatarRetargeter()
    motion_data = retargeter.process_frames(extracted)

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(motion_data, f, indent=2)

    print(f"[OK] Motion data berhasil disimpan ke: {out_p}")
