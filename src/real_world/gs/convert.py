import numpy as np
from io import BytesIO


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z], dtype=np.float32)


def rot_mat_to_quat(rot_mat):
    w = np.sqrt(1 + rot_mat[0, 0] + rot_mat[1, 1] + rot_mat[2, 2]) / 2
    x = (rot_mat[2, 1] - rot_mat[1, 2]) / (4 * w)
    y = (rot_mat[0, 2] - rot_mat[2, 0]) / (4 * w)
    z = (rot_mat[1, 0] - rot_mat[0, 1]) / (4 * w)
    return np.array([w, x, y, z], dtype=np.float32)


def save_to_splat(pts, colors, scales, quats, opacities, output_file):
    pts_mean = np.mean(pts, axis=0)
    pts = pts - pts_mean
    buffer = BytesIO()
    for (v, c, s, q, o) in zip(pts, colors, scales, quats, opacities):
        position = np.array([v[0], v[1], v[2]], dtype=np.float32)
        scales = np.array([s[0], s[1], s[2]], dtype=np.float32)
        rot = np.array([q[0], q[1], q[2], q[3]], dtype=np.float32)
        # SH_C0 = 0.28209479177387814
        # color = np.array([0.5 + SH_C0 * c[0], 0.5 + SH_C0 * c[1], 0.5 + SH_C0 * c[2], o[0]])
        color = np.array([c[0], c[1], c[2], o[0]])

        # rotate around x axis
        rot_x_90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        rot_x_90 = np.linalg.inv(rot_x_90)
        position = np.dot(rot_x_90, position)
        rot = quat_mult(rot_mat_to_quat(rot_x_90), rot)

        buffer.write(position.tobytes())
        buffer.write(scales.tobytes())
        buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
        buffer.write(
            ((rot / np.linalg.norm(rot)) * 128 + 128)
            .clip(0, 255)
            .astype(np.uint8)
            .tobytes()
        )
    with open(output_file, "wb") as f:
        f.write(buffer.getvalue())
