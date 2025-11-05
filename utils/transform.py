import torch
import numpy as np
from scipy.spatial.transform import Rotation
from pytorch3d import transforms


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def rot_to_wxyz_quat(rot: np.ndarray) -> np.ndarray:
    rot = Rotation.from_matrix(rot)
    quat = rot.as_quat()  # x, y, z, w
    w = quat[3]
    quat[1:] = quat[0:3]
    quat[0] = w
    return quat


def cross_product_to_skew_symmetric(batched_cross_product):
    batch_size = batched_cross_product.size(0)
    device = batched_cross_product.device
    skew_symmetric_matrices = torch.zeros(
        batch_size, 3, 3, dtype=batched_cross_product.dtype, device=device
    )
    skew_symmetric_matrices[:, 0, 1] = -batched_cross_product[:, 2]
    skew_symmetric_matrices[:, 0, 2] = batched_cross_product[:, 1]
    skew_symmetric_matrices[:, 1, 2] = -batched_cross_product[:, 0]
    skew_symmetric_matrices[:, 1, 0] = -skew_symmetric_matrices[:, 0, 1]
    skew_symmetric_matrices[:, 2, 0] = -skew_symmetric_matrices[:, 0, 2]
    skew_symmetric_matrices[:, 2, 1] = -skew_symmetric_matrices[:, 1, 2]
    return skew_symmetric_matrices


def calculate_rotation_from_vectors(
    v1: torch.Tensor, v2: torch.Tensor, representation: str = "mat", eps: float = 1e-7
) -> torch.Tensor:
    """
    Calculate the rotation matrix that rotates v1 to v2.
    """
    rots = None
    v2 = v2 / torch.norm(v2, dim=1, keepdim=True)
    pairwise_dot = torch.sum(v1 * v2, dim=1)
    pairwise_dot = torch.clamp(pairwise_dot, -1 + eps, 1 - eps)

    cross = torch.cross(v1, v2, dim=1)  # Nx3
    K = cross_product_to_skew_symmetric(cross)  # Nx3x3
    Id = torch.eye(3, device=K.device).repeat(K.shape[0], 1, 1)  # Nx3x3
    R = Id + K + torch.bmm(K, K) / (1 + pairwise_dot)[:, None, None]
    if representation == "quat":
        rots = transforms.matrix_to_quaternion(R)
    return rots
