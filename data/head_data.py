"""
Class for loading head data from different synthetic datasets into a common format.
The resulting data is used for the OpenGL based renderer to generate the input images to the method.
Currently supports USC-HairSalon and Cem-Yuksel datasets.
"""

from typing import NamedTuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from pytorch3d.io import load_obj
from pytorch3d.ops.points_normals import estimate_pointcloud_normals


class HeadData(NamedTuple):
    verts: np.ndarray
    colors: np.ndarray
    normals: np.ndarray
    faces: np.ndarray


def load_head_from_usc_dataset(
    file_path: str, normal_required: bool = False
) -> HeadData:
    verts, faces, aux = load_obj(file_path)
    verts = verts.numpy()
    faces = faces.verts_idx.numpy()
    color = np.array([0.75, 0.75, 0.75, 1])
    colors = np.tile(color, (verts.shape[0], 1))
    normals = None
    if normal_required:
        normals = aux.normals.numpy()
        if normals.shape[0] == faces.shape[0]:
            faces_flat = faces.flatten()
            verts_normal = np.zeros((verts.shape[0], 3))
            verts_normal[faces_flat] = normals
            normals = verts_normal
        elif normals.shape[0] != verts.shape[0]:
            # estimate normals
            verts_ = torch.from_numpy(verts).cuda()
            verts_ = verts_.unsqueeze(0)
            normals = estimate_pointcloud_normals(verts_)[0].cpu().numpy()
    return HeadData(verts=verts, colors=colors, normals=normals, faces=faces)


def load_head_from_cy_dataset(file_path: str) -> HeadData:
    verts, faces, aux = load_obj(file_path)
    verts = verts.numpy()
    # cm to m and scale to match a realisitc head size => 0.17m in diameter
    verts = 0.25 * verts / 100
    # from z-up to y-up, mesh is already zero centered
    rot1 = Rotation.from_euler("x", -90, degrees=True)
    rot2 = Rotation.from_euler("y", -90, degrees=True)
    transform = rot2.as_matrix() @ rot1.as_matrix()
    verts = (transform @ verts.T).T

    faces = faces.verts_idx.numpy()
    # color
    color = np.array([0.75, 0.75, 0.75, 1])
    colors = np.tile(color, (verts.shape[0], 1))
    normals = aux.normals.numpy()
    if normals.shape[0] == faces.shape[0]:
        faces_flat = faces.flatten()
        verts_normal = np.zeros((verts.shape[0], 3))
        verts_normal[faces_flat] = normals
        normals = verts_normal
    elif normals.shape[0] != verts.shape[0]:
        # estimate normals
        verts_ = torch.from_numpy(verts).cuda()
        verts_ = verts_.unsqueeze(0)
        normals = estimate_pointcloud_normals(verts_)[0].cpu().numpy()
    return HeadData(verts=verts, colors=colors, normals=normals, faces=faces)


head_data_load_callbacks = {
    "usc_hair_salon": load_head_from_usc_dataset,
    "cem_yuksel": load_head_from_cy_dataset,
}
