# The code is modified based on original 3DGS which is under the following license:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from datetime import datetime
import sys
import random
import os

import torch
import numpy as np
from plyfile import PlyData, PlyElement


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def safe_state(silent):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(
                        x.replace(
                            "\n",
                            " [{}]\n".format(
                                str(datetime.now().strftime("%d/%m %H:%M:%S"))
                            ),
                        )
                    )
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def enable_accelerated_rasterization():
    """
    Enable distwar backward pass in the differentiable rasterizer, controlled by environment variables.
    """
    os.environ["BW_IMPLEMENTATION"] = "1"
    os.environ["BALANCE_THRESHOLD"] = "8"


def save_ply_edges(vertex_xyz, vertex_color, edges, file_path: str):
    """
    Save ply file where the Gaussian Model (Hair) is saved as polylines.

    Args:
        vertex_xyz: np.ndarray of shape (N, 3) where N is the number of vertices.
        vertex_color: np.ndarray of shape (N, 3) where N is the number of vertices.
        edges: np.ndarray of shape (M, 2) where M is the number of edges.
        file_path: str, path to the ply file.
    """
    ply_elements = []
    # create vertex element
    vertex_xyz = vertex_xyz.astype(np.float32)
    dtype = [(attribute, "float32") for attribute in ["x", "y", "z"]]
    dtype += [(attribute, "u1") for attribute in ["red", "green", "blue"]]
    attributes = np.concatenate((vertex_xyz, vertex_color), axis=1)
    elements = np.empty(vertex_xyz.shape[0], dtype=dtype)
    elements[:] = list(map(tuple, attributes))
    vertex_elem = PlyElement.describe(elements, "vertex")
    ply_elements.append(vertex_elem)
    # create edge element
    if edges is not None:
        dtype = [("vertex1", "i4"), ("vertex2", "i4")]
        elements = np.empty(edges.shape[0], dtype=dtype)
        elements[:] = list(map(tuple, edges))
        connectivity_elem = PlyElement.describe(elements, "edge")
        ply_elements.append(connectivity_elem)
    # Save
    PlyData(ply_elements).write(file_path)


def save_ply_faces(vertex_xyz, vertex_color, edges, file_path: str):
    """
    Save ply file where each line segment (A, B) is stored as a triangle primitive (A, (A+B)/2, B).
    This is for visualization purposes as 3D viewers such as meshlab does not support polyline rendering.

    Args:
        vertex_xyz: np.ndarray of shape (N, 3) where N is the number of vertices.
        vertex_color: np.ndarray of shape (N, 3) where N is the number of vertices.
        edges: np.ndarray of shape (M, 2) where M is the number of edges.
        file_path: str, path to the ply file.
    """
    ply_elements = []
    # handle midpoint
    num_points = vertex_xyz.shape[0]
    num_segments = edges.shape[0]
    segment = vertex_xyz[edges]
    midpoint = (segment[:, 0] + segment[:, 1]) / 2
    midpoint_color = (vertex_color[edges[:, 0]] + vertex_color[edges[:, 1]]) / 2
    # Create vertex element
    vertex_xyz = np.concatenate((vertex_xyz, midpoint), axis=0)
    vertex_color = np.concatenate((vertex_color, midpoint_color), axis=0)
    vertex_xyz = vertex_xyz.astype(np.float32)
    dtype = [(attribute, "float32") for attribute in ["x", "y", "z"]]
    dtype += [(attribute, "u1") for attribute in ["red", "green", "blue"]]
    attributes = np.concatenate((vertex_xyz, vertex_color), axis=1)
    elements = np.empty(vertex_xyz.shape[0], dtype=dtype)
    elements[:] = list(map(tuple, attributes))
    vertex_elem = PlyElement.describe(elements, "vertex")
    ply_elements.append(vertex_elem)
    # Create face element
    if edges is not None:
        midpoint_indices = np.arange(num_segments) + num_points
        faces = np.column_stack((edges[:, 0], midpoint_indices, edges[:, 1]))
        faces = [(list(face),) for face in faces]
        dtype = [("vertex_indices", "i4", (3,))]
        elements = np.array(faces, dtype=dtype)
        connectivity_elem = PlyElement.describe(elements, "face")
        ply_elements.append(connectivity_elem)
    # Save
    PlyData(ply_elements).write(file_path)
