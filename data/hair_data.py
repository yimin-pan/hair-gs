"""
Class for loading hair data from different synthetic datasets into a common format.
The resulting data is used for:
- The OpenGL based renderer to generate the input images to the method.
- Saving a npz file which is later used in (quantitative) evaluation.
Currently supports USC-HairSalon and Cem-Yuksel datasets.
"""

from typing import NamedTuple
import array

import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation
from pytorch3d.ops.points_normals import estimate_pointcloud_normals

from .cy_hair import CYHairFile


class HairData(NamedTuple):
    verts: np.ndarray
    colors: np.ndarray
    normals: np.ndarray
    edges: np.ndarray
    strand_root_idx: np.ndarray
    verts_id_to_strand_id: np.ndarray


def save_hair_eval_data_npz(file_path: str, hair_dataset: HairData):
    """
    Save the loaded dataset to a npz file for later evaluation.

    Args:
        file_path (str): The path to the npz file.
        hair_dataset (HairData): The hair dataset to save.
    """
    points = hair_dataset.verts[
        hair_dataset.edges[:, 0]
    ]  # Leave the tip of the strand as we dont have direction for it
    segment_points = hair_dataset.verts[hair_dataset.edges]
    directions = segment_points[:, 1] - segment_points[:, 0]
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    verts_id_to_strand_id = hair_dataset.verts_id_to_strand_id[hair_dataset.edges[:, 0]]
    # remove last segment (edge) of each strand
    edges = hair_dataset.edges
    mask = np.isin(edges[:, 1], edges[:, 0])
    edges = edges[mask]
    old_indices = np.unique(edges)
    new_indices = np.arange(old_indices.shape[0])
    mapping = np.zeros(old_indices.max() + 1, dtype=new_indices.dtype)
    mapping[old_indices] = new_indices
    edges = mapping[edges]
    np.savez(
        file_path,
        points=points,
        directions=directions,
        points_id_to_strand_id=verts_id_to_strand_id,
        edges=edges,
    )


def load_hair_from_usc_dataset(
    file_path: str,
    normal_required: bool = False,
    hsv_spectre_color: bool = True,
    pct_strands: float = 100,
) -> HairData:
    color_palette = np.array(
        [[0.545, 0.271, 0.075, 1], [0.639, 0.341, 0.125, 1], [0.561, 0.388, 0.196, 1]]
    )
    strands = []
    edges = []
    colors = []
    strand_root_idx = []
    verts_id_to_strand_id = []
    last_idx = 0
    with open(file_path, "rb") as file:
        num_strands = int.from_bytes(file.read(4), "little")
        strands_to_load = int(num_strands * pct_strands / 100)
        load_freq = num_strands // strands_to_load
        assert num_strands == 10000, f"Expected 10000 strands, got: {num_strands}"
        hues = np.linspace(start=0, stop=180, num=num_strands)
        for i in range(num_strands):
            num_verts = int.from_bytes(file.read(4), "little")
            assert (
                num_verts == 1 or num_verts == 100
            ), f"Num_verts should be 1 or 100, got: {num_verts}"
            # get strand data
            strand_data_xyz = array.array("f")
            strand_data_xyz.fromfile(file, 3 * num_verts)
            # skip with frequency
            if i % load_freq != 0 or num_verts == 1:
                continue
            strand_data_xyz = np.array(strand_data_xyz).reshape(-1, 3)
            strand_root_idx.append(last_idx)
            strands.append(strand_data_xyz)
            edge_c1 = np.arange(last_idx, last_idx + num_verts - 1).astype(np.uint32)
            edge_c2 = np.arange(last_idx + 1, last_idx + num_verts).astype(np.uint32)
            edges_ = np.column_stack([edge_c1, edge_c2])
            edges.append(edges_)
            last_idx += num_verts
            verts_id_to_strand_id_ = (
                (len(strands) - 1) * np.ones(num_verts, dtype=np.uint32)
            ).astype(np.uint32)
            verts_id_to_strand_id.append(verts_id_to_strand_id_)
            # add colors with some noise
            color = color_palette[i % color_palette.shape[0]]
            if hsv_spectre_color:
                hue = hues[i]
                hsv = np.uint8([[[hue, 255, 255]]])
                color = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                color = color[0, 0] / 255
                color = np.append(color, 1)
            color = np.tile(color, (num_verts, 1))
            # color += np.random.normal(0, 0.05, color.shape)
            colors.append(color)
    strands = np.concatenate(strands, axis=0)
    colors = np.concatenate(colors, axis=0)
    normals = None
    edges = np.concatenate(edges, axis=0)
    strand_root_idx = np.array(strand_root_idx)
    verts_id_to_strand_id = np.concatenate(verts_id_to_strand_id, axis=0)
    if normal_required:
        strands_ = torch.from_numpy(strands).cuda()
        strands_ = strands_.unsqueeze(0)
        normals = estimate_pointcloud_normals(strands_)
        normals = normals[0].cpu().numpy()
    return HairData(
        verts=strands,
        colors=colors,
        normals=normals,
        edges=edges,
        strand_root_idx=strand_root_idx,
        verts_id_to_strand_id=verts_id_to_strand_id,
    )


def load_hair_from_cy_dataset(
    file_path: str, hsv_spectre_color: bool = True, pct_strands: float = 100
) -> HairData:
    color_palette = np.array(
        [[1.0, 0.85, 0.47, 1], [0.76, 0.75, 0.65, 1], [0.95, 0.8, 0.53, 1]]
    )
    hf = CYHairFile()
    hf.LoadFromFile(file_path)
    all_points = np.array(hf.GetPointsArray()).reshape(-1, 3)
    num_strands = hf.GetHeader().hair_count
    strand_points = hf.GetSegmentsArray()
    if strand_points is None:
        num_joints = int(all_points.shape[0] / (3 * num_strands))
        strand_points = num_joints * np.ones(num_strands, dtype=np.int32)
    else:
        strand_points = np.array(strand_points)
    raw_colors = (
        np.array(hf.GetColorsArray()) if not hf.GetColorsArray() is None else None
    )
    strands = []
    directions = []
    edges = []
    colors = []
    strand_root_idx = []
    verts_id_to_strand_id = []
    all_points_idx = 0
    last_idx = 0
    load_freq = num_strands // int(num_strands * pct_strands / 100)
    hues = np.linspace(start=0, stop=180, num=num_strands)
    for i in range(num_strands):
        num_curr_strand_points = strand_points[i]
        start_idx = all_points_idx
        all_points_idx += num_curr_strand_points
        if i % load_freq != 0:
            continue
        strand_data_xyz = all_points[start_idx : start_idx + num_curr_strand_points]
        strand_root_idx.append(last_idx)
        curr_strand_directions = strand_data_xyz[1:] - strand_data_xyz[:-1]
        last_joint_direction = np.array([[0, 0, 1]])
        curr_strand_directions = np.concatenate(
            [curr_strand_directions, last_joint_direction], axis=0
        )
        curr_strand_directions = curr_strand_directions / np.linalg.norm(
            curr_strand_directions, axis=1, keepdims=True
        )
        directions.append(curr_strand_directions)
        strands.append(strand_data_xyz)
        edge_c1 = np.arange(last_idx, last_idx + num_curr_strand_points - 1).astype(
            np.uint32
        )
        edge_c2 = np.arange(last_idx + 1, last_idx + num_curr_strand_points).astype(
            np.uint32
        )
        edges_ = np.column_stack([edge_c1, edge_c2])
        edges.append(edges_)
        last_idx += num_curr_strand_points
        verts_id_to_strand_id_ = (
            (len(strands) - 1) * np.ones(num_curr_strand_points, dtype=np.uint32)
        ).astype(np.uint32)
        verts_id_to_strand_id.append(verts_id_to_strand_id_)
        # add colors with some noise
        if raw_colors is None or hsv_spectre_color:
            color = color_palette[i % color_palette.shape[0]]
            if hsv_spectre_color:
                hue = hues[i]
                hsv = np.uint8([[[hue, 255, 255]]])
                color = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                color = color[0, 0] / 255
                color = np.append(color, 1)
            color = np.tile(color, (num_curr_strand_points, 1))
        else:
            color = raw_colors[start_idx : start_idx + num_curr_strand_points]
        colors.append(color)
    strands = np.concatenate(strands, axis=0)
    # cm to m and scale to match a realisitc head size => 0.17m in diameter
    strands = 0.25 * strands / 100
    # from z-up to y-up, mesh is already zero centered
    rot1 = Rotation.from_euler("x", -90, degrees=True)
    rot2 = Rotation.from_euler("y", -90, degrees=True)
    transform = rot2.as_matrix() @ rot1.as_matrix()
    strands = (transform @ strands.T).T
    colors = np.concatenate(colors, axis=0)
    normals = np.concatenate(directions, axis=0)
    edges = np.concatenate(edges, axis=0)
    strand_root_idx = np.array(strand_root_idx)
    verts_id_to_strand_id = np.concatenate(verts_id_to_strand_id, axis=0)
    return HairData(
        verts=strands,
        colors=colors,
        normals=normals,
        edges=edges,
        strand_root_idx=strand_root_idx,
        verts_id_to_strand_id=verts_id_to_strand_id,
    )


hair_data_load_callbacks = {
    "usc_hair_salon": load_hair_from_usc_dataset,
    "cem_yuksel": load_hair_from_cy_dataset,
}
