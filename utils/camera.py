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
from typing import Dict
import collections

import numpy as np
import cv2
import torch
import pyrr
from scipy.spatial.transform import Rotation
import pyvista as pv
from dreifus.pyvista import (
    add_floor,
    add_coordinate_axes,
    add_camera_frustum,
    Pose,
    PoseType,
    Intrinsics,
    CameraCoordinateConvention,
)

from .graphics import focal2fov

WARNED = False

CAM_COLOR_LIST = ["red", "green", "blue", "cyan", "magenta", "yellow", "white", "black"]

ColmapCamera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)


def generate_cameras(
    number_cameras: int,
    height: int,
    width: int,
    cam_pose: np.ndarray = np.eye(4),
    anchor_pos: np.ndarray = np.array([0, 0, 0]),
    offset: float = 0.5,
    rotation_axis: str = "y",
):
    """
    Function used to generate a set of viewing cameras around the anchor point.

    Args:
        number_cameras: Number of cameras to generate.
        height: Resolution height of the camera.
        width: Resolution width of the camera.
        cam_pose: Camera pose.
        anchor_pos: Anchor point.
        offset: Offset of the camera.
        rotation_axis: Rotation axis.
    """
    num_cameras_full_circle = number_cameras - 1
    cameras = {}
    Es = {}
    focal_length_px = 500
    # Rotate camera around the anchor point with respect to the rotation axis
    for i in range(num_cameras_full_circle):
        curr_pose = cam_pose.copy()
        rot_angle_rad = 2 * np.pi * (i / num_cameras_full_circle)
        curr_pose[:3, 3] -= anchor_pos
        rot = Rotation.from_euler(rotation_axis, rot_angle_rad)
        transform = np.eye(4)
        transform[:3, :3] = rot.as_matrix()
        curr_pose = transform @ curr_pose
        curr_pose[:3, 3] += anchor_pos
        Es[i + 1] = np.linalg.inv(curr_pose)  # w2c
        camera = ColmapCamera(
            id=i + 1,
            model="SIMPLE_PINHOLE",
            width=width,
            height=height,
            params=[focal_length_px, width / 2, height / 2],
        )
        cameras[camera.id] = camera
    # Add an extra top view camera
    curr_pose = cam_pose.copy()
    curr_pose[:3, 3] = anchor_pos + np.array([0, offset, 0])
    rot = Rotation.from_euler("x", 3 * np.pi / 2)
    rot = rot.as_matrix()
    curr_pose[:3, :3] = rot @ curr_pose[:3, :3]
    Es[number_cameras] = np.linalg.inv(curr_pose)  # w2c
    camera = ColmapCamera(
        id=number_cameras,
        model="SIMPLE_PINHOLE",
        width=width,
        height=height,
        params=[focal_length_px, width / 2, height / 2],
    )
    cameras[camera.id] = camera
    return cameras, Es


def project_opencv(
    camera: ColmapCamera, E: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """
    Project 3D points to 2D image plane given the camera intrinsics and extrinsics, following OpenCV conventions.

    Args:
        camera: Camera object with intrinsics parameters.
        E: Camera extrinsics.
        points: 3D points to project.

    Returns:
        points: points projected to the image plane.
    """
    # get K
    K = np.eye(3)
    K[0, 0] = camera.params[0]
    K[1, 1] = camera.params[0]
    K[0, 2] = camera.params[1]
    K[1, 2] = camera.params[2]

    rvect = cv2.Rodrigues(E[:3, :3])[0]
    points_, _ = cv2.projectPoints(points, rvect, E[:3, 3], K, None)
    points = points_.reshape(-1, 2)
    points = points.astype(np.int16)
    return points


def _ndc2Pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5


def project_opengl(points_world: torch.Tensor, camera) -> (torch.Tensor, torch.Tensor):
    """
    Project 3D points to pixel space based on OpenGL conventions

    Args:
        points_world: 3D points in world coordinates.
        camera: scene.cameras.Camera object.

    Returns:
        points_pixel: points projected to the image plane.
        visible_points: boolean mask indicating which points are visible.
    """
    ones = torch.ones(points_world.shape[0], 1, device=points_world.device)
    points_world_homo = torch.cat([points_world, ones], dim=1)
    points_view = points_world_homo @ camera.world_view_transform  # (N, 4)

    # Projection to pixel space
    eps = 1e-7
    points_clip = points_world_homo @ camera.full_proj_transform  # (N, 4)
    points_ndc = points_clip[:, :3] / (points_clip[:, 3] + eps).unsqueeze(1)

    # Simplified frustum culling (adapted to same culling used in GS rasterization). Near plane set at 0.2
    visible_points = torch.logical_and(
        torch.logical_and(points_view[:, 2] > 0.2, torch.all(points_ndc <= 1, dim=1)),
        torch.all(points_ndc >= -1, dim=1),
    )

    points_pixel = points_ndc
    points_pixel[:, 0] = _ndc2Pix(points_pixel[:, 0], camera.image_width)
    points_pixel[:, 1] = _ndc2Pix(points_pixel[:, 1], camera.image_height)
    return points_pixel, visible_points


def plot_cameras(cameras: Dict[int, ColmapCamera], Es: Dict[int, np.ndarray]) -> None:
    """
    Plot the cameras in a pyvista plotter.

    Args:
        cameras: Dictionary of ColmapCameras.
        Es: Dictionary of camera extrinsics.
    """
    p = pv.Plotter(notebook=False)
    add_floor(p, square_size=1, max_distance=1)
    add_coordinate_axes(p, scale=0.1, draw_labels=False)
    for cam_id, cam in cameras.items():
        pose = Es[cam_id]
        k = np.eye(3)
        k[0, 0] = cam.params[0]
        k[1, 1] = cam.params[0]
        k[0, 2] = cam.params[1]
        k[1, 2] = cam.params[2]
        E = Pose(
            pose,
            pose_type=PoseType.WORLD_2_CAM,
            camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
        )
        K = Intrinsics(k)
        color = (
            CAM_COLOR_LIST[cam_id - 1]
            if cam_id - 1 < len(CAM_COLOR_LIST)
            else "lightgray"
        )
        add_camera_frustum(p, E, K, color=color)
    axes_actor = p.add_axes()
    axes_actor.SetXAxisLabelText("X")
    axes_actor.SetYAxisLabelText("Y")
    axes_actor.SetZAxisLabelText("Z")
    p.show()


def colmap_camera_to_projection_matrix(
    cam: ColmapCamera,
    w: float = None,
    h: float = None,
    znear: float = 0.01,
    zfar: float = 5,
) -> np.array:
    """
    Convert a ColmapCamera object to a perspective projection matrix.

    Args:
        cam: ColmapCamera object.
        w: Width of the image.
        h: Height of the image.
        znear: Near plane distance.
        zfar: Far plane distance.

    Returns:
        projection: Perspective projection matrix.
    """
    fy = cam.params[0]
    cx = cam.params[1]
    cy = cam.params[2]
    if cam.model != "SIMPLE_PINHOLE":
        fy = cam.params[1]
        cx = cam.params[2]
        cy = cam.params[3]
    if w is None:
        w = cx * 2
    if h is None:
        h = cy * 2
    fov_y = focal2fov(fy, h)
    fov_y_deg = np.rad2deg(fov_y)
    projection = pyrr.matrix44.create_perspective_projection(
        fov_y_deg, w / h, znear, zfar
    )
    projection = (
        projection.T
    )  # pyrr gives matrix in column-major order, we need row-major
    return projection


def opencv_to_opengl_view_matrix(w2c: np.ndarray) -> np.ndarray:
    """
    Convert a world-to-camera (OpenCV) matrix to an OpenGL view matrix.

    Args:
        w2c: World-to-camera matrix.

    Returns:
        view: OpenGL view matrix.
    """
    E = Pose(
        w2c,
        pose_type=PoseType.WORLD_2_CAM,
        camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
    )
    E = E.change_pose_type(PoseType.CAM_2_WORLD)
    E = E.change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_GL)
    E = E.change_pose_type(PoseType.WORLD_2_CAM)
    view = E.numpy()
    return view
