from typing import Dict, List, Union
from math import ceil, sqrt

import numpy as np
import torch
import cv2
import pyvista as pv
import pyvistaqt as pvqt
from dreifus.pyvista import (
    add_coordinate_axes,
    add_camera_frustum,
    Pose,
    PoseType,
    Intrinsics,
    CameraCoordinateConvention,
)

from .sh import SH2RGB
from .graphics import fov2focal
from .camera import CAM_COLOR_LIST


def render_image_dict_from_cameras(gaussians, cameras, bg=None):
    """
    Render gaussians into a dictionary of images corresponding to each camera.

    Args:
        gaussians: GaussianModel object.
        cameras: List of scene.cameras.Camera objects.
        bg: Background tensor.
    """
    from gaussian_renderer import render

    if bg is None:
        bg = torch.tensor([0, 0, 0], dtype=torch.float32, device=gaussians.device)
    images_dict = {}
    for camera in cameras:
        gt_image = (
            camera.original_image.cpu().numpy().transpose(1, 2, 0) * 255
        ).astype(np.uint8)
        images_dict[str(camera.uid) + "-gt"] = gt_image
        with torch.no_grad():
            render_pkg = render(camera, gaussians, bg)
            image = render_pkg["render"]
            images_dict[str(camera.uid) + "-render"] = (
                image.cpu().numpy().transpose(1, 2, 0) * 255
            ).astype(np.uint8)
    return images_dict


def create_subplots_from_dict(
    image_dict: Dict[str, np.ndarray],
    blend: bool = False,
    image_w: int = 1920,
    image_h: int = 1080,
) -> np.ndarray:
    """
    Function to create a subplot with the images in the dictionary.
    The images are resized to fit the screen resolution.
    If blend is True, the gt images are blended with the rendered lines.
    The screen resolution is calculated based on the screen resolution and the number of columns.
    """
    assert (
        len(image_dict) % 2 == 0
    ), "The dictionary must have an even number of elements"
    ALPHA = 0.7
    screen_w = image_w
    screen_h = image_h * 0.9
    # Calculate the image size based on the screen resolution and the number of columns
    num_images = len(image_dict)
    num_cols = ceil(sqrt(num_images))
    num_rows = ceil(num_images / num_cols)
    w = int(screen_w / num_cols)
    h = int(screen_h / num_rows)
    # Accumulate images in image grid
    image_grid = []
    image_row = []
    for text, orig_image in image_dict.items():
        # If it is the gt image, alpha blend it with the rendered lines
        if blend and "gt" in text:
            cam_id = text.split("-gt")[0]
            rendered_image = image_dict[cam_id + "-render"].copy()
            non_black_positions = np.any(rendered_image != 0, axis=-1)
            rendered_image[non_black_positions] = [0, 1, 0]
            orig_image = cv2.addWeighted(
                orig_image, ALPHA, rendered_image, 1 - ALPHA, 0
            )
        image = cv2.resize(orig_image, (w, h))
        cv2.putText(
            image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1
        )
        image_row.append(image)
        if len(image_row) == num_cols:
            image_grid.append(image_row)
            image_row = []
    # Fill with black images if necessary
    remaining_cols = num_cols - len(image_row)
    if remaining_cols:
        for _ in range(remaining_cols):
            image_row.append(np.zeros((h, w, 3), dtype=np.uint8))
        image_grid.append(image_row)
    final_image = np.concatenate(
        [np.concatenate(row, axis=1) for row in image_grid], axis=0
    )
    return final_image


def get_joints_and_segments_from_hair_gs(
    gaussians,
    polydata: pv.PolyData = None,
    non_transparent: bool = False,
):
    endpoints = gaussians._endpoints.detach().cpu().numpy()
    endpoints_indices = gaussians.endpoint_pairs.detach().cpu().numpy()
    num_lines = endpoints_indices.shape[0]
    padding = np.ones((num_lines, 1), dtype=np.int32) * 2
    lines = np.hstack([padding, endpoints_indices])
    lines = lines.flatten()
    # rgba
    sh = gaussians._features_dc.transpose(1, 2).detach().cpu().numpy()
    rgb = SH2RGB(sh).squeeze()
    rgb = np.clip(rgb, 0, 1)
    if non_transparent:
        alpha = np.ones((rgb.shape[0], 1))
    else:
        alpha = gaussians.get_opacity.detach().cpu().numpy()
    rgba = np.hstack([rgb, alpha])
    if polydata is None:
        polydata = pv.PolyData(endpoints, lines=lines)
    else:
        polydata.points = endpoints
        polydata.lines = lines
    polydata["rgba"] = rgba
    return polydata


def pv_visualize(
    plotter=None,
    point_clouds: Dict[str, Union[np.ndarray, torch.Tensor]] = {},
    lines: Dict[str, Union[np.ndarray, torch.Tensor]] = {},
    title: str = "",
):
    """
    Function to visualize the point clouds and lines using pyvista.
    If plotter is not given, a new plotter is created and the visualization is shown.
    Otherwise, the visualization is added to the given plotter.
    """
    is_subplot = plotter is not None
    if not is_subplot:
        plotter = pv.Plotter()
    point_opacity = 1 if len(lines.keys()) == 0 else 0.5
    point_size = 1 if len(lines.keys()) == 0 else 3
    for color, point_cloud in point_clouds.items():
        if isinstance(point_cloud, torch.Tensor):
            plotter.add_points(
                point_cloud.detach().cpu().numpy(),
                color=color,
                point_size=point_size,
                opacity=point_opacity,
            )
        else:
            plotter.add_points(
                point_cloud, color=color, point_size=point_size, opacity=point_opacity
            )
    for color, line in lines.items():
        if isinstance(line, torch.Tensor):
            plotter.add_lines(line.detach().cpu().numpy(), color=color, width=1)
        else:
            plotter.add_lines(line, color=color, width=1)
    axes_actor = plotter.add_axes()
    axes_actor.SetXAxisLabelText("X")
    axes_actor.SetYAxisLabelText("Y")
    axes_actor.SetZAxisLabelText("Z")
    if is_subplot:
        color = "black" if plotter.background_color.name == "white" else "white"
        plotter.add_text(title, font_size=18, color=color)
    else:
        plotter.add_title(title)
        plotter.render()
        plotter.show()


def create_pv_background_plotter(
    gaussians, cameras
) -> (pv.PolyData, pvqt.BackgroundPlotter):
    """
    Create a pyvista plotter with the background mesh and the cameras.

    Args:
        gaussians: HairGaussianModel object.
        cameras: List of scene.cameras.Camera objects.

    Returns:
        plotter: BackgroundPlotter object.
        poly_data: PolyData object.
    """
    # Create plotter and add mesh
    poly_data = get_joints_and_segments_from_hair_gs(gaussians)
    plotter = pvqt.BackgroundPlotter()
    plotter.add_mesh(poly_data, show_scalar_bar=False, rgba=True)
    # Add cameras
    add_coordinate_axes(plotter, scale=0.1, draw_labels=False)
    for i, cam in enumerate(cameras):
        pose = np.eye(4)
        pose[:3, :3] = cam.R
        pose[:3, 3] = cam.T
        k = np.eye(3)
        k[0, 0] = fov2focal(cam.FoVx, cam.image_width)
        k[1, 1] = fov2focal(cam.FoVy, cam.image_height)
        k[0, 2] = cam.image_width / 2
        k[1, 2] = cam.image_height / 2
        E = Pose(
            pose,
            pose_type=PoseType.WORLD_2_CAM,
            camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
        )
        K = Intrinsics(k)
        color = CAM_COLOR_LIST[i - 1] if i - 1 < len(CAM_COLOR_LIST) else "lightgray"
        add_camera_frustum(plotter, E, K, color=color)
    axes_actor = plotter.add_axes()
    axes_actor.SetXAxisLabelText("X")
    axes_actor.SetYAxisLabelText("Y")
    axes_actor.SetZAxisLabelText("Z")
    plotter.view_isometric()

    # Add key event to switch between camera viewpoints
    def wrapped_change_viewpoint(cam_id: int):
        def _change_viewpoint():
            W2C = np.eye(4)
            W2C[:3, :3] = cameras[cam_id].R
            W2C[:3, 3] = cameras[cam_id].T
            C2W = np.linalg.inv(W2C)
            plotter.camera.SetPosition(C2W[:3, 3])
            view_direction = np.array([0, 0, 1])
            view_direction = C2W[:3, :3] @ view_direction
            plotter.camera.SetFocalPoint(view_direction)
            up_vector = np.array([0, -1, 0])
            up_vector = C2W[:3, :3] @ up_vector
            plotter.camera.SetViewUp(up_vector)
            fov = 180 * cameras[cam_id].FoVx / np.pi
            plotter.camera.SetViewAngle(fov)

        return _change_viewpoint

    for i in range(len(cameras)):
        plotter.add_key_event(str(i + 1), wrapped_change_viewpoint(i))

    return plotter, poly_data


def orientation_map_to_vis(orientation_map, confidence_map) -> np.ndarray:
    """
    Given orientation map of shape (H,W) where each pixel contains a theta value in radians,
    create a rgb image where the color represents the orientation.
    """

    if isinstance(orientation_map, torch.Tensor):
        orientation_map = orientation_map.squeeze().detach().cpu().numpy()
    if isinstance(confidence_map, torch.Tensor):
        confidence_map = confidence_map.detach().cpu().numpy()
    confidence_mask = confidence_map == 1.0
    vis_h = 180 * orientation_map / np.pi
    vis_s = 255 * np.ones_like(orientation_map)
    vis_v = 255 * np.ones_like(orientation_map)
    vis_hsv = np.stack([vis_h, vis_s, vis_v], axis=2)
    vis_hsv = vis_hsv.astype(np.uint8)
    vis_rgb = cv2.cvtColor(vis_hsv, cv2.COLOR_HSV2RGB)
    vis_rgb[confidence_mask] = 0
    return vis_rgb
