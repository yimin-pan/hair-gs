from math import exp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pytorch3d.ops import knn_points

from gaussian_renderer import render
from scene.cameras import Camera
from scene.hair_gaussian_model import HairGaussianModel, GaussianModel
from c_utils import filter_strand_list_segments


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    def _ssim(img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel)
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
            - mu1_mu2
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def bidirectional_angle_difference(angle1, angle2):
    """
    Compute the bidirectional angle difference between two angles, i.e. min(diff(angle1, angle2), diff(angle1, angle2 + pi)).

    Args:
        angle1: The first angle.
        angle2: The second angle.

    Returns:
        The minimum difference.
    """
    pi_2 = np.pi / 2
    if isinstance(angle1, torch.Tensor):
        bidirectional_diff = pi_2 - torch.abs(torch.abs(angle1 - angle2) - pi_2)
    else:
        bidirectional_diff = pi_2 - np.abs(np.abs(angle1 - angle2) - pi_2)
    return bidirectional_diff


def strand_joints_magnet_loss(gaussians: HairGaussianModel):
    """
    This loss encourages adjacent joints of the strands to be close to each other (if the rendered result is still similar, e.g. a line),
    This will allows more merging and reduce unnecessary joints.

    Args:
        gaussians: The HairGaussianModel object.

    Returns:
        The loss value.
    """
    segments = gaussians.endpoint_pairs
    u, c = torch.unique(segments, return_counts=True)
    strand_endpoint_ids = u[c == 1]
    strand_endpoint_complementary_ids, _ = gaussians.get_complementary_endpoint_idx(
        strand_endpoint_ids
    )
    endpoint_mapping = torch.zeros(
        gaussians._endpoints.shape[0], device=gaussians.device, dtype=torch.long
    )
    endpoint_mapping[strand_endpoint_ids] = strand_endpoint_complementary_ids
    # Exclude collapsed strand
    detached_endpoints = gaussians._endpoints.detach()
    self_direction = (
        detached_endpoints[strand_endpoint_ids]
        - detached_endpoints[strand_endpoint_complementary_ids]
    )
    norm = torch.norm(self_direction, dim=1)
    valid_mask = norm > gaussians.min_val
    self_direction = self_direction[valid_mask]
    strand_endpoint_ids = strand_endpoint_ids[valid_mask]
    strand_endpoint_complementary_ids = strand_endpoint_complementary_ids[valid_mask]
    # Compute nearest neighbors
    strand_endpoints = gaussians._endpoints[strand_endpoint_ids]
    sq_dists, nn_idx, _ = knn_points(
        strand_endpoints.unsqueeze(0),
        strand_endpoints.unsqueeze(0),
        return_sorted=True,
        K=3,
    )
    sq_dists = sq_dists.squeeze(0)
    nn_idx = nn_idx.squeeze(0)
    # Choose the NN from top 3 that is not self or complementary
    self_idx = torch.arange(strand_endpoints.shape[0], device=gaussians.device)
    second_nn_valid = torch.logical_and(
        nn_idx[:, 1] != self_idx, nn_idx[:, 1] != strand_endpoint_complementary_ids
    )
    sq_dists = torch.where(second_nn_valid, sq_dists[:, 1], sq_dists[:, 2])
    nn_idx = torch.where(second_nn_valid, nn_idx[:, 1], nn_idx[:, 2])
    # Compute angle difference with filtering of invalid directions. Angles should only scale the distance but not contribute to backpropagation
    self_dir_norm = torch.norm(self_direction, dim=1, keepdim=True)
    self_dir_mask = self_dir_norm > gaussians.min_val
    self_direction = self_direction / self_dir_norm
    nn_complementary_idx = endpoint_mapping[nn_idx]
    nn_direction = detached_endpoints[nn_idx] - detached_endpoints[nn_complementary_idx]
    nn_dir_norm = torch.norm(nn_direction, dim=1, keepdim=True)
    nn_dir_mask = nn_dir_norm > gaussians.min_val
    nn_direction = nn_direction / nn_dir_norm
    final_mask = torch.logical_and(self_dir_mask, nn_dir_mask).squeeze()
    self_direction = self_direction[final_mask]
    nn_direction = nn_direction[final_mask]
    dot_prod = (self_direction * nn_direction).sum(dim=1)
    dot_prod = torch.clamp(dot_prod, -1, 1)
    sq_dists = sq_dists[final_mask]
    dists = sq_dists * sq_dists
    loss = torch.mean(dists)
    return loss


def angle_smoothness_loss(
    gaussians: HairGaussianModel,
    threshold: float = 30,
    eps: float = 1e-6,
):
    """
    This loss mitigates the formation of unnatural sharp angles in the strands.

    Args:
        gaussians: The HairGaussianModel object.
        threshold (float, optional): _description_. Defaults to 30.
        eps (float, optional): _description_. Defaults to 1e-6.

    Returns:
        The loss value.
    """
    threshold_rad = threshold * np.pi / 180
    angle_sim_th = np.cos(threshold_rad)
    loss = 0
    strands_list = gaussians.strands_info.list_strands
    indices = filter_strand_list_segments(
        strands_list
    )  # filter out strands with less than 2 segments
    if len(indices) > 0:
        indices = np.array(indices)
        indices = torch.tensor(indices, device=gaussians.device, dtype=torch.long)
        segment_pair_positions = gaussians._endpoints[indices]  # (N, 2, 2, 3)
        segment_pair_directions = (
            segment_pair_positions[:, :, 1] - segment_pair_positions[:, :, 0]
        )  # (N, 2, 3)
        segment_pair_directions = segment_pair_directions / torch.norm(
            segment_pair_directions, dim=2, keepdim=True
        )
        segment_pair_directions_dot = torch.sum(
            segment_pair_directions[:, 0] * segment_pair_directions[:, 1], dim=1
        )  # (N,)
        # filter out similar directions by threshold
        segment_pair_directions_dot = segment_pair_directions_dot[
            segment_pair_directions_dot <= angle_sim_th
        ]
        if segment_pair_directions_dot.shape[0] > 0:
            segment_pair_directions_dot = torch.clamp(
                segment_pair_directions_dot, -1 + eps, 1 - eps
            )
            angle_diff = torch.acos(segment_pair_directions_dot)
            loss = torch.mean(angle_diff**2)
    return loss


def orientation_loss_rast(
    gaussians: GaussianModel,
    camera: Camera,
    args,
    bg: torch.tensor = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
):
    """
    This loss encourages the orientation of the strands to be similar to the ground truth.
    A strand mask is applied to the rendered orientation image to only consider the pixels belonging to the hair.
    For each pixel, bidirectional angle difference is computed.

    Args:
        gaussians (GaussianModel): The GaussianModel object.
        camera (Camera): the camera from where rasterization is performed.
        args (Args): args used for the gaussian renderer.
        bg (torch.tensor, optional): The background color. Default is black.

    Returns:
        The loss value.

    """
    # Render world space orientation map
    orientation_world = gaussians.get_orientation
    orientation_map_world = render(
        camera, gaussians, bg, override_color=orientation_world
    )["render"]
    # World to pixel coordinate
    orientation_map_world = orientation_map_world.permute(1, 2, 0)  # (H, W, 3)
    orientation_map_world_ = orientation_map_world.flatten(
        start_dim=0, end_dim=1
    )  # (H*W, 3)
    orientation_map_view_ = orientation_map_world_ @ camera.world_view_transform[:3, :3]
    orientation_map_pixel_ = orientation_map_view_[
        :, :2
    ]  # (H*W, 2) ommit z not relevant for orientation
    orientation_map_pixel_ = orientation_map_pixel_ / (
        torch.norm(orientation_map_pixel_, dim=1, keepdim=True) + gaussians.min_val
    )
    # Compute thetas in [0, pi) wrt to y-axis clockwise
    x = orientation_map_pixel_[:, 0]
    y = orientation_map_pixel_[:, 1]
    y = torch.where(y < gaussians.min_val, y + gaussians.min_val, y)
    thetas = torch.atan2(x, y)
    thetas = torch.where(thetas < 0, thetas + np.pi, thetas)
    # Convert back to orientation map with thetas
    h, w = orientation_map_world.shape[:2]
    orientation_map = thetas.reshape(h, w)
    # Compute confidence weighted orientation difference
    gt_orientation_map = camera.orientation_field
    confidence = camera.orientation_confidence
    # Filter background
    mask = (
        torch.any(orientation_map_world != bg, dim=2)
        if camera.mask is None
        else camera.mask
    )
    confidence = confidence[mask]
    orientation_map = orientation_map[mask]
    gt_orientation_map = gt_orientation_map[mask]
    orientation_diff = bidirectional_angle_difference(
        orientation_map, gt_orientation_map
    )
    # Orientation_diff = orientation_diff / (np.pi / 2)
    weighted_diff = orientation_diff * confidence
    loss = weighted_diff.mean()
    return loss


def mask_loss_rast(
    gaussians: GaussianModel,
    camera: Camera,
    args,
    bg: torch.tensor = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
):
    """
    This loss encourages the mask of the strands to be similar to the ground truth.
    A mask image is rasterized based on the mask value of each Gaussian.

    Args:
        gaussians (GaussianModel): The GaussianModel object.
        camera (Camera): the camera from where rasterization is performed.
        args (Args): args used for the gaussian renderer.
        bg (torch.tensor, optional): The background color. Default is black.

    Returns:
        The loss value.
    """
    masks = gaussians.get_mask.repeat(1, 3)
    rendered_mask = render(camera, gaussians, bg, override_color=masks)["render"][0]
    gt_mask = camera.float_mask
    bce = nn.BCEWithLogitsLoss()
    loss = bce(rendered_mask, gt_mask)
    return loss


def loss_function(gaussians, image, viewpoint_cam: Camera, args):
    """
    This function computes the set of losses based on args, i.e. if the corresponding weight is not 0.

    Args:
        gaussians (GaussianModel): The GaussianModel object.
        image (torch.tensor): The rendered image.
        viewpoint_cam (Camera): the camera from where rasterization is performed.
        args (Args): args used for the gaussian renderer.

    Returns:
        The loss value.
    """
    gt_image = viewpoint_cam.original_image
    loss_dict = {}
    loss = 0
    # Image-space losses
    loss_dict["l1"] = l1_loss(image, gt_image)
    loss += max(0, 1.0 - args.lambda_dssim) * loss_dict["l1"]
    loss_dict["dssim"] = 1.0 - ssim(image, gt_image)
    loss += args.lambda_dssim * loss_dict["dssim"]
    # Mask and orientation losses
    if args.lambda_mask > 0 and viewpoint_cam.mask is not None:
        loss_dict["mask"] = mask_loss_rast(gaussians, viewpoint_cam, args)
        loss += args.lambda_mask * loss_dict["mask"]
    if args.lambda_orientation > 0:
        loss_dict["orientation"] = orientation_loss_rast(gaussians, viewpoint_cam, args)
        loss += args.lambda_orientation * loss_dict["orientation"]
    # Strand specific losses
    if isinstance(gaussians, HairGaussianModel):
        if args.lambda_smooth > 0:
            loss_dict["smooth"] = angle_smoothness_loss(gaussians)
            loss += args.lambda_smooth * loss_dict["smooth"]
        if args.lambda_magnet > 0:
            loss_dict["magnet"] = strand_joints_magnet_loss(gaussians)
            loss += args.lambda_magnet * loss_dict["magnet"]
    return loss, loss_dict
