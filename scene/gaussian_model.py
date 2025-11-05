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
import os

import torch
import numpy as np
from torch import nn
from torch.distributions import Normal
from plyfile import PlyData, PlyElement

from simple_knn._C import distCUDA2
from utils import (
    build_scaling_rotation,
    build_rotation,
    mkdir_p,
    RGB2SH,
    BasicPointCloud,
    inverse_sigmoid,
    get_expon_lr_func,
    strip_symmetric,
    TrainingInfo,
)


class GaussianModel:
    min_val = 1e-7
    dist_to_scale_factor = 0.5102133812190369  # computed from pval = 0.05
    pval = 0.05
    opacity_th = 0.005
    foreground_binarization_th = 0.25

    def __init__(
        self, sh_degree: int = 3, spatial_lr_scale: float = 1.0, device: str = "cuda"
    ):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._mask = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = spatial_lr_scale
        self.device = device
        self.setup_functions()

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.mask_activation = torch.sigmoid
        self.inverse_mask_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._mask,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._mask,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_mask(self):
        return self.mask_activation(self._mask)

    @property
    def get_orientation(self):
        scale = self.get_scaling
        rots = build_rotation(self._rotation)
        main_axis_idx = torch.argmax(scale, dim=1)
        main_axis = torch.zeros_like(scale)
        main_axis[torch.arange(scale.shape[0], device=self.device), main_axis_idx] = 1
        rotated_main_axis = torch.bmm(rots, main_axis.unsqueeze(2)).squeeze(-1)
        return rotated_main_axis

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud):
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device
            )
        )
        masks = inverse_sigmoid(
            0.5
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device
            )
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._mask = nn.Parameter(masks.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

    def training_setup(self, training_args):
        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1), device=self.device
        )
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {"params": [self._mask], "lr": training_args.mask_lr, "name": "mask"},
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )
        self.set_pval(training_args.pval)
        self.training_args = training_args

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        l.append("mask")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        masks = self._mask.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, masks, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        masks = np.asarray(plydata.elements[0]["mask"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device=self.device).requires_grad_(
                True
            )
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device=self.device)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device=self.device)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(
                opacities, dtype=torch.float, device=self.device
            ).requires_grad_(True)
        )
        self._mask = nn.Parameter(
            torch.tensor(masks, dtype=torch.float, device=self.device).requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device=self.device).requires_grad_(
                True
            )
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device=self.device).requires_grad_(
                True
            )
        )
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

        self.active_sh_degree = self.max_sh_degree

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        keep_mask = ~mask
        optimizable_tensors = self._prune_optimizer(keep_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._mask = optimizable_tensors["mask"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[keep_mask]
        self.denom = self.denom[keep_mask]
        self.max_radii2D = self.max_radii2D[keep_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_mask,
        new_scaling,
        new_rotation,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "mask": new_mask,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._mask = optimizable_tensors["mask"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1), device=self.device
        )
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

    def densify_and_split(
        self,
        grads,
        grad_threshold,
        scene_extent,
        N=2,
        training_info: TrainingInfo = None,
    ):
        split_threshold = self.training_args.percent_dense * scene_extent
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > split_threshold,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_mask = self._mask[selected_pts_mask].repeat(N, 1)

        if training_info is not None:
            training_info.densification_info["split"] = int(selected_pts_mask.sum())

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_mask,
            new_scaling,
            new_rotation,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(
                    N * selected_pts_mask.sum(), device=self.device, dtype=bool
                ),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(
        self, grads, grad_threshold, scene_extent, training_info: TrainingInfo = None
    ):
        split_threshold = self.training_args.percent_dense * scene_extent
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= split_threshold,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_mask = self._mask[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        if training_info is not None:
            training_info.densification_info["clone"] = int(selected_pts_mask.sum())

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_mask,
            new_scaling,
            new_rotation,
        )

    def densification(
        self, extent, max_screen_size, training_info: TrainingInfo = None
    ):
        max_grad = self.training_args.densify_grad_threshold
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, training_info=training_info)
        self.densify_and_split(grads, max_grad, extent, training_info=training_info)

        prune_mask = (self.get_opacity < self.opacity_th).squeeze()

        if training_info is not None:
            training_info.densification_info["prune_low_opacity"] = int(
                prune_mask.sum()
            )

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )

            if training_info is not None:
                # training_info.densification_info["pruned_big_vs_points"] = int(big_points_vs.sum())
                training_info.densification_info["prune_big_ws"] = int(
                    big_points_ws.sum()
                )

        if training_info is not None:
            training_info.densification_info["prune_total"] = int(prune_mask.sum())

        if prune_mask.sum() != self.get_xyz.shape[0]:
            self.prune_points(prune_mask)

        if self.device == "cuda":
            torch.cuda.empty_cache()

    def update_densification_stats(self, viewspace_point_tensor, radii, update_filter):
        self.max_radii2D[update_filter] = torch.max(
            self.max_radii2D[update_filter], radii[update_filter]
        )
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    ################################### NEW FUNCTIONS ###############################

    def set_dist_to_scale_factor(self, dist_to_scale_factor):
        # pval = 2 * (1 - cdf(1 / dist_to_scale_factor))
        if not torch.is_tensor(dist_to_scale_factor):
            dist_to_scale_factor = torch.tensor(dist_to_scale_factor)
        self.dist_to_scale_factor = dist_to_scale_factor
        normal_ = Normal(loc=0, scale=1)
        x = 1 / self.dist_to_scale_factor
        p_hat = normal_.cdf(x)
        self.pval = 2 * (1 - p_hat)

    def set_pval(self, pval):
        # dist_to_scale_factor = 1 / icdf(1 - pval/2)
        if not torch.is_tensor(pval):
            pval = torch.tensor(pval)
        self.pval = pval
        normal_ = Normal(loc=0, scale=1)
        p_hat = 1 - pval / 2
        x = normal_.icdf(p_hat).item()
        self.dist_to_scale_factor = 1 / x

    def get_segment_endpoint(self):
        """
        Compute the endpoints of the segment which is defined by the center point, the scaling, rotation and a p-value
        returns: tensor of shape (N, 2, 3)
        """
        center = self.get_xyz
        scale = self.get_scaling
        num_points = center.shape[0]
        rows_idx = torch.arange(num_points, device=self.device)
        main_axis = torch.zeros((num_points, 3), device=self.device)
        axis_idx = torch.argmax(scale, dim=1)
        main_axis[rows_idx, axis_idx] = 1
        scale_to_dist_factor = 1 / self.dist_to_scale_factor
        scaled_axis = main_axis * scale
        dist = scaled_axis * scale_to_dist_factor
        rots = build_rotation(self._rotation)  # N, 3, 3
        rotated_dist = torch.bmm(rots, dist.unsqueeze(2)).squeeze(-1)
        endpoints0 = center + rotated_dist
        endpoints1 = center - rotated_dist
        return torch.stack((endpoints0, endpoints1), dim=1)

    def compute_foreground_mask(self, lines_only: bool = False):
        """
        Compute a mask of gaussians that are in the foreground based on opacity, mask value and binarization threshold.
        """
        non_transparent_mask = (self.get_opacity >= self.opacity_th).squeeze(1)
        foreground_mask = (self.get_mask >= self.foreground_binarization_th).squeeze(1)
        mask = torch.logical_and(non_transparent_mask, foreground_mask)

        if lines_only:
            factor_threshold = 5
            eps = 1e-1
            radius_threshold = 2.5e-5
            scales = self.get_scaling
            scale_threshold = radius_threshold * self.dist_to_scale_factor
            x_line = torch.logical_and(
                scales[:, 0] / scales[:, 1] > factor_threshold,
                scales[:, 0] / scales[:, 2] > factor_threshold,
            )
            x_line = torch.logical_and(
                x_line,
                torch.logical_or(
                    scales[:, 1] / scales[:, 2] > 1 - eps,
                    scales[:, 1] / scales[:, 2] < 1 + eps,
                ),
            )
            x_line = torch.logical_and(
                x_line,
                torch.logical_and(
                    scales[:, 1] <= scale_threshold, scales[:, 2] <= scale_threshold
                ),
            )
            y_line = torch.logical_and(
                scales[:, 1] / scales[:, 0] > factor_threshold,
                scales[:, 1] / scales[:, 2] > factor_threshold,
            )
            y_line = torch.logical_and(
                y_line,
                torch.logical_or(
                    scales[:, 0] / scales[:, 2] > 1 - eps,
                    scales[:, 0] / scales[:, 2] < 1 + eps,
                ),
            )
            y_line = torch.logical_and(
                y_line,
                torch.logical_and(
                    scales[:, 0] <= scale_threshold, scales[:, 2] <= scale_threshold
                ),
            )
            z_line = torch.logical_and(
                scales[:, 2] / scales[:, 0] > factor_threshold,
                scales[:, 2] / scales[:, 1] > factor_threshold,
            )
            z_line = torch.logical_and(
                z_line,
                torch.logical_or(
                    scales[:, 0] / scales[:, 1] > 1 - eps,
                    scales[:, 0] / scales[:, 1] < 1 + eps,
                ),
            )
            z_line = torch.logical_and(
                z_line,
                torch.logical_and(
                    scales[:, 0] <= scale_threshold, scales[:, 1] <= scale_threshold
                ),
            )
            line_mask = torch.logical_xor(torch.logical_xor(x_line, y_line), z_line)
            mask = torch.logical_and(mask, line_mask)

        return mask

    def to_hair_gaussian_model(self):
        """
        Function to convert the current GaussianModel to a HairGaussianModel.
        Each gaussian will generate a line segment where the main axis is defined by the two endpoints computed from the covariance matrix. Other attributes like color features and opacity are cloned.
        """
        from scene.hair_gaussian_model import (
            HairGaussianModel,
        )  #  avoid circular import

        assert isinstance(self, GaussianModel), "The model must be a GaussianModel"

        # Create a new HairGaussianModel object
        hair_gs = HairGaussianModel(
            sh_degree=self.max_sh_degree,
            spatial_lr_scale=self.spatial_lr_scale,
            device=self.device,
        )
        hair_gs.set_dist_to_scale_factor(self.dist_to_scale_factor)
        hair_gs.active_sh_degree = self.active_sh_degree
        num_points = self.get_xyz.shape[0]

        # Compute endpoint from main axis
        scale = self.get_scaling
        all_rows = torch.arange(num_points, device=self.device)
        axis_idx = torch.argmax(scale, dim=1)
        endpoints = self.get_segment_endpoint()
        endpoints = torch.cat((endpoints[:, 0], endpoints[:, 1]), dim=0)

        # Determine the width as the maximum of the remaining two scales (axes)
        other_axes = torch.ones((num_points, 3), device=self.device)
        other_axes[all_rows, axis_idx] = 0
        other_axes = scale * other_axes
        width = torch.mean(other_axes, dim=1)
        width = width.unsqueeze(1)
        width = self.scaling_inverse_activation(width)  # back to log space

        # For each point, create a segment with two endpoints (all segments disconnected)
        endpoint_pairs = torch.cat(
            (
                torch.arange(num_points).unsqueeze(1),
                torch.arange(start=num_points, end=2 * num_points).unsqueeze(1),
            ),
            dim=1,
        ).to(self.device)
        features_dc = self._features_dc.clone()
        features_rest = self._features_rest.clone()
        opacity = self._opacity.clone()
        mask = self._mask.clone()
        hair_gs._endpoints = nn.Parameter(endpoints, requires_grad=True)
        hair_gs.endpoint_pairs = endpoint_pairs
        hair_gs._features_dc = nn.Parameter(features_dc, requires_grad=True)
        hair_gs._features_rest = nn.Parameter(features_rest, requires_grad=True)
        hair_gs._opacity = nn.Parameter(opacity, requires_grad=True)
        hair_gs._mask = nn.Parameter(mask, requires_grad=True)
        hair_gs._width = nn.Parameter(width, requires_grad=True)

        # Update strand root and strands info
        hair_gs.ref_strand_root = self.ref_strand_root
        hair_gs.update_strand_root()
        hair_gs.compute_strands_info()
        hair_gs.training_setup(self.training_args)

        return hair_gs

    def clean_gaussians(self):
        prune_mask = ~self.compute_foreground_mask()
        self.prune_points(prune_mask)
