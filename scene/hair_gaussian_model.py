import os
import torch
from typing import NamedTuple

import numpy as np
from torch import nn
from pytorch3d.ops import knn_points
from scipy.spatial import cKDTree
from tqdm import tqdm
from plyfile import PlyData, PlyElement

from utils import (
    build_scaling_rotation,
    calculate_rotation_from_vectors,
    mkdir_p,
    BasicPointCloud,
    inverse_sigmoid,
    get_expon_lr_func,
    strip_symmetric,
    TrainingInfo,
)
from scene.gaussian_model import GaussianModel


class StrandsInfo(NamedTuple):
    """
    Auxiliar class to store the information about strands of the HairGaussianModel.
    """

    list_strands: (
        np.ndarray
    )  # np.ndarray of dtype=object, each of them are np.ndarray of shape (num_segments, 2)
    list_strands_segments_id: (
        np.ndarray
    )  # mapping from strand to row in gs.endpoint_pairs
    id_to_strand_id: (
        np.ndarray
    )  # mapping from gs._endpoints id to strand id in list_strands
    strand_endpoint_id_to_complementary: (
        np.ndarray
    )  # mapping from strand endpoint id to its complementary endpoint id


class HairGaussianModel(GaussianModel):
    """
    Class for Hair Gaussian Model (HairGS).
    """

    def __init__(
        self, sh_degree: int = 3, spatial_lr_scale: float = 1.0, device: str = "cuda"
    ):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.ref_strand_root = np.empty(0)
        self.strand_root_endpoint_idx = torch.empty(0)
        self.endpoint_pairs = torch.empty(0)
        self._endpoints = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._opacity = torch.empty(0)
        self._mask = torch.empty(0)
        self._width = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = spatial_lr_scale
        self.device = device
        self.setup_functions()
        self.strands_info = None  # strands_info needs to be updated before topological operations: merging, growing, etc.

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
            self.ref_strand_root,
            self.strand_root_endpoint_idx,
            self.endpoint_pairs,
            self._endpoints,
            self._features_dc,
            self._features_rest,
            self._opacity,
            self._mask,
            self._width,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self.ref_strand_root,
            self.strand_root_endpoint_idx,
            self.endpoint_pairs,
            self._endpoints,
            self._features_dc,
            self._features_rest,
            self._opacity,
            self._mask,
            self._width,
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
        pairs = self._endpoints[self.endpoint_pairs]
        diff = torch.sub(pairs[:, 1], pairs[:, 0])
        dist_x = torch.norm(diff, p=2, dim=1, keepdim=True)
        dist_x = dist_x / 2  # scale is symmetric
        scale_x = dist_x * self.dist_to_scale_factor  # dist to sigma under pval
        scale_x = torch.clamp(scale_x, min=self.min_val)
        scale_yz = self._width.repeat(1, 2)
        scale_yz = self.scaling_activation(scale_yz)
        scale = torch.cat((scale_x, scale_yz), dim=1)
        return scale

    @property
    def get_rotation(self):
        # Compute rotation that aligns v1 to v2
        pairs = self._endpoints[self.endpoint_pairs]
        rotation = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=torch.float, device=pairs.device
        ).repeat(pairs.shape[0], 1)
        v1 = torch.tensor(
            [[1.0, 0.0, 0.0]], dtype=torch.float, device=pairs.device
        ).repeat(pairs.shape[0], 1)
        v2 = torch.sub(pairs[:, 1], pairs[:, 0])
        # Exclude collaped segments
        valid_mask = torch.norm(v2, p=2, dim=1) > self.min_val
        v1 = v1[valid_mask]
        v2 = v2[valid_mask]
        valid_rotation = calculate_rotation_from_vectors(v1, v2, representation="quat")
        rotation[valid_mask] = valid_rotation
        # return self.rotation_activation(rotation) # normalization not needed as quat is computed from R
        return rotation

    @property
    def get_xyz(self):
        # Center of the segments
        endpoints = self._endpoints[self.endpoint_pairs]
        segments = torch.mean(endpoints, dim=1)
        return segments

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
        # Compute world-space normalized direction of the segments
        segments = self._endpoints[self.endpoint_pairs]
        direction_3d = segments[:, 1] - segments[:, 0]
        norm = torch.norm(direction_3d, p=2, dim=1, keepdim=True)
        non_collapsed_mask = (norm >= self.min_val).squeeze()
        normalized_direction_3d = torch.tensor(
            [[1.0, 0.0, 0.0]], dtype=torch.float, device=self.device
        ).repeat(direction_3d.shape[0], 1)
        normalized_direction_3d[non_collapsed_mask] = (
            direction_3d[non_collapsed_mask] / norm[non_collapsed_mask]
        )
        return normalized_direction_3d

    def get_covariance(self, scaling_modifier=0.5):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self.get_rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self, training_args):
        self.max_radii2D = torch.zeros(
            (self.endpoint_pairs.shape[0]), device=self.device
        )
        self.xyz_gradient_accum = torch.zeros(
            (self.endpoint_pairs.shape[0], 1), device=self.device
        )
        self.denom = torch.zeros((self.endpoint_pairs.shape[0], 1), device=self.device)

        l = [
            {
                "params": [self._endpoints],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "endpoints",
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
            {"params": [self._mask], "lr": training_args.mask_lr, "name": "mask"},
            {"params": [self._width], "lr": training_args.scaling_lr, "name": "width"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.endpoints_scheduler = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )
        self.merge_dist_th = training_args.merge_dist_th_init
        self.merge_dist_th_scheduler = get_expon_lr_func(
            lr_init=training_args.merge_dist_th_init,
            lr_final=training_args.merge_dist_th_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )
        self.merge_angle_th = training_args.merge_angle_th_init
        self.merge_angle_th_scheduler = get_expon_lr_func(
            lr_init=training_args.merge_angle_th_init,
            lr_final=training_args.merge_angle_th_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        self.set_pval(training_args.pval)
        self.training_args = training_args

        # Find the max length for segments from farthest points of foreground point cloud
        fg_endpoint_mask = torch.zeros(
            self._endpoints.shape[0], dtype=torch.bool, device=self.device
        )
        fg_mask = (self.get_mask >= self.foreground_binarization_th).squeeze()
        fg_segments = self.endpoint_pairs[fg_mask]
        fg_endpoint_mask[fg_segments.flatten()] = True
        max_point = torch.max(self._endpoints[fg_endpoint_mask], dim=0)
        min_point = torch.min(self._endpoints[fg_endpoint_mask], dim=0)
        max_strand_length = torch.norm(max_point.values - min_point.values)
        self.max_segment_length = (
            max_strand_length / self.training_args.num_points_strand
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "endpoints":
                lr = self.endpoints_scheduler(iteration)
                param_group["lr"] = lr
        # update other schedulers
        self.merge_dist_th = self.merge_dist_th_scheduler(iteration)
        self.merge_angle_th = self.merge_angle_th_scheduler(iteration)

    def construct_list_of_attributes(self):
        l = []
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        l.append("mask")
        l.append("width")
        return l

    def create_from_pcd(self, pcd: BasicPointCloud):
        raise NotImplementedError("This method is only intended for Gaussian Model")

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        # pc1: endpoints and normals
        endpoints = self._endpoints.detach().cpu().numpy()
        normals = np.zeros_like(endpoints)
        dtype_full = [
            (attribute, "f4") for attribute in ["x", "y", "z", "nx", "ny", "nz"]
        ]
        elements = np.empty(endpoints.shape[0], dtype=dtype_full)
        attributes = np.concatenate((endpoints, normals), axis=1)
        elements[:] = list(map(tuple, attributes))
        element_1 = PlyElement.describe(elements, "vertex")
        # pc2: edge connectivity
        edge_connectivity = self.endpoint_pairs.detach().cpu().numpy()
        dtype_full = [("vertex1", "i4"), ("vertex2", "i4")]
        elements = np.empty(edge_connectivity.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, edge_connectivity))
        element_2 = PlyElement.describe(elements, "edge")
        # pc3: segment features and opacities, width
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
        widths = self._width.detach().cpu().numpy()
        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]
        elements = np.empty(f_dc.shape[0], dtype=dtype_full)
        attributes = np.concatenate((f_dc, f_rest, opacities, masks, widths), axis=1)
        elements[:] = list(map(tuple, attributes))
        element_3 = PlyElement.describe(elements, "segment")
        # pc4: strand root
        strand_root_idx = self.strand_root_endpoint_idx.detach().cpu().numpy()
        dtype_full = [("strand_root_idx", "i4")]
        elements = np.empty(strand_root_idx.shape[0], dtype=dtype_full)
        elements[:] = strand_root_idx.tolist()
        element_4 = PlyElement.describe(elements, "strand_root_idx")
        # pc5: ref strand root
        dtype_full = [(attribute, "f4") for attribute in ["x", "y", "z"]]
        elements = np.empty(self.ref_strand_root.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, self.ref_strand_root))
        element_5 = PlyElement.describe(elements, "ref_strand_root")
        # save
        PlyData([element_1, element_2, element_3, element_4, element_5]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)
        assert len(plydata.elements) == 5, (
            "Ply file must have 4 elements: endpoints, edge connectivity, segments, strand root. "
            "But got {}"
        ).format(len(plydata.elements))
        # endpoints
        element1 = plydata.elements[0]
        endpoints = np.stack(
            (
                np.asarray(element1["x"]),
                np.asarray(element1["y"]),
                np.asarray(element1["z"]),
            ),
            axis=1,
        )
        # edge connectivity
        element2 = plydata.elements[1]
        endpoint_pairs = np.stack(
            (np.asarray(element2["vertex1"]), np.asarray(element2["vertex2"])), axis=1
        )
        # segments
        element3 = plydata.elements[2]
        opacities = np.asarray(element3["opacity"])[..., np.newaxis]
        masks = np.asarray(element3["mask"])[..., np.newaxis]
        widths = np.asarray(element3["width"])[..., np.newaxis]
        features_dc = np.zeros((opacities.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(element3["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(element3["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(element3["f_dc_2"])
        # strand root idx
        element4 = plydata.elements[3]
        strand_root_endpoint_idx = np.asarray(element4["strand_root_idx"])
        # ref strand root
        element5 = plydata.elements[4]
        ref_strand_root = np.stack(
            (
                np.asarray(element5["x"]),
                np.asarray(element5["y"]),
                np.asarray(element5["z"]),
            ),
            axis=1,
        )
        # getting colors features
        extra_f_names = [
            p.name for p in element3.properties if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((opacities.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(element3[attr_name])
        # Reshape color features from (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )
        # set class attributes
        self._endpoints = nn.Parameter(
            torch.tensor(
                endpoints, dtype=torch.float, device=self.device
            ).requires_grad_(True)
        )
        self.endpoint_pairs = torch.tensor(
            endpoint_pairs, dtype=torch.long, device=self.device
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
        self._width = nn.Parameter(
            torch.tensor(widths, dtype=torch.float, device=self.device).requires_grad_(
                True
            )
        )
        self.active_sh_degree = self.max_sh_degree
        self.strand_root_endpoint_idx = torch.tensor(
            strand_root_endpoint_idx, dtype=torch.long, device=self.device
        )
        self.ref_strand_root = ref_strand_root
        self.compute_strands_info()

    ########################################## OPERATION #########################################
    def _replace_tensor_in_optimizer(self, tensor, name):
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

    def _prune_tensor_in_optimizer(self, endpoints_keep_mask, segments_keep_mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            mask = (
                endpoints_keep_mask
                if group["name"] == "endpoints"
                else segments_keep_mask
            )
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = group["params"][0][mask].requires_grad_(True)
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = group["params"][0][mask].requires_grad_(True)
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _cat_tensors_in_optimizer(self, tensors_dict):
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

    def cat_segments(
        self,
        new_endpoint_pairs,
        new_endpoints,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_masks,
        new_widths,
    ):
        """
        Given new segments, features and opacities, concatenate them to the current scene.
        Entries are added to endpoint_pairs and _endpoints accordingly.

        new_segments: index pairs of new segments to add
        new_endpoints: new endpoints to add
        new_features_dc: new features_dc to add
        new_features_rest: new features_rest to add
        new_opacities: new opacities to add
        """
        # cat endpoint pairs
        self.endpoint_pairs = torch.cat(
            [self.endpoint_pairs, new_endpoint_pairs], dim=0
        )
        # cat segment attributes
        d = {
            "endpoints": new_endpoints,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "mask": new_masks,
            "width": new_widths,
        }
        optimizable_tensors = self._cat_tensors_in_optimizer(d)
        self._endpoints = optimizable_tensors["endpoints"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._mask = optimizable_tensors["mask"]
        self._width = optimizable_tensors["width"]
        self.xyz_gradient_accum = torch.zeros(
            (self.endpoint_pairs.shape[0], 1), device=self.device
        )
        self.denom = torch.zeros((self.endpoint_pairs.shape[0], 1), device=self.device)
        self.max_radii2D = torch.zeros(
            (self.endpoint_pairs.shape[0]), device=self.device
        )

    def prune_segments(self, segments_prune_mask: torch.tensor):
        """
        Given a mask of segments to prune, remove them from the scene.
        Indices in endpoint_pairs are updated accordingly to keep consistency.
        Endpoints are removed if it is not referenced by any segment.

        segments_prune_mask: mask of segments to prune
        """
        segments_keep_mask = ~segments_prune_mask
        self.endpoint_pairs = self.endpoint_pairs[segments_keep_mask]
        endpoints_keep_mask = torch.zeros(
            self._endpoints.shape[0], dtype=torch.bool, device=self.device
        )
        endpoints_keep_mask[self.endpoint_pairs.flatten()] = True
        # map old indices to new indices after pruning for consistency
        old_indices = torch.unique(self.endpoint_pairs, sorted=True)
        new_indices = torch.arange(old_indices.shape[0], device=self.device)
        mapping = torch.zeros(
            old_indices.max() + 1, dtype=new_indices.dtype, device=self.device
        )
        mapping[old_indices] = new_indices
        self.endpoint_pairs = mapping[self.endpoint_pairs]
        self.strand_root_endpoint_idx = mapping[self.strand_root_endpoint_idx]
        # prune endpoint and segment attributes
        optimizable_tensors = self._prune_tensor_in_optimizer(
            endpoints_keep_mask, segments_keep_mask
        )
        self._endpoints = optimizable_tensors["endpoints"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._mask = optimizable_tensors["mask"]
        self._width = optimizable_tensors["width"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[segments_keep_mask]
        self.denom = self.denom[segments_keep_mask]
        self.max_radii2D = self.max_radii2D[segments_keep_mask]

    def merge_endpoint_pairs(self, endpoint_pair_index: torch.Tensor):
        """
        For each pair of segment endpoint id in the batch, merge those points into a new endpoint
        with the location being the average of the two endpoints. Previous points that connect to
        the merged endpoints are updated to connect to the new endpoint.
        """
        if endpoint_pair_index.shape[0] == 0:
            return
        endpoint_pair_pos = self._endpoints[endpoint_pair_index]  # (N, 2, 3)
        segment_1_complementary, segment_1_row_indices = (
            self.get_complementary_endpoint_idx(endpoint_pair_index[:, 0])
        )
        segment_2_complementary, segment_2_row_indices = (
            self.get_complementary_endpoint_idx(endpoint_pair_index[:, 1])
        )
        # segment_1_length = torch.norm(endpoint_pair_pos[:, 0] - endpoint_pair_pos[:, 1], p=2, dim=1, keepdim=True)
        # segment_2_length = torch.norm(endpoint_pair_pos[:, 0] - endpoint_pair_pos[:, 1], p=2, dim=1, keepdim=True)
        # lambda_ = segment_1_length / (segment_1_length + segment_2_length)
        lambda_ = 0.5
        # Create new endpoints
        new_endpoints = (
            lambda_ * endpoint_pair_pos[:, 1] + (1 - lambda_) * endpoint_pair_pos[:, 0]
        )
        new_endpoints_indices = (
            torch.arange(new_endpoints.shape[0], device=self.device)
            + self.endpoint_pairs.max()
            + 1
        )
        new_endpoints_map = torch.arange(
            self._endpoints.shape[0], device=self.device
        )  # map indices that are being merged to newly created endpoint indices
        new_endpoints_map[endpoint_pair_index[:, 0]] = new_endpoints_indices
        new_endpoints_map[endpoint_pair_index[:, 1]] = new_endpoints_indices
        # Create new segments
        segment_1_indices = torch.cat(
            (
                new_endpoints_map[segment_1_complementary].unsqueeze(1),
                new_endpoints_indices.unsqueeze(1),
            ),
            dim=1,
        )
        segment_2_indices = torch.cat(
            (
                new_endpoints_indices.unsqueeze(1),
                new_endpoints_map[segment_2_complementary].unsqueeze(1),
            ),
            dim=1,
        )
        new_segments_index_pair = torch.cat(
            (segment_1_indices, segment_2_indices), dim=0
        )

        # Clone other attributes from original segments
        new_features_dc_1 = self._features_dc[segment_1_row_indices]
        new_features_dc_2 = self._features_dc[segment_2_row_indices]
        new_features_rest_1 = self._features_rest[segment_1_row_indices]
        new_features_rest_2 = self._features_rest[segment_2_row_indices]
        new_opacity_1 = self._opacity[segment_1_row_indices]
        new_opacity_2 = self._opacity[segment_2_row_indices]
        new_mask_1 = self._mask[segment_1_row_indices]
        new_mask_2 = self._mask[segment_2_row_indices]
        new_width_1 = self._width[segment_1_row_indices]
        new_width_2 = self._width[segment_2_row_indices]
        new_features_dc = torch.cat((new_features_dc_1, new_features_dc_2), dim=0)
        new_features_rest = torch.cat((new_features_rest_1, new_features_rest_2), dim=0)
        new_opacity = torch.cat((new_opacity_1, new_opacity_2), dim=0)
        new_mask = torch.cat((new_mask_1, new_mask_2), dim=0)
        new_width = torch.cat((new_width_1, new_width_2), dim=0)
        self.cat_segments(
            new_segments_index_pair,
            new_endpoints,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_mask,
            new_width,
        )
        # # Calculate strand root map for consistency after pruning original endpoints used in merging
        # mapping = torch.ones(self._endpoints.shape[0], dtype=torch.long, device=self.device)
        # mapping[endpoint_pair_index.flatten()] = 0
        # mapping = torch.cumsum(mapping, dim=0) - 1
        # Prune original segment
        segments_prune_mask = torch.zeros(
            self.endpoint_pairs.shape[0], device=self.device, dtype=torch.bool
        )
        segments_prune_mask[segment_1_row_indices] = True
        segments_prune_mask[segment_2_row_indices] = True
        self.prune_segments(segments_prune_mask)

    ########################################## UTILS #########################################

    # Given distance sorted pairs, remove duplicate rows. Leaving rows where both first and second element are first occurrence
    def remove_duplicate_endpoint_rows(
        self, index_pairs_to_merge: torch.Tensor, return_mask: bool = False
    ):
        index_pairs_flatten = index_pairs_to_merge.flatten()
        mask = torch.zeros(
            index_pairs_flatten.shape[0], dtype=torch.bool, device=self.device
        )
        first_occurrence_index = self.get_first_occurence_index(index_pairs_flatten)
        mask[first_occurrence_index] = True
        # both cols should be first occurrence
        mask = mask.reshape((-1, 2))
        mask = torch.logical_and(mask[:, 0], mask[:, 1])
        index_pairs_to_merge = index_pairs_to_merge[mask]
        if return_mask:
            return index_pairs_to_merge, mask
        return index_pairs_to_merge

    def get_endpoint_pairs_row_indices(
        self, endpoint_id: torch.Tensor, exclude_segments: torch.Tensor = None
    ):
        """
        Get row index where the given endpoint_id is located in the endpoint_pairs attribute.
        If exclude_segments is given, it will be excluded from the search. And if it's the only
        segment of endpoint_id then it will return -1.
        endpoint_id: tensor of shape N with idx of endpoints to search
        exclude_segments: bool tensor of shape M where the first dim matchesn self.endpoint_pairs
        """
        mapping = -torch.ones(
            self.endpoint_pairs.max() + 1, dtype=torch.long, device=self.device
        )
        row_ids = torch.arange(self.endpoint_pairs.shape[0], device=self.device)
        endpoint_pairs = self.endpoint_pairs
        if exclude_segments is not None:
            endpoint_pairs = endpoint_pairs[~exclude_segments]
        first_col_values = endpoint_pairs[:, 0]
        second_col_values = endpoint_pairs[:, 1]
        mapping[first_col_values] = row_ids
        mapping[second_col_values] = row_ids
        row_indices = mapping[endpoint_id]
        return row_indices

    """
        Get complementary endpoint of the given segment endpoint.
        Note that these endpoints should be unique in the endpoint_pairs.
    """

    def get_complementary_endpoint_idx(
        self, endpoint_id: torch.Tensor, exclude_segments: torch.Tensor = None
    ):
        row_indices = self.get_endpoint_pairs_row_indices(endpoint_id, exclude_segments)
        selected_endpoint_pairs = self.endpoint_pairs[row_indices]
        complementary_endpoint_idx = torch.where(
            selected_endpoint_pairs[:, 1] == endpoint_id,
            selected_endpoint_pairs[:, 0],
            selected_endpoint_pairs[:, 1],
        )
        return complementary_endpoint_idx, row_indices

    """
        Get the first occurrence row index of each unique value in the tensor.
    """

    def get_first_occurence_index(self, tensor: torch.Tensor):
        unique_values, inverse_indices = torch.unique(
            tensor, return_inverse=True, sorted=False, dim=0
        )
        perm = torch.arange(
            inverse_indices.shape[0], dtype=inverse_indices.dtype, device=tensor.device
        )
        inverse_indices, perm = inverse_indices.flip([0]), perm.flip([0])
        first_occurrence_index = inverse_indices.new_empty(
            unique_values.shape[0]
        ).scatter_(0, inverse_indices, perm)
        return first_occurrence_index

    ########################################## DENSIFICATION #########################################

    def densification(
        self, extent, max_screen_size, training_info: TrainingInfo = None
    ):
        """
        Perform one step of the densification strategy.
        If a segment has a low opacity or a large covariance, then it is pruned.
        Otherwise if the gradient is large enough it will be either split into 2 segments or cloned to a new line depending on the
        norm of the covariance.

        max_grad: gradient threshold for splitting/cloning
        min_opacity: opacity threshold for pruning
        extent: scene extent (radius of sphere formed by the cameras)
        max_screen_size: maximum screen size of a point
        """
        # Densification
        grads = self.xyz_gradient_accum / self.denom  # segment average gradient
        grads[grads.isnan()] = 0.0  # avoid nan gradients
        self.clone_strategy(grads, extent, training_info)
        self.split_strategy(grads, extent, training_info)
        self.merge_collapsed_segments(training_info)
        self.prune_strategy(
            extent, max_screen_size, training_info=training_info, avoid_connected=True
        )

        # Clear memory
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Update strands info
        self.compute_strands_info()

    """
        Split segments with large view-space gradient and large covariance.
        A segment with that condition will be split into 2 new segments
        
        grads: view-space gradient of the segments
        grad_threshold: gradient threshold for splitting
        scene_extent: scene extent (radius of sphere formed by the cameras)
    """

    def split_strategy(self, grads, scene_extent, training_info: TrainingInfo = None):
        # Get segments that satisfy the gradient and covariance condition
        split_threshold = self.training_args.percent_dense * scene_extent
        num_original_segments = self.endpoint_pairs.shape[0]

        # Padding is needed to keep consistency, given that split or clone might change the number of segments
        padded_grad = torch.zeros((num_original_segments), device=self.device)
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(
            padded_grad >= self.training_args.densify_grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > split_threshold,
        )

        # Get segments that are long enough to split
        segments = self._endpoints[self.endpoint_pairs]
        segment_lengths = torch.norm(segments[:, 1] - segments[:, 0], p=2, dim=1)
        long_segments_mask = segment_lengths >= self.max_segment_length
        selected_pts_mask = torch.logical_or(selected_pts_mask, long_segments_mask)

        # Filter with foreground mask
        mask = (self.get_mask > self.foreground_binarization_th).squeeze()
        selected_pts_mask = torch.logical_and(selected_pts_mask, mask)

        # Sample points from gaussian PDF (currently fixed to the middle of segment)
        # stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        # stds = torch.zeros_like(self.get_scaling[selected_pts_mask], device=self.device)
        # means = torch.zeros((stds.size(0), 3), device=self.device)
        # samples = torch.normal(mean=means, std=stds)
        # rots = build_rotation(self.get_rotation[selected_pts_mask])
        # rotated_samples = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        # new_endpoints = rotated_samples + self.get_xyz[selected_pts_mask]   # (N, 3)
        new_endpoints = self.get_xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask].repeat(2, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(2, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(2, 1)
        new_mask = self._mask[selected_pts_mask].repeat(2, 1)
        new_width = self._width[selected_pts_mask].repeat(2, 1)

        # Connect new points to original points
        num_new_endpoints = new_endpoints.shape[0]
        max_index = torch.max(self.endpoint_pairs)
        new_endpoints_indices = (
            torch.arange(num_new_endpoints, device=self.device) + 1 + max_index
        )
        original_segment_indices = self.endpoint_pairs[selected_pts_mask]
        new_segments_1 = torch.cat(
            [
                original_segment_indices[:, 0].unsqueeze(1),
                new_endpoints_indices.unsqueeze(1),
            ],
            dim=1,
        )
        new_segments_2 = torch.cat(
            [
                new_endpoints_indices.unsqueeze(1),
                original_segment_indices[:, 1].unsqueeze(1),
            ],
            dim=1,
        )
        new_segments_index_pair = torch.cat([new_segments_1, new_segments_2], dim=0)

        # Concatenate new segments and new endpoints
        self.cat_segments(
            new_segments_index_pair,
            new_endpoints,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_mask,
            new_width,
        )

        # Prune original segments
        new_segment_mask = torch.zeros(
            2 * selected_pts_mask.sum(), device=self.device, dtype=bool
        )
        segments_prune_mask = torch.cat((selected_pts_mask, new_segment_mask))

        if training_info is not None:
            training_info.densification_info["split"] = int(selected_pts_mask.sum())

        self.prune_segments(segments_prune_mask)

    def clone_strategy(self, grads, scene_extent, training_info: TrainingInfo = None):
        """
        Clone the segments with large view-space gradient and small covariance to a new line.
        Each segment will create 2 new endpoints and create a new segment that connects them.

        grads: view-space gradient of the segments
        grad_threshold: gradient threshold for splitting
        scene_extent: scene extent (radius of sphere formed by the cameras)
        """
        split_threshold = self.training_args.percent_dense * scene_extent

        # Extract segments that satisfy the gradient and cov condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= self.training_args.densify_grad_threshold,
            True,
            False,
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= split_threshold,
        )

        # Create new endpoints
        selected_endpoints_pair_index = self.endpoint_pairs[selected_pts_mask]
        new_edpoints = self._endpoints[selected_endpoints_pair_index]  # (N, 2, 3)
        new_edpoints = new_edpoints.flatten(start_dim=0, end_dim=1)  # (2N, 3)

        # Create new segments
        num_new_endpoints = new_edpoints.shape[0]
        new_endpoints_indices = (
            torch.arange(num_new_endpoints, device=self.device)
            + self.endpoint_pairs.max()
            + 1
        )
        new_segments_index_pair = new_endpoints_indices.reshape(-1, 2)
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_masks = self._mask[selected_pts_mask]
        new_widths = self._width[selected_pts_mask]

        if training_info is not None:
            training_info.densification_info["clone"] = int(selected_pts_mask.sum())

        self.cat_segments(
            new_segments_index_pair,
            new_edpoints,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_masks,
            new_widths,
        )

    def merge_collapsed_segments(self, training_info: TrainingInfo = None):
        """
        Merge segments that are collapsed into a single point or that is not foreground.
        Only points that appear in two segments are considered for merging.
        Otherwise, it will be pruned by the prune_strategy.
        """
        if training_info is not None:
            training_info.densification_info["merge_collapsed"] = 0
        while True:
            collapsed_mask = (
                torch.norm(
                    self._endpoints[self.endpoint_pairs[:, 1]]
                    - self._endpoints[self.endpoint_pairs[:, 0]],
                    dim=1,
                )
                < self.min_val
            )
            bg_mask = ~self.compute_foreground_mask()
            mask = torch.logical_or(collapsed_mask, bg_mask)
            collapsed_segments_id = self.endpoint_pairs[mask]
            # Identify connected segments
            u, c = torch.unique(self.endpoint_pairs, return_counts=True)
            non_unique = u[c != 1]
            collapse_segments_merge_mask = torch.all(
                torch.isin(collapsed_segments_id, non_unique), dim=1
            )
            # Prune: disconnect collapsed segments
            mask[mask == True] = collapse_segments_merge_mask
            segments_to_merge = collapsed_segments_id[collapse_segments_merge_mask]
            segments_to_merge, non_dup_mask = self.remove_duplicate_endpoint_rows(
                segments_to_merge, return_mask=True
            )
            mask[mask == True] = non_dup_mask
            self.prune_segments(mask)
            # Merge: all segments referenced by second column of segments_to_merge should be updated to be first column index
            mapping = torch.arange(self.endpoint_pairs.max() + 1, device=self.device)
            mapping[segments_to_merge[:, 1]] = segments_to_merge[:, 0]
            self.endpoint_pairs = mapping[self.endpoint_pairs]
            self.prune_segments(
                torch.zeros(
                    self.endpoint_pairs.shape[0], dtype=torch.bool, device=self.device
                )
            )  # let prune_segments function update the indexing
            num_collapsed_merge = segments_to_merge.shape[0]
            if training_info is not None:
                training_info.densification_info[
                    "merge_collapsed"
                ] += num_collapsed_merge
            if num_collapsed_merge == 0:
                break

    def prune_strategy(
        self,
        extent,
        max_screen_size,
        training_info: TrainingInfo = None,
        avoid_connected: bool = False,
    ):
        """
        Prune collapsed segments, ones with low opacity or large covariance.
        min_opacity: opacity threshold
        extent: scene extent (radius of sphere formed by the cameras), world space size threshold
        max_screen_size: screen space size threshold
        """
        # Collapsed segments
        prune_mask = (
            torch.norm(
                self._endpoints[self.endpoint_pairs[:, 1]]
                - self._endpoints[self.endpoint_pairs[:, 0]],
                dim=1,
            )
            < self.min_val
        )
        if training_info is not None:
            training_info.densification_info["prune_collapsed"] = int(prune_mask.sum())
        # Low opacity segments
        low_opacity_mask = (self.get_opacity < self.opacity_th).squeeze()
        if training_info is not None:
            training_info.densification_info["prune_low_opacity"] = int(
                low_opacity_mask.sum()
            )
        prune_mask = torch.logical_or(prune_mask, low_opacity_mask)
        # Big vs/ws points
        if max_screen_size and extent != 0.0:  # extent == 0.0 is the case of 1 cam
            # big_points_vs_mask = self.max_radii2D > max_screen_size
            # prune_mask = torch.logical_or(prune_mask, big_points_vs_mask) # [Not used, densification_postfix will set radii2D to 0]
            big_points_ws_mask = self.get_scaling.max(dim=1).values > 0.1 * extent
            if training_info is not None:
                training_info.densification_info["prune_big_ws"] = int(
                    big_points_ws_mask.sum()
                )
            prune_mask = torch.logical_or(prune_mask, big_points_ws_mask)
        # Skip connected segments: prune happens on segments that are either disconnected or not fg
        if avoid_connected and prune_mask.sum() != 0:
            u, c = torch.unique(self.endpoint_pairs, return_counts=True)
            unique = u[c == 1]
            is_end_segment = torch.any(torch.isin(self.endpoint_pairs, unique), dim=1)
            is_not_fg = (self.get_mask < self.foreground_binarization_th).squeeze()
            mask = torch.logical_or(is_end_segment, is_not_fg)
            if training_info is not None:
                training_info.densification_info["prune_avoided"] = int(
                    (prune_mask).sum() - (prune_mask[mask]).sum()
                )
            prune_mask = torch.logical_and(prune_mask, mask)
        # prune
        total_prune_points = int(prune_mask.sum())
        if training_info is not None:
            training_info.densification_info["prune_total"] = total_prune_points
        if total_prune_points > 0 and total_prune_points < self._opacity.shape[0]:
            self.prune_segments(prune_mask)

    def merging(self, training_info: TrainingInfo = None):
        """
        Function that handles the merging operation. Candidates joints will be computed by compute_endpoint_pair_to_merge and then merged by merge_endpoint_pairs. After merging, strands_info will be updated.

        training_info: TrainingInfo object to store the merging information
        """
        self.compute_strands_info()
        endpoint_pair_to_merge = self.compute_endpoint_pair_to_merge()
        if training_info is not None:
            training_info.densification_info["merge"] = int(
                endpoint_pair_to_merge.shape[0]
            )
        self.merge_endpoint_pairs(endpoint_pair_to_merge)
        # Clear memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
        # Update strands info
        self.compute_strands_info()

    def growing(
        self,
        training_info: TrainingInfo = None,
        growth_length: float = 0.002,
    ):
        """
        Grow the strands by adding new segments to the tip of the strands.
        New point location is determined by the tip position and the direction of the last few segments.

        Args:
            training_info: TrainingInfo object to store the growing information
            growth_length: length of the new segment
        """
        max_strand_length = self.training_args.num_points_strand
        new_segments_index_pair = []
        new_endpoints = []
        new_features_dc = []
        new_features_rest = []
        new_opacities = []
        new_masks = []
        new_widths = []
        points_position = self._endpoints.cpu().numpy()
        features_dc = self._features_dc.cpu().numpy()
        features_rest = self._features_rest.cpu().numpy()
        opacity = self._opacity.cpu().numpy()
        mask = self._mask.cpu().numpy()
        width = self._width.cpu().numpy()
        total_endpoints = points_position.shape[0]
        counter = 0
        # iterate over strands
        for segment_id_pairs, segment_id in zip(
            self.strands_info.list_strands, self.strands_info.list_strands_segments_id
        ):
            if segment_id_pairs.shape[0] >= max_strand_length:
                continue
            tip_postion = points_position[segment_id_pairs[-1, 1]]
            # get last few segments
            num_averaging_points = min(
                segment_id_pairs.shape[0], self.training_args.growth_averaging_points
            )
            segments = segment_id_pairs[-num_averaging_points:]
            segment_id = segment_id[-num_averaging_points:]
            directions = (
                points_position[segments[:, 1]] - points_position[segments[:, 0]]
            )
            # skip collapsed segments
            directions_norm = np.linalg.norm(directions, axis=1)
            collapsed_mask = directions_norm < self.min_val
            segments = segments[~collapsed_mask]
            directions = directions[~collapsed_mask]
            directions_norm = directions_norm[~collapsed_mask]
            segment_id_ = segment_id[~collapsed_mask]
            if segments.shape[0] == 0:
                continue
            # get new point position as the average of the last few segments
            directions = directions / directions_norm[:, np.newaxis]
            avg_direction = np.mean(directions, axis=0)
            if growth_length is None:
                growth_length = np.mean(directions_norm)
            new_point_position = tip_postion + avg_direction * growth_length
            new_segment_index_pair = [
                segment_id_pairs[-1, 1],
                total_endpoints + counter,
            ]
            new_segments_index_pair.append(new_segment_index_pair)
            new_endpoints.append(new_point_position)
            # get other attributes also as average of the last few segments
            new_features_dc.append(np.mean(features_dc[segment_id_], axis=0))
            new_features_rest.append(np.mean(features_rest[segment_id_], axis=0))
            new_opacities.append(np.mean(opacity[segment_id_], axis=0))
            new_masks.append(np.mean(mask[segment_id_], axis=0))
            new_widths.append(np.mean(width[segment_id_], axis=0))
            counter += 1
        new_segments_index_pair = np.array(new_segments_index_pair)
        new_endpoints = np.array(new_endpoints)
        new_features_dc = np.array(new_features_dc)
        new_features_rest = np.array(new_features_rest)
        new_opacities = np.array(new_opacities)
        new_masks = np.array(new_masks)
        new_widths = np.array(new_widths)
        new_segments_index_pair = torch.tensor(
            new_segments_index_pair, device=self.device
        )
        new_endpoints = torch.tensor(new_endpoints, device=self.device)
        new_features_dc = torch.tensor(new_features_dc, device=self.device)
        new_features_rest = torch.tensor(new_features_rest, device=self.device)
        new_opacities = torch.tensor(new_opacities, device=self.device)
        new_masks = torch.tensor(new_masks, device=self.device)
        new_widths = torch.tensor(new_widths, device=self.device)
        self.cat_segments(
            new_segments_index_pair,
            new_endpoints,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_widths,
        )
        if training_info is not None:
            training_info.densification_info["grow"] = counter

        # Clear memory
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Update strands info
        self.compute_strands_info()

    def compute_endpoint_pair_to_merge(
        self,
        chunk_size: int = -1,
        max_num_nn: int = -1,
    ):
        """
        Compute endpoint (strand root or tip) pairs to merge. Two nearby strand endpoints are matched
        if they are within certain distance and angle threshold.
        The problem itself is a one-to-one matching problem, i.e. Hungarian algorithm. But here we dont compute
        the optimal solution, instead we use a greedy approach based on sorting by distance and angle difference.

        Args:
            chunk_size: used to processes strand endpoints in chunks (to avoid memory limitations by RAM)
            max_num_nn: maximum number of nearest neighbors to consider (to limit memory and complexity)

        Returns:
            endpoint_pairs_to_merge: tensor of shape (N, 2) containing the endpoints to merge
        """

        # Given endpoint indices compute the direction of the segment containing them
        def compute_endpoint_segment_direction(endpoint_index: torch.Tensor):
            endpoint = self._endpoints[endpoint_index]
            complementary_idx, _ = self.get_complementary_endpoint_idx(endpoint_index)
            complementary_endpoint = self._endpoints[complementary_idx]
            endpoint_to_complementary_dir = complementary_endpoint - endpoint
            endpoint_to_complementary_dir = endpoint_to_complementary_dir / torch.norm(
                endpoint_to_complementary_dir, dim=1, keepdim=True
            )
            return endpoint_to_complementary_dir

        # Given distance sorted pairs, remove rows where either of their complementary appeared before
        def remove_complementary_rows(
            index_pairs_to_merge: torch.Tensor, complementary_map: torch.Tensor
        ):
            disabled = torch.zeros(
                complementary_map.max() + 1, dtype=torch.bool, device=self.device
            )
            mask = torch.ones(
                index_pairs_to_merge.shape[0], dtype=torch.bool, device=self.device
            )
            n = index_pairs_to_merge.shape[0]
            for i in range(n):
                el1 = index_pairs_to_merge[i, 0]
                el2 = index_pairs_to_merge[i, 1]
                if disabled[el1] or disabled[el2]:
                    mask[i] = False
                else:
                    disabled[complementary_map[el1]] = True
                    disabled[complementary_map[el2]] = True
            index_pairs_to_merge = index_pairs_to_merge[mask]
            return index_pairs_to_merge

        dist_th = self.merge_dist_th
        angle_th = self.merge_angle_th
        dir_th = np.cos(np.deg2rad(angle_th))

        # Find strand endpoints (root/tip) skipping gaussians that are not foregournd
        endpoints_ids, counts = torch.unique(self.endpoint_pairs, return_counts=True)
        strand_endpoint_id = endpoints_ids[counts == 1]
        fg_mask = self.compute_foreground_mask()
        fg_segments_id = self.endpoint_pairs[fg_mask]
        strand_endpoint_mask = torch.isin(strand_endpoint_id, fg_segments_id.flatten())
        strand_endpoint_id = strand_endpoint_id[strand_endpoint_mask]

        # Filter out strand root
        # strand_root_mask = torch.isin(strand_endpoint_id, self.strand_root_endpoint_idx)
        # strand_endpoint_id = strand_endpoint_id[~strand_root_mask]

        filtered_endpoints = self._endpoints[strand_endpoint_id].detach().cpu().numpy()
        filtered_endpoints_direction = (
            compute_endpoint_segment_direction(strand_endpoint_id)
            .detach()
            .cpu()
            .numpy()
        )
        strand_endpoint_id = strand_endpoint_id.detach().cpu().numpy()

        selected_p1 = []
        selected_p2 = []
        dist = []
        num_points = strand_endpoint_id.shape[0]
        position_tree = cKDTree(filtered_endpoints)
        chunk_size = num_points if chunk_size <= 0 else chunk_size
        # iterate over all strand endpoints by chunks
        loop_range = (
            range(0, num_points, chunk_size)
            if chunk_size == num_points
            else tqdm(range(0, num_points, chunk_size))
        )
        for chunk_start in loop_range:
            remaining_points = num_points - chunk_start
            chunk_size_ = min(chunk_size, remaining_points)
            p1_nns = position_tree.query_ball_point(
                filtered_endpoints[chunk_start : chunk_start + chunk_size_],
                workers=-1,
                r=dist_th,
                return_sorted=True,
            )
            # for each point in the chunk
            for i in range(chunk_size_):
                p1_nn = np.array(p1_nns[i])
                curr_point_idx = chunk_start + i
                curr_point_idx_global = strand_endpoint_id[curr_point_idx]
                strand_complementary_endpoint = (
                    self.strands_info.strand_endpoint_id_to_complementary[
                        curr_point_idx_global
                    ]
                )
                p1_nn_global = strand_endpoint_id[p1_nn]
                # complementary (root <-> tip) and self filter
                complementary_filter = p1_nn_global != strand_complementary_endpoint
                self_filter = p1_nn_global != curr_point_idx_global
                # apply filter
                filter = np.logical_and(complementary_filter, self_filter)
                p1_nn = p1_nn[filter]
                # if any matched
                if len(p1_nn) > 0:
                    p1_dir = -filtered_endpoints_direction[
                        curr_point_idx
                    ]  # they should point opposite directions
                    p1_nn_dirs = filtered_endpoints_direction[p1_nn]
                    dot_product = p1_nn_dirs @ p1_dir.T
                    if self.training_args.bidirectional_merge:
                        dot_product = np.abs(dot_product)
                    dir_filter = dot_product >= dir_th
                    p1_nn = p1_nn[dir_filter]
                    p1_nn_dists = np.linalg.norm(
                        filtered_endpoints[curr_point_idx] - filtered_endpoints[p1_nn],
                        axis=1,
                    )
                    num_nn = (
                        len(p1_nn) if max_num_nn <= 0 else min(max_num_nn, len(p1_nn))
                    )
                    # add all candidates
                    for j in range(num_nn):
                        selected_p1.append(strand_endpoint_id[curr_point_idx])
                        selected_p2.append(strand_endpoint_id[p1_nn[j]])
                        dist.append(p1_nn_dists[j])

        selected_p1 = torch.tensor(selected_p1, device=self.device)
        selected_p2 = torch.tensor(selected_p2, device=self.device)
        dist = torch.tensor(dist, device=self.device)

        # Sort asc by distance and remove duplicates
        _, indices = torch.sort(dist, descending=False)
        selected_p1 = selected_p1[indices]
        selected_p2 = selected_p2[indices]
        index_pairs_to_merge = torch.cat(
            (selected_p1.unsqueeze(1), selected_p2.unsqueeze(1)), dim=1
        )  # (N, 2)
        index_pairs_to_merge = self.remove_duplicate_endpoint_rows(index_pairs_to_merge)
        complementary_map = torch.from_numpy(
            self.strands_info.strand_endpoint_id_to_complementary
        ).to(self.device)
        index_pairs_to_merge = remove_complementary_rows(
            index_pairs_to_merge, complementary_map
        )
        return index_pairs_to_merge

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self._replace_tensor_in_optimizer(
            opacities_new, "opacity"
        )
        self._opacity = optimizable_tensors["opacity"]

    def update_strand_root(self, dist_th: float = 1e-2):
        if self.ref_strand_root is None or self.ref_strand_root.shape[0] == 0:
            return
        ref_strand_root = (
            torch.from_numpy(self.ref_strand_root)
            .to(self.device)
            .to(self._endpoints.dtype)
        )
        endpoints = self._endpoints
        # Strategy 1: set as strand root all endpoints that is close to the gt strand roots
        # find endpoints that are this dist to gt strand roots
        selected_endpoint_mask = torch.zeros(
            endpoints.shape[0], device=self.device, dtype=torch.bool
        )
        strand_root_ = ref_strand_root.unsqueeze(0)
        dist, p1_nn, _ = knn_points(strand_root_, endpoints.unsqueeze(0), K=1)
        selected_nn = dist[0] <= dist_th
        p1_nn = p1_nn[0]
        p1_nn = p1_nn[selected_nn]
        selected_endpoint_mask[p1_nn] = True
        # convert to mask and then back to remove duplicate
        endpoint_idx = torch.arange(
            endpoints.shape[0], device=self.device, dtype=torch.long
        )
        endpoint_idx = endpoint_idx[selected_endpoint_mask]
        self.strand_root_endpoint_idx = endpoint_idx.long().to(self.device)
        print(f"Identified {selected_endpoint_mask.sum()} endpoints as strand roots")

    def update_densification_stats(self, viewspace_point_tensor, radii, update_filter):
        self.max_radii2D[update_filter] = torch.max(
            self.max_radii2D[update_filter], radii[update_filter]
        )
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def compute_strands_info(self, only_foreground: bool = True):
        """
        Updates the `strands_info` attribute of the class. This information is used for the merging and growing strategies.
        Assumes that `endpoint_pairs` is well-formed: each point appears either once (strand endpoint) or twice (internal point), and no cycles are present.
        """
        if self.ref_strand_root is None or self.ref_strand_root.shape[0] == 0:
            raise ValueError("ref_strand_root is not set")
        strand_root_kdTree = cKDTree(self.ref_strand_root)
        endpoints = self._endpoints.detach().cpu().numpy()
        endpoint_pairs = self.endpoint_pairs.cpu().numpy()
        if only_foreground:
            mask = self.compute_foreground_mask().cpu().numpy()
            endpoint_pairs = endpoint_pairs[mask]
        # Create a map from endpoint id to endpoint_pairs row id
        id_to_row_id = -np.ones(
            (endpoint_pairs.max() + 1, 2), dtype=np.int32
        )  # each can map to 2 rows at max
        for row_id, pair in enumerate(endpoint_pairs):
            for id in pair:
                col_ = 0 if id_to_row_id[id][0] == -1 else 1
                id_to_row_id[id][col_] = row_id
        # Compute strand endpoints
        endpoints_ids, counts = np.unique(endpoint_pairs, return_counts=True)
        strand_endpoint_id = endpoints_ids[counts == 1]
        # Aggregate strand points and disambiguate direction by comparing the distance to the ref
        num_strands = strand_endpoint_id.shape[0] // 2
        list_strands = np.empty(
            num_strands, dtype=object
        )  # [np.array([[x1, x2], [x2, x3], ...]), ...]
        list_strands_segments_id = np.empty(
            num_strands, dtype=object
        )  # [np.array([row_id1, row_id2, ...]), ...]
        id_to_strand_id = -np.ones(endpoints.shape[0], dtype=np.int32)
        strand_endpoint_id_to_complementary = -np.ones(
            endpoints.shape[0], dtype=np.int32
        )
        visited = np.zeros(
            strand_endpoint_id.max() + 1, dtype=bool
        )  # visited strands endpoint
        strand_counter = 0
        for strand_start_id in strand_endpoint_id:
            if not visited[strand_start_id]:
                curr_strand = []
                curr_strand_segments_id = []
                curr_point_id = strand_start_id
                row_id = id_to_row_id[curr_point_id][
                    0
                ]  # strand endpoint appears only in one row
                while row_id != -1:
                    id_to_strand_id[curr_point_id] = strand_counter
                    endpoint_pair_row = endpoint_pairs[row_id]  # (2,)
                    next_point_id = (
                        endpoint_pair_row[0]
                        if endpoint_pair_row[0] != curr_point_id
                        else endpoint_pair_row[1]
                    )
                    curr_strand.append([curr_point_id, next_point_id])
                    curr_strand_segments_id.append(row_id)
                    curr_point_id = next_point_id
                    row_id = (
                        id_to_row_id[curr_point_id][0]
                        if id_to_row_id[curr_point_id][0] != row_id
                        else id_to_row_id[curr_point_id][1]
                    )
                strand_endpoint_id_to_complementary[strand_start_id] = curr_point_id
                strand_endpoint_id_to_complementary[curr_point_id] = strand_start_id
                visited[strand_start_id] = True
                visited[curr_point_id] = True
                id_to_strand_id[curr_point_id] = strand_counter
                curr_strand = np.array(curr_strand)
                curr_strand_segments_id = np.array(curr_strand_segments_id)
                # Determine if strand should be flipped by distance of start point and end point to ref_strand_root
                strand_endpoint_pos = np.stack(
                    [endpoints[strand_start_id], endpoints[curr_point_id]]
                )
                distances, _ = strand_root_kdTree.query(strand_endpoint_pos, k=1)
                flip = distances[0] > distances[1]
                if flip:
                    curr_strand = np.flip(np.flip(curr_strand, axis=1), axis=0)
                    curr_strand_segments_id = np.flip(curr_strand_segments_id)
                list_strands[strand_counter] = curr_strand
                list_strands_segments_id[strand_counter] = curr_strand_segments_id
                strand_counter += 1
        self.strands_info = StrandsInfo(
            list_strands=list_strands,
            list_strands_segments_id=list_strands_segments_id,
            id_to_strand_id=id_to_strand_id,
            strand_endpoint_id_to_complementary=strand_endpoint_id_to_complementary,
        )

    ########################################## CONVERSION & CLEANING #########################################

    def clean_gaussians(self, avoid_connected: bool = True):
        """
        Remove gaussians that background, transparent or not part line-like structures.
        """
        prune_mask = ~self.compute_foreground_mask()
        # avoid connected
        if avoid_connected:
            u, c = torch.unique(self.endpoint_pairs, return_counts=True)
            unique = u[c == 1]
            segment_to_prune = self.endpoint_pairs[prune_mask]
            is_unique = torch.isin(segment_to_prune, unique)
            is_end_segment = torch.logical_or(is_unique[:, 0], is_unique[:, 1])
            prune_mask[prune_mask == True] = is_end_segment
        self.prune_segments(prune_mask)
