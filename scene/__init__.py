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
import random
import json

from plyfile import PlyData

from .hair_gaussian_model import GaussianModel, HairGaussianModel
from arguments import ModelParams
from data import (
    readColmapSceneInfo,
    eval_data_loading_callbacks,
    HairEvalData,
    load_head_reconstruction_data_npz,
    HeadReconstruction,
)
from utils import search_for_max_interation
from .cameras import cameraList_from_camInfos, camera_to_JSON


class Scene:
    gaussians: GaussianModel = None
    cameras: dict = None
    cameras_extent: float = None
    loaded_iter: int = None
    gt: HairEvalData = None
    head_reconstruction: HeadReconstruction = None

    def __init__(self, args: ModelParams, shuffle=True, resolution_scales=[1.0]):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.cameras = {}

        # Load scene info from COLMAP
        scene_info = readColmapSceneInfo(args.source_path, args.images)

        # Create input.ply and cameras.json if model_path is empty
        try:
            self.loaded_iter = search_for_max_interation(
                os.path.join(self.model_path, "point_cloud")
            )
        except FileNotFoundError:
            with open(scene_info.ply_path, "rb") as src_file, open(
                os.path.join(self.model_path, "input.ply"), "wb"
            ) as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.cameras:
                camlist.extend(scene_info.cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        # Load cameras from scene_info
        if shuffle:
            random.shuffle(scene_info.cameras)
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        for resolution_scale in resolution_scales:
            self.cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.cameras, resolution_scale, args
            )

        if self.loaded_iter is None:
            # Initialize GaussianModel from pcd
            self.gaussians = GaussianModel(
                args.sh_degree, self.cameras_extent, device=args.data_device
            )
            self.gaussians.create_from_pcd(scene_info.point_cloud)
            print(f"Created {self.gaussians.__class__.__name__} from PCD")
            self.loaded_iter = 0
        else:
            # Load GaussianModel/HairGaussianModel previously iteration ply
            path = os.path.join(
                self.model_path,
                "point_cloud",
                "iteration_" + str(self.loaded_iter),
                "point_cloud.ply",
            )
            plydata = PlyData.read(path)
            self.gaussians = (
                GaussianModel(
                    args.sh_degree, self.cameras_extent, device=args.data_device
                )
                if len(plydata.elements) == 1
                else HairGaussianModel(
                    args.sh_degree, self.cameras_extent, device=args.data_device
                )
            )
            print(
                f"Loaded {self.gaussians.__class__.__name__} from PLY at iteration {self.loaded_iter}"
            )
            self.gaussians.load_ply(path)

        # Load GT (for evaluation)
        gt_data_path = os.path.join(args.source_path, "hair_eval_data.npz")
        if os.path.exists(gt_data_path):
            self.gt = eval_data_loading_callbacks["gt"](gt_data_path)
            print(f"GT loaded from {gt_data_path}")

        # Load head reconstruction data
        head_reconstruction_data_path = os.path.join(
            args.source_path, "head_reconstruction_data.npz"
        )
        if os.path.exists(head_reconstruction_data_path):
            self.head_reconstruction = load_head_reconstruction_data_npz(
                head_reconstruction_data_path
            )
            self.gaussians.ref_strand_root = self.head_reconstruction.scalp_verts
            if isinstance(self.gaussians, HairGaussianModel):
                self.gaussians.update_strand_root()
                self.gaussians.compute_strands_info()
            print(f"Head reconstruction loaded from {head_reconstruction_data_path}")

    def save(self, iteration: int = 0):
        if self.loaded_iter:
            iteration += self.loaded_iter
        point_cloud_path = os.path.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getCameras(self, scale=1.0):
        return self.cameras[scale]
