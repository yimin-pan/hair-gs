#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import torchvision

from gaussian_renderer import render
from utils import orientation_map_to_vis, safe_state
from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams, GeneralParams, get_combined_args

type_map = {
    -1: "all",
    0: "rgb",
    1: "rgb_foreground",
    2: "mask_foreground",
    3: "mask_other",
    4: "orientation_map",
}


def render_set(args, name, iteration, views, gaussians, optimization, type):
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=args.data_device)
    model_path = args.model_path

    type_name = type_map[type]
    render_path = os.path.join(
        model_path,
        "render",
        name,
        "iteration_{}".format(iteration),
        "renders",
        type_name,
    )
    gts_path = os.path.join(
        model_path, "render", name, "iteration_{}".format(iteration), "gt", type_name
    )
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    if type == 1:
        gaussians.training_setup(optimization)
        gaussians.clean_gaussians()

    for idx, view in enumerate(views):
        if type == 0 or type == 1:
            rendering = render(view, gaussians, background)["render"]
            gt = view.original_image[0:3, :, :]
        elif type == 2:
            mask = (
                gaussians.get_mask.repeat(1, 3) >= gaussians.foreground_binarization_th
            ).float()
            rendering = render(view, gaussians, background, override_color=mask)[
                "render"
            ][0]
            gt = view.float_mask
        elif type == 3:
            color = (
                gaussians.get_mask.repeat(1, 3) < gaussians.foreground_binarization_th
            ).float()
            rendering = render(view, gaussians, background, override_color=color)[
                "render"
            ][0]
            gt = (~view.mask).float()
        elif type == 4:
            orientation_world = gaussians.get_orientation
            orientation_map_world = render(
                view, gaussians, background, override_color=orientation_world
            )["render"]
            orientation_map_world = orientation_map_world.permute(1, 2, 0)  # (H, W, 3)
            orientation_map_world_ = orientation_map_world.flatten(
                start_dim=0, end_dim=1
            )  # (H*W, 3)
            orientation_map_view_ = (
                orientation_map_world_ @ view.world_view_transform[:3, :3]
            )
            orientation_map_pixel_ = orientation_map_view_[
                :, :2
            ]  # (H*W, 2) ommit z not relevant for orientation
            orientation_map_pixel_ = orientation_map_pixel_ / (
                torch.norm(orientation_map_pixel_, dim=1, keepdim=True)
                + gaussians.min_val
            )
            # compute thetas in [0, pi) wrt to y-axis clockwise
            x = orientation_map_pixel_[:, 0]
            y = orientation_map_pixel_[:, 1]
            y = torch.where(y < gaussians.min_val, y + gaussians.min_val, y)
            thetas = torch.atan2(x, y)
            thetas = torch.where(thetas < 0, thetas + np.pi, thetas)
            rendering = thetas.reshape(orientation_map_world.shape[:2])
            gt = view.orientation_field
            # from theta to hsv space for visualization
            rendering = orientation_map_to_vis(rendering, view.orientation_confidence)
            rendering = torch.from_numpy(rendering / 255).permute(2, 0, 1)
            gt = orientation_map_to_vis(gt, view.orientation_confidence)
            gt = torch.from_numpy(gt / 255).permute(2, 0, 1)
        else:
            raise ValueError("Invalid rendering type")

        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
        )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    optimization = OptimizationParams(parser)
    general = GeneralParams(parser)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--type", "-t", type=int, default=-1, help="Type of rendering")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    with torch.no_grad():
        scene = Scene(args)
        gaussians = scene.gaussians
        types = (
            [args.type] if args.type != -1 else [0, 2, 3, 4, 1]
        )  # 1 is special case as it deletes gaussians so it should be last
        for type in tqdm(types):
            if not args.skip_train:
                render_set(
                    args,
                    "train",
                    scene.loaded_iter,
                    scene.getCameras(),
                    gaussians,
                    optimization,
                    type,
                )
