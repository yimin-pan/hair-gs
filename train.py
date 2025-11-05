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

import sys
import torch
import numpy as np
import cv2
from argparse import ArgumentParser
from random import randint
from tqdm import tqdm

from utils import (
    create_subplots_from_dict,
    create_pv_background_plotter,
    get_joints_and_segments_from_hair_gs,
    render_image_dict_from_cameras,
    prepare_output_path,
    enable_accelerated_rasterization,
    safe_state,
    get_logger,
    TrainingInfo,
)
from loss import loss_function, compute_metrics
from data import compute_eval_data_from_gs, compute_eval_data_from_hair_gs
from arguments import ModelParams, OptimizationParams, GeneralParams
from gaussian_renderer import render, network_gui
from scene import Scene, HairGaussianModel


def training(
    mp: ModelParams, opt: OptimizationParams, gp: GeneralParams, args: ArgumentParser
):
    scene = Scene(args, shuffle=True)
    gaussians = scene.gaussians
    gaussians.training_setup(opt)
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    training_info = TrainingInfo()
    training_info.iter = scene.loaded_iter
    logger = get_logger(args)

    # Visualization setup
    cameras = scene.getCameras()
    images_dict = render_image_dict_from_cameras(gaussians, cameras, bg)
    training_info.composed_image = create_subplots_from_dict(
        images_dict, image_w=1920, image_h=1080
    )
    if gp.vis2d:
        cv2.imshow(
            f"Image Grid",
            cv2.cvtColor(training_info.composed_image, cv2.COLOR_RGB2BGR),
        )
        cv2.waitKey(1)
    if gp.vis3d and isinstance(gaussians, HairGaussianModel):
        plotter, polydata = create_pv_background_plotter(gaussians, cameras)

    # Initial evaluation
    if scene.gt is not None:
        pred = (
            compute_eval_data_from_hair_gs(gaussians)
            if isinstance(gaussians, HairGaussianModel)
            else compute_eval_data_from_gs(gaussians)
        )
        training_info.eval_metrics, training_info.eval_thresholds = compute_metrics(
            pred=pred,
            gt=scene.gt,
            bidirectional=op.bidirectional_eval,
        )
    logger.log(training_info, gaussians)

    # Loop initialization
    convert_SHs_python = False
    compute_cov3D_python = False
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(
        range(scene.loaded_iter, training_info.iter + opt.iterations + 1),
        desc="Training progress",
    )

    # Training loop
    for iteration in range(1, opt.iterations + 1):
        training_info.iter = scene.loaded_iter + iteration

        # Network GUI
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    convert_SHs_python,
                    compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    net_image = render(
                        custom_cam,
                        gaussians,
                        bg,
                        scaling_modifer,
                        convert_SHs_python=convert_SHs_python,
                        compute_cov3D_python=compute_cov3D_python,
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, mp.source_path)
                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Increase SH degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Get random camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Forward and backward pass
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            bg,
            convert_SHs_python=convert_SHs_python,
            compute_cov3D_python=compute_cov3D_python,
        )
        image = render_pkg["render"]
        loss, loss_dict = loss_function(gaussians, image, viewpoint_cam, args)
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_info.elapsed_time = iter_start.elapsed_time(iter_end)
            training_info.loss = loss
            training_info.loss_dict = loss_dict

            # Adaptive densification
            if iteration < opt.densify_until_iter:
                gaussians.update_densification_stats(
                    render_pkg["viewspace_points"],
                    render_pkg["radii"],
                    render_pkg["visibility_filter"],
                )
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        opt.prune_max_radii_2d
                        if iteration > opt.opacity_reset_interval
                        else None
                    )
                    gaussians.densification(
                        scene.cameras_extent, size_threshold, training_info
                    )

                # Reset opacity
                if iteration % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()

            # Merge and growth
            if isinstance(gaussians, HairGaussianModel):
                if iteration % opt.merge_interval == 0:
                    gaussians.merging(training_info=training_info)
                if iteration % opt.growth_interval == 0:
                    gaussians.growing(training_info=training_info)

            # Optimizer step
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            # 2D visualization
            images_dict[str(viewpoint_cam.uid) + "-render"] = (
                image.detach().clone().cpu().numpy().transpose(1, 2, 0).copy() * 255
            ).astype(np.uint8)
            if iteration % gp.update_vis2d_frequency == 0:
                training_info.composed_image = create_subplots_from_dict(
                    images_dict, image_w=1920, image_h=1080
                )
                if gp.vis2d:
                    cv2.imshow(
                        f"Image Grid",
                        cv2.cvtColor(training_info.composed_image, cv2.COLOR_RGB2BGR),
                    )
                    cv2.waitKey(1)

            # 3D visualization
            if gp.vis3d and isinstance(gaussians, HairGaussianModel):
                polydata = get_joints_and_segments_from_hair_gs(
                    gaussians, polydata=polydata
                )
                plotter.render()
                plotter.app.processEvents()

            # Eval
            if (
                scene.gt is not None
                and iteration % gp.eval_frequency == 0
                or iteration == opt.iterations
            ):
                training_info.pred = (
                    compute_eval_data_from_hair_gs(gaussians)
                    if isinstance(gaussians, HairGaussianModel)
                    else compute_eval_data_from_gs(gaussians)
                )
                training_info.eval_metrics, training_info.eval_thresholds = (
                    compute_metrics(
                        pred=pred,
                        gt=scene.gt,
                        bidirectional=op.bidirectional_eval,
                    )
                )

            # Log
            logger.log(training_info, gaussians)

            # Save scene
            if iteration % args.save_frequency == 0 or iteration == opt.iterations:
                print("\n[ITER {}] Saving scene".format(iteration))
                scene.save(iteration)

    print(
        f"Training completed, gaussians saved to {scene.model_path}/point_cloud/iteration_{scene.loaded_iter + opt.iterations}/point_cloud.ply"
    )

    # Clean up visualization
    if gp.vis2d:
        cv2.destroyAllWindows()
    if gp.vis3d and isinstance(gaussians, HairGaussianModel):
        plotter.close()
        plotter.deep_clean()


if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser(description="Training script parameters")
    mp = ModelParams(parser)
    op = OptimizationParams(parser)
    gp = GeneralParams(parser)
    args = parser.parse_args(sys.argv[1:])

    prepare_output_path(args)
    safe_state(args.quiet)
    enable_accelerated_rasterization()

    # Start GUI server, configure and run training
    network_gui.init(gp.ip, gp.port)
    training(mp.extract(args), op.extract(args), gp.extract(args), args)
