"""
This script modifies the last iteration .ply of the optimization output folder
specified by model_path. The original .ply will be saved renamed to original.ply
"""

import time

import torch
import pyvista as pv
import cv2

from arguments import ModelParams, ArgumentParser, OptimizationParams, GeneralParams
from scene import Scene, HairGaussianModel
from data import compute_eval_data_from_hair_gs

from utils import (
    pv_visualize,
    get_joints_and_segments_from_hair_gs,
    render_image_dict_from_cameras,
    create_subplots_from_dict,
    get_logger,
    TrainingInfo,
)
from loss import compute_metrics

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    mp = ModelParams(parser)
    op = OptimizationParams(parser)
    gp = GeneralParams(parser)
    args = parser.parse_args()

    # Load the scene with the specified parameters
    scene = Scene(args)
    gaussians = scene.gaussians
    gaussians.training_setup(op)
    training_info = TrainingInfo()
    logger = get_logger(args)
    assert not isinstance(
        gaussians, HairGaussianModel
    ), "This process is meant to be called after first stage optimization to convert Gaussian Model to Hair Gaussian Model, check the last iteration .ply file"

    with torch.inference_mode():
        if gp.vis3d:
            p = pv.Plotter(shape=(1, 2), border=False)
            p.set_background("black")
            p.subplot(0, 0)
            pv_visualize(
                plotter=p,
                point_clouds={"blue": scene.gaussians.get_xyz},
                title="Initial Gaussians",
            )

        # Convert GS-Points to GS-Lines
        hair_gs = gaussians.to_hair_gaussian_model()
        scene.gaussians = hair_gs
        del gaussians

        # Visualize results before merging
        if gp.vis3d:
            p = pv.Plotter(shape=(1, 3), border=False)
            p.set_background("black")
            p.subplot(0, 0)
            pv_visualize(
                plotter=p,
                point_clouds={
                    "blue": scene.gaussians.get_xyz,
                    "red": hair_gs.ref_strand_root,
                },
                title="Gaussian (B) and GT \n Strand Root (R)",
            )
            p.subplot(0, 1)
            pv_visualize(
                plotter=p,
                point_clouds={"blue": hair_gs._endpoints},
                title="Endpoints of converted Gaussian lines (B)",
            )
        if gp.vis3d:
            p.subplot(0, 2)
            pv_visualize(
                plotter=p,
                point_clouds={
                    "blue": hair_gs._endpoints[hair_gs.strand_root_endpoint_idx]
                },
                title="Initialized Strand Root",
            )
            p.link_views()
            p.show()

        # Images
        cameras = scene.getCameras()
        images_dict = render_image_dict_from_cameras(hair_gs, cameras)
        training_info.composed_image = create_subplots_from_dict(
            images_dict, image_w=1920, image_h=1080
        )
        if gp.vis2d:
            cv2.imshow(
                f"Image Grid",
                cv2.cvtColor(training_info.composed_image, cv2.COLOR_RGB2BGR),
            )
            cv2.waitKey(0)

        # Compute metrics
        if scene.gt is not None:
            pred = compute_eval_data_from_hair_gs(hair_gs)
            training_info.eval_metrics, training_info.eval_thresholds = compute_metrics(
                pred=pred,
                gt=scene.gt,
                bidirectional=op.bidirectional_eval,
            )
        logger.log(training_info, hair_gs)

        # Merging loop
        for i in range(1, op.iterations + 1):
            training_info.iter = scene.loaded_iter + i
            start = time.time()
            endpoint_pairs_to_merge = hair_gs.compute_endpoint_pair_to_merge()
            training_info.densification_info["merged_segments"] = (
                endpoint_pairs_to_merge.shape[0]
            )

            # 3D visualization
            if args.vis3d:
                if i == 1:
                    # Create new plotter
                    p = pv.Plotter()
                    p.set_background("black")
                    polydata = get_joints_and_segments_from_hair_gs(
                        hair_gs, non_transparent=True
                    )
                    p.add_mesh(polydata, show_scalar_bar=False, rgba=True)
                else:
                    # Append current plot to previously created plotter
                    p.subplot(0, 1)
                    polydata = get_joints_and_segments_from_hair_gs(
                        hair_gs, non_transparent=True
                    )
                    p.add_mesh(polydata, show_scalar_bar=False, rgba=True)
                    p.link_views()
                p.show()
                # Create new plotter for next iteration
                if endpoint_pairs_to_merge.shape[0] > 0:
                    p = pv.Plotter(shape=(1, 2), border=False)
                    p.set_background("black")
                    p.subplot(0, 0)
                    polydata = get_joints_and_segments_from_hair_gs(
                        hair_gs, non_transparent=True
                    )
                    p.add_mesh(polydata, show_scalar_bar=False, rgba=True)
                    pv_visualize(
                        plotter=p,
                        title=f"Merge step {i}",
                        lines={
                            "white": hair_gs._endpoints[
                                endpoint_pairs_to_merge
                            ].flatten(start_dim=0, end_dim=1)
                        },
                    )

            # Break if no more pairs to merge
            if endpoint_pairs_to_merge.shape[0] == 0:
                logger.log(training_info, hair_gs)
                break

            hair_gs.merge_endpoint_pairs(endpoint_pairs_to_merge)
            hair_gs.compute_strands_info()
            training_info.elapsed_time = time.time() - start

            # update render image dict
            images_dict = render_image_dict_from_cameras(hair_gs, cameras)
            training_info.composed_image = create_subplots_from_dict(
                images_dict, image_w=1920, image_h=1080
            )

            # Compute metrics
            if scene.gt is not None:
                pred = compute_eval_data_from_hair_gs(hair_gs)
                training_info.eval_metrics, training_info.eval_thresholds = (
                    compute_metrics(
                        pred=pred,
                        gt=scene.gt,
                        bidirectional=op.bidirectional_eval,
                    )
                )

            # Log
            logger.log(training_info, hair_gs)

        scene.gaussians = hair_gs
        scene.save(i)
        print(
            f"Merge completed, gaussians saved to {scene.model_path}/point_cloud/iteration_{scene.loaded_iter + i}/point_cloud.ply"
        )
