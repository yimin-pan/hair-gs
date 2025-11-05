"""
Script to parse NeRSemble dataset into COLMAP format output.
"""

import os
from argparse import ArgumentParser
import json
import shutil

from tqdm import tqdm
import cv2
import numpy as np
import torch
import pyvista as pv
from dreifus.pyvista import (
    Pose,
    PoseType,
    Intrinsics,
    CameraCoordinateConvention,
)

from data import (
    HeadReconstruction,
    generate_colmap_data,
    write_cameras_binary,
    write_images_binary,
    write_points3D_binary,
    Camera as ColmapCamera,
)
from utils import estimate_orientation_field
from scene.flame import FLAME

HAIR_CLASS_ID = 14
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
RAW_DATASET_PATH = os.path.join(SCRIPT_PATH, "../dataset/raw/nersemble")
PARSED_DATASET_PATH = os.path.join(SCRIPT_PATH, "../dataset/parsed/nersemble")

FLAME_MODEL_PATH = os.path.join(SCRIPT_PATH, "../dataset/FLAME/flame2023.pkl")
STATIC_LANDMARK_EMBEDDING_PATH = os.path.join(
    SCRIPT_PATH, "../dataset/FLAME/flame_static_embedding.pkl"
)
DYNAMIC_LANDMARK_EMBEDDING_PATH = os.path.join(
    SCRIPT_PATH, "../dataset/FLAME/flame_dynamic_embedding.npy"
)
FLAME_MASK_PATH = os.path.join(SCRIPT_PATH, "../dataset/FLAME/FLAME_masks.pkl")

if __name__ == "__main__":
    parser = ArgumentParser(
        "Generate data needed to optimize GS from public hair datasets"
    )
    parser.add_argument(
        "--participant_id",
        "-i",
        default=None,
        type=int,
        help="ID of the participant, if None data for all subjects are generated",
    )
    parser.add_argument(
        "--flame_model_path",
        type=str,
        default=FLAME_MODEL_PATH,
        help="flame model path",
    )
    parser.add_argument(
        "--static_landmark_embedding_path",
        type=str,
        default=STATIC_LANDMARK_EMBEDDING_PATH,
        help="Static landmark embeddings path for FLAME",
    )
    parser.add_argument(
        "--dynamic_landmark_embedding_path",
        type=str,
        default=DYNAMIC_LANDMARK_EMBEDDING_PATH,
        help="Dynamic contour embedding path for FLAME",
    )
    parser.add_argument(
        "--flame_mask_path",
        type=str,
        default=FLAME_MASK_PATH,
        help="FLAME mask path",
    )
    parser.add_argument(
        "--shape_params", type=int, default=300, help="the number of shape parameters"
    )
    parser.add_argument(
        "--expression_params",
        type=int,
        default=100,
        help="the number of expression parameters",
    )
    parser.add_argument(
        "--pose_params", type=int, default=6, help="the number of pose parameters"
    )
    parser.add_argument(
        "--use_face_contour",
        default=True,
        type=bool,
        help="If true apply the landmark loss on also on the face contour.",
    )
    parser.add_argument(
        "--use_3D_translation",
        default=True,  # Flase for RingNet project
        type=bool,
        help="If true apply the landmark loss on also on the face contour.",
    )
    parser.add_argument(
        "--optimize_eyeballpose",
        default=True,  # False for For RingNet project
        type=bool,
        help="If true optimize for the eyeball pose.",
    )
    parser.add_argument(
        "--optimize_neckpose",
        default=True,  # False For RingNet project
        type=bool,
        help="If true optimize for the neck pose.",
    )
    parser.add_argument(
        "--num_worker", type=int, default=4, help="pytorch number worker."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Training batch size."
    )
    parser.add_argument("--ring_margin", type=float, default=0.5, help="ring margin.")
    parser.add_argument(
        "--ring_loss_weight", type=float, default=1.0, help="weight on ring loss."
    )
    args = parser.parse_args()
    flame = FLAME(args)
    flame_faces = flame.faces

    list_ids = None

    if args.participant_id is not None:
        list_ids = [args.participant_id]
    else:
        list_ids = [f.name for f in os.scandir(RAW_DATASET_PATH) if f.is_dir()]

    for participant_id in tqdm(list_ids):
        # check if subject folder (tum dataset is not consistent with the ids)
        participant_id = str(participant_id).zfill(3)
        participant_folder = os.path.join(RAW_DATASET_PATH, participant_id)
        if not os.path.exists(participant_folder):
            continue
        # delete old folder if exists
        output = os.path.join(PARSED_DATASET_PATH, participant_id)
        if os.path.exists(output):
            shutil.rmtree(output)

        # Cameras
        resolution = (0, 0)
        world_2_cam_poses = {}
        colmap_cameras = {}
        Es = {}
        source_camera_path = os.path.join(
            participant_folder, "calibration/camera_params.json"
        )
        camera_params = json.load(open(source_camera_path))
        intrinsics = Intrinsics(camera_params["intrinsics"])
        intrinsics = intrinsics.rescale(0.5)
        for cam_id, world_2_cam_pose in camera_params["world_2_cam"].items():
            if resolution == (0, 0):
                image = cv2.imread(
                    os.path.join(
                        participant_folder,
                        "sequences",
                        "EXP-1-head",
                        "timesteps",
                        "frame_00000",
                        "images-2x",
                        f"cam_{cam_id}.jpg",
                    )
                )
                resolution = image.shape[:2]
            cam_id = int(cam_id)
            world_2_cam_pose_ = Pose(
                world_2_cam_pose,
                camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
                pose_type=PoseType.WORLD_2_CAM,
            )
            world_2_cam_poses[cam_id] = world_2_cam_pose_
            fx = int(intrinsics.fx)
            fy = int(intrinsics.fy)
            cx = int(intrinsics.cx)
            cy = int(intrinsics.cy)
            colmap_cameras[cam_id] = ColmapCamera(
                id=cam_id,
                model="PINHOLE",
                width=resolution[1],
                height=resolution[0],
                params=[fx, fy, cx, cy],
            )
            Es[cam_id] = world_2_cam_pose_.numpy()

        # Load masks
        masks = {}
        alpha_maps = {}
        target_mask_path = os.path.join(output, "masks")
        os.makedirs(target_mask_path, exist_ok=True)
        alpha_map_folder = os.path.join(
            participant_folder,
            "sequences",
            "EXP-1-head",
            "timesteps",
            "frame_00000",
            "alpha_map",
        )
        segmentation_folder = os.path.join(
            participant_folder,
            "sequences",
            "EXP-1-head",
            "timesteps",
            "frame_00000",
            "facer_segmentation_masks",
        )
        try:
            for cam_id in colmap_cameras.keys():
                path = os.path.join(alpha_map_folder, f"cam_{cam_id}.png")
                alpha_map = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                alpha_map = cv2.resize(alpha_map, (resolution[1], resolution[0]))
                alpha_maps[cam_id] = alpha_map
                path = os.path.join(
                    segmentation_folder, f"segmentation_cam_{cam_id}.png"
                )
                segmentation = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                # mask as intersection of alpha map and dilated hair segmentation mask
                bg_segmentation = segmentation == 0
                hair_segmentation = segmentation == HAIR_CLASS_ID
                bg_or_hair_segmentation = bg_segmentation | hair_segmentation
                segmentation = bg_or_hair_segmentation & (
                    cv2.dilate((hair_segmentation.view(np.uint8)), np.ones((13, 13)))
                ).astype(bool)
                segmentation = segmentation.astype(np.uint8) * 255
                segmentation = cv2.resize(segmentation, (resolution[1], resolution[0]))
                mask = (alpha_map > 0) * (segmentation > 0)
                mask = mask.astype(np.uint8) * 255
                masks[cam_id] = mask
                cv2.imwrite(os.path.join(target_mask_path, f"image_{cam_id}.png"), mask)
        except:
            print(f"Missing masks for participant {participant_id}, skipping")
            shutil.rmtree(output)
            continue

        # Images and orientation field
        images = {}
        target_image_path = os.path.join(output, "images")
        os.makedirs(target_image_path, exist_ok=True)
        target_orientation_path = os.path.join(output, "orientations")
        os.makedirs(target_orientation_path, exist_ok=True)
        source_images_folder = os.path.join(
            participant_folder,
            "sequences",
            "EXP-1-head",
            "timesteps",
            "frame_00000",
            "images-2x",
        )
        try:
            for cam_id in colmap_cameras.keys():
                path = os.path.join(source_images_folder, f"cam_{cam_id}.jpg")
                image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                image = (image * (alpha_maps[cam_id] / 255)[:, :, None]).astype(
                    np.uint8
                )  # remove background with alpha map
                images[cam_id] = image
                cv2.imwrite(
                    os.path.join(target_image_path, f"image_{cam_id}.png"),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                )
                orientation_field, confidence = estimate_orientation_field(image)
                orientation_field = orientation_field * 255 / np.pi
                orientation_field = orientation_field.astype(np.uint8)
                confidence = confidence * 255
                confidence = confidence.astype(np.uint8)
                cv2.imwrite(
                    os.path.join(
                        target_orientation_path, f"image_{cam_id}_orientation.png"
                    ),
                    orientation_field,
                )
                cv2.imwrite(
                    os.path.join(
                        target_orientation_path, f"image_{cam_id}_confidence.png"
                    ),
                    confidence,
                )
        except:
            print(f"Missing images for participant {participant_id}, skipping")
            shutil.rmtree(output)
            continue

        # Load fitted FLAME model
        flame_path = os.path.join(
            participant_folder,
            "sequences",
            "EXP-1-head",
            "annotations",
            "tracking",
            "FLAME2023_v2",
            "tracked_flame_params.npz",
        )
        flame_params = np.load(flame_path)
        with torch.no_grad():
            shape = torch.from_numpy(flame_params["shape"][0]).unsqueeze(0)
            expression = torch.from_numpy(flame_params["expression"][0]).unsqueeze(0)
            pose = torch.tensor([0, 0, 0, 0, 0, 0]).unsqueeze(0)
            neck = torch.from_numpy(flame_params["neck"][0]).unsqueeze(0)
            eyes = torch.from_numpy(flame_params["eyes"][0]).unsqueeze(0)
            flame_vertices, _ = flame.forward(
                shape_params=shape,  # We always assume the same shape params for all timesteps
                expression_params=expression,
                pose_params=pose,
                neck_pose=neck,
                eye_pose=eyes,
                transl=None,
            )
            B = flame_vertices.shape[0]
            V = flame_vertices.shape[1]
            model_transformations = torch.stack(
                [
                    torch.from_numpy(
                        Pose.from_euler(
                            flame_params["rotation"][0],
                            flame_params["translation"][0],
                            "XYZ",
                        )
                    )
                ]
            )
            model_transformations[:, :3, :3] *= torch.from_numpy(
                flame_params["scale"][0]
            )
            flame_vertices = torch.cat([flame_vertices, torch.ones((B, V, 1))], dim=-1)
            flame_vertices = torch.bmm(
                flame_vertices, model_transformations.permute(0, 2, 1)
            )
            flame_vertices = flame_vertices[..., :3]
            flame_vertices = flame_vertices[0].cpu().numpy()
            scalp_vertices = flame_vertices[flame.scalp_mask]

        head_reconstruction = HeadReconstruction(
            head_verts=flame_vertices, scalp_verts=scalp_vertices
        )
        np.savez(
            os.path.join(output, "head_reconstruction_data.npz"),
            head_verts=head_reconstruction.head_verts,
            scalp_verts=head_reconstruction.scalp_verts,
        )

        # save flame_vertices and flame_faces into ply mesh for later visualization
        faces_formatted = np.hstack(
            [np.full((flame_faces.shape[0], 1), 3), flame_faces]
        )
        pv_mesh = pv.PolyData(flame_vertices, faces_formatted)
        pv_mesh.save(os.path.join(output, "head_mesh.ply"))

        # generate and save COLMAP format scene: use only head vertices as initial point cloud for GS
        flame_colors = np.ones_like(flame_vertices) * 0.5
        images, points_3d = generate_colmap_data(
            colmap_cameras, Es, flame_vertices, flame_colors
        )
        sparse_0_path = os.path.join(output, "sparse", "0")
        os.makedirs(sparse_0_path, exist_ok=True)
        write_cameras_binary(colmap_cameras, os.path.join(sparse_0_path, "cameras.bin"))
        write_images_binary(images, os.path.join(sparse_0_path, "images.bin"))
        write_points3D_binary(points_3d, os.path.join(sparse_0_path, "points3D.bin"))
