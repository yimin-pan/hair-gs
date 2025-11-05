"""
Script to parse the USC-HairSalon dataset into COLMAP format output for 3DGS. Images are generated with OpenGL renderer with virtual cameras spawn across a sphere around the target hair.

Official USC-HairSalon dataset website: https://huliwenkidkid.github.io/liwenhu.github.io/
"""

import os
import shutil
import numpy as np
import cv2
from argparse import ArgumentParser
from tqdm import tqdm

from data import (
    hair_data_load_callbacks,
    head_data_load_callbacks,
    save_head_reconstruction_data_npz,
    save_hair_eval_data_npz,
    generate_colmap_data,
    write_cameras_binary,
    write_images_binary,
    write_points3D_binary,
)
from data import save_hair_eval_data_npz, generate_colmap_data
from utils import (
    estimate_orientation_field,
    generate_cameras,
    colmap_camera_to_projection_matrix,
    opencv_to_opengl_view_matrix,
    plot_cameras,
)
from scene.OpenGLRenderer import *

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
RAW_DATASET_PATH = os.path.join(SCRIPT_PATH, "../dataset/raw/hairstyles")
PARSED_DATASET_PATH = os.path.join(SCRIPT_PATH, "../dataset/parsed/usc_hairsalon")

if __name__ == "__main__":
    parser = ArgumentParser(
        "Generate data needed to optimize GS from public hair datasets"
    )
    parser.add_argument(
        "--strand_id",
        "-i",
        default=None,
        type=int,
        help="ID of the strand to use, if None all strands are generated",
    )
    parser.add_argument(
        "--pct_strands",
        "-p",
        default=100,
        type=float,
        help="Percentage of number of strands to use",
    )
    parser.add_argument(
        "--line_width",
        "-w",
        default=1,
        type=float,
        help="Width of the rendered lines (pixels)",
    )
    parser.add_argument(
        "--use_gt_hair_verts",
        action="store_true",
        default=False,
        help="Whether to use hair vertices as initialization for the GS optimization",
    )
    parser.add_argument(
        "--use_strand_root_verts",
        action="store_true",
        default=False,
        help="Whether to use strand root vertices as initialization for the GS optimization",
    )
    parser.add_argument(
        "--hsv",
        action="store_true",
        help="Iterate over hsv color space for each strand",
    )
    parser.add_argument(
        "--cam_z", default=0.5, type=float, help="Z coordinate of the camera"
    )

    parser.add_argument(
        "--show_camera",
        action="store_true",
        help="Visualize the generated cameras and frustums",
    )
    parser.add_argument("--cameras", default=16, type=int, help="Number of cameras")
    parser.add_argument("--height", default=1000, type=int, help="Height of the image")
    parser.add_argument("--width", default=1000, type=int, help="Width of the image")
    args = parser.parse_args()
    if args.hsv:
        PARSED_DATASET_PATH = PARSED_DATASET_PATH + "_hsv"

    # load face
    face_file_path = os.path.join(RAW_DATASET_PATH, "head_model.obj")
    face = head_data_load_callbacks["usc_hair_salon"](
        face_file_path, normal_required=True
    )
    renderer = OpenGLRenderer(resolution=(args.width, args.height))
    lighting = OpenGLLighting(
        light_pos=np.array([0, 5, 5]),
        ambient_color=np.array([1, 1, 1, 1]),
        diffuse_color=np.array([1, 1, 1, 1]),
    )
    renderer.lighting = lighting
    # black face
    face_model_black = OpenGLModel(
        face.verts,
        faces=face.faces,
        colors=face.colors,
        normals=face.normals,
        use_lighting=False,
    )
    face_model_black.colors = np.zeros_like(face_model_black.colors)
    renderer.models.append(face_model_black)
    # colored face
    face_model = OpenGLModel(
        face.verts,
        faces=face.faces,
        colors=face.colors,
        normals=face.normals,
        use_lighting=True,
        ka=0.5,
        kd=0.5,
    )
    # face_model.colors = np.zeros_like(face_model.colors)
    renderer.models.append(face_model)
    renderer.setup()

    # load corresponding hairs
    list_ids = [args.strand_id] if args.strand_id is not None else range(1, 515)
    first_hair = True
    for strand_id in tqdm(list_ids):
        # check if strand exists (usc_hair_salon dataset is not consistent with the ids)
        strand_id_str = str(strand_id).zfill(5)
        hair_file_path = os.path.join(RAW_DATASET_PATH, f"strands{strand_id_str}.data")
        if not os.path.exists(hair_file_path):
            continue

        # delete old folder if exists
        output = os.path.join(PARSED_DATASET_PATH, strand_id_str)
        if os.path.exists(output):
            shutil.rmtree(output)

        # add hair to renderer
        hair = hair_data_load_callbacks["usc_hair_salon"](
            hair_file_path,
            normal_required=True,
            hsv_spectre_color=args.hsv,
            pct_strands=args.pct_strands,
        )
        hair_model = OpenGLModel(
            hair.verts,
            edges=hair.edges,
            colors=hair.colors,
            normals=hair.normals,
            use_lighting=True,
            line_width=args.line_width,
            ka=0.5,
            kd=0.5,
        )
        if first_hair:
            renderer.models.append(hair_model)
            first_hair = False
        else:
            renderer.models[-1] = hair_model
        renderer.setup_meshes()

        # generate cameras in OpenCV (COLMAP) coordinate system, flipped y and z wrt OpenGL
        cam_pose = np.eye(4)
        delta_y = hair.verts[:, 1].max() - hair.verts[:, 1].min()
        cam_y = (hair.verts[:, 1].max() + hair.verts[:, 1].min()) / 2
        cam_pose[:3, 3] = [0, cam_y, args.cam_z]
        cam_pose[:3, 1:3] *= -1
        anchor_pos = np.array([0, cam_y, 0])
        colmap_cameras, Es = generate_cameras(
            args.cameras,
            args.height,
            args.width,
            cam_pose=cam_pose,
            anchor_pos=anchor_pos,
            offset=args.cam_z,
        )

        # render all images
        image_path = os.path.join(output, "images")
        os.makedirs(image_path, exist_ok=True)
        orientation_path = os.path.join(output, "orientations")
        os.makedirs(orientation_path, exist_ok=True)
        mask_path = os.path.join(output, "masks")
        os.makedirs(mask_path, exist_ok=True)
        for cam_id, cam in colmap_cameras.items():
            # rendered image
            projection = colmap_camera_to_projection_matrix(cam)
            view = opencv_to_opengl_view_matrix(Es[cam_id])
            renderer.camera = OpenGLCamera(view, projection)
            renderer.setup_camera()
            rendered_img = renderer.render(mesh_indices=[1, 2])
            cv2.imwrite(
                os.path.join(image_path, f"image_{cam_id}.png"),
                cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR),
            )
            # orientation field estimation
            orientation_field, confidence = estimate_orientation_field(rendered_img)
            orientation_field = orientation_field * 255 / np.pi
            orientation_field = orientation_field.astype(np.uint8)
            confidence = confidence * 255
            confidence = confidence.astype(np.uint8)
            cv2.imwrite(
                os.path.join(orientation_path, f"image_{cam_id}_orientation.png"),
                orientation_field,
            )
            cv2.imwrite(
                os.path.join(orientation_path, f"image_{cam_id}_confidence.png"),
                confidence,
            )
            # hair mask
            rendered_img = renderer.render(mesh_indices=[0, 2])
            h, w, _ = rendered_img.shape
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            non_black_pixels = np.any(rendered_img != [0, 0, 0], axis=2)
            binary_mask[non_black_pixels] = 255
            cv2.imwrite(os.path.join(mask_path, f"image_{cam_id}.png"), binary_mask)

        save_hair_eval_data_npz(os.path.join(output, "hair_eval_data.npz"), hair)
        save_head_reconstruction_data_npz(
            os.path.join(output, "head_reconstruction_data.npz"), hair, face
        )

        # generate and save COLMAP format scene: use only head vertices as initial point cloud for GS
        points = None
        colors = None
        if args.use_gt_hair_verts:
            points = hair.verts
            colors = hair.colors
        elif args.use_strand_root_verts:
            points = hair.verts[hair.strand_root_idx]
            colors = hair.colors[hair.strand_root_idx]
        else:
            points = face.verts
            colors = face.colors
        colmap_images, colmap_points_3d = generate_colmap_data(
            colmap_cameras, Es, points, colors
        )
        sparse_0_path = os.path.join(output, "sparse", "0")
        os.makedirs(sparse_0_path, exist_ok=True)
        write_cameras_binary(colmap_cameras, os.path.join(sparse_0_path, "cameras.bin"))
        write_images_binary(colmap_images, os.path.join(sparse_0_path, "images.bin"))
        write_points3D_binary(
            colmap_points_3d, os.path.join(sparse_0_path, "points3D.bin")
        )

        if args.show_camera:
            plot_cameras(colmap_cameras, Es)
