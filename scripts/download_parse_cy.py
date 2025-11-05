"""
This script downloads and parse all the public Cem-Yuksel public hair models into COLMAP format output for 3DGS.

Official Cem-Yuksel dataset website: https://www.cemyuksel.com/research/hairmodels/
"""

import os
from argparse import ArgumentParser
from pathlib import Path
import shutil
import requests
import zipfile

from tqdm import tqdm
import pyvista as pv

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
from utils import (
    estimate_orientation_field,
    generate_cameras,
    colmap_camera_to_projection_matrix,
    opencv_to_opengl_view_matrix,
    plot_cameras,
)
from scene.OpenGLRenderer import *

HEAD_MODEL_URL = "https://www.cemyuksel.com/research/hairmodels/woman.zip"
HAIRSTYLES_URL = [
    "https://www.cemyuksel.com/research/hairmodels/wStraight.zip",
    "https://www.cemyuksel.com/research/hairmodels/wCurly.zip",
    "https://www.cemyuksel.com/research/hairmodels/wWavy.zip",
    "https://www.cemyuksel.com/research/hairmodels/wWavyThin.zip",
]
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_PATH = os.path.join(SCRIPT_PATH, "../dataset/raw/cem_yuksel")
PARSED_PATH = os.path.join(SCRIPT_PATH, "../dataset/parsed/cem_yuksel")


def download_extract_zip(download_url, download_path, extract_path):
    # Download the zip file
    if not os.path.exists(download_path):
        if not os.path.exists(os.path.dirname(download_path)):
            os.makedirs(os.path.dirname(download_path))

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "*/*",
        }

        response = requests.get(download_url, stream=True, headers=headers)
        response.raise_for_status()

        with open(download_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    # Extract the zip file
    if os.path.exists(download_path):
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)


if __name__ == "__main__":
    parser = ArgumentParser(
        "Generate data needed to optimize GS from public hair datasets"
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
        "--cam_z", default=0.3, type=float, help="Z coordinate of the camera"
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
        PARSED_PATH = PARSED_PATH + "_hsv"

    # Download the head model
    head_model_download_path = os.path.join(DOWNLOAD_PATH, "woman.zip")
    download_extract_zip(HEAD_MODEL_URL, head_model_download_path, DOWNLOAD_PATH)
    print("Head model downloaded and extracted into dataset/raw/cem_yuksel/")

    # Load face
    face_file_path = os.path.join(DOWNLOAD_PATH, "woman.obj")
    face = head_data_load_callbacks["cem_yuksel"](face_file_path)
    renderer = OpenGLRenderer(resolution=(args.width, args.height))
    lighting = OpenGLLighting(
        light_pos=np.array([0, 5, 5]),
        ambient_color=np.array([1, 1, 1, 1]),
        diffuse_color=np.array([1, 1, 1, 1]),
    )
    renderer.lighting = lighting
    face_model_black = OpenGLModel(
        face.verts,
        faces=face.faces,
        colors=np.zeros_like(face.colors),
        normals=face.normals,
        use_lighting=False,
    )
    renderer.models.append(face_model_black)
    face_model = OpenGLModel(
        face.verts,
        faces=face.faces,
        colors=face.colors,
        normals=face.normals,
        use_lighting=True,
        ka=0.5,
        kd=0.5,
    )
    renderer.models.append(face_model)
    renderer.setup()

    # For each hairstyle
    first_hair = True
    for hairstyle_url in tqdm(HAIRSTYLES_URL):
        # Download the hairstyle data
        hairstyle = os.path.basename(hairstyle_url).replace(".zip", "")
        hairstyle_downlaod_path = os.path.join(
            DOWNLOAD_PATH, os.path.basename(hairstyle_url)
        )
        download_extract_zip(hairstyle_url, hairstyle_downlaod_path, DOWNLOAD_PATH)

        # Delete old folder if exists
        output = os.path.join(PARSED_PATH, hairstyle)
        if os.path.exists(output):
            shutil.rmtree(output)
        os.makedirs(output, exist_ok=True)

        # Add hair to renderer
        hair = hair_data_load_callbacks["cem_yuksel"](
            os.path.join(DOWNLOAD_PATH, f"{hairstyle}.hair"), hsv_spectre_color=args.hsv
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

        # Generate cameras in OpenCV (COLMAP) coordinate system, flipped y and z wrt OpenGL
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

        # Render all images
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
        # Save rotated head mesh
        faces_formatted = np.hstack([np.full((face.faces.shape[0], 1), 3), face.faces])
        pv_mesh = pv.PolyData(face.verts, faces_formatted)
        pv_mesh.save(os.path.join(output, "head_mesh.ply"))

        # Generate and save COLMAP format scene: use only head vertices as initial point cloud for GS
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
