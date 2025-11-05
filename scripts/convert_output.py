"""
This script is meant for converting the output from different reconstruction methods to a visualizable format (edges/faces).
Currently supporting:
 - Hair-GS (ours)
 - Strand Integration: https://github.com/elerac/strand_integration
 - Neural Haircut: https://github.com/egorzakharov/NeuralHaircut
"""

import os

import numpy as np
import cv2
from arguments import ArgumentParser

from utils import save_ply_edges, save_ply_faces
from data import eval_data_loading_callbacks, HairEvalData

if __name__ == "__main__":
    parser = ArgumentParser("Convert GS output to visualizable ply lines")
    parser.add_argument("--input", "-i", type=str, help="Input ply file path")
    parser.add_argument(
        "--type", "-t", type=str, default="gs", help="Type of the input ply file"
    )
    parser.add_argument(
        "--edges",
        "-e",
        action="store_true",
        help="Save ply edges (polylines) instead of faces",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output file path"
    )
    args = parser.parse_args()

    # Handle output path
    output_path = args.output
    if output_path is None:
        input_dir = os.path.dirname(args.input)
        output_path = os.path.join(input_dir, "strands.ply")
        print(f"Output path not specified. Saving to {output_path}")
    else:
        print(f"Saving to {output_path}")

    vertices = None
    vertex_colors = None
    edges = None

    if not args.type in eval_data_loading_callbacks:
        raise ValueError(f"Type {args.type} not supported")

    data: HairEvalData = eval_data_loading_callbacks[args.type](args.input)
    vertices = data.points
    edges = data.edges
    if edges is None:
        raise ValueError("Edges are None")
    default_color = np.array([128, 128, 128], dtype=np.uint8)
    vertex_colors = np.tile(default_color, (vertices.shape[0], 1))

    # Assign different color for each strand
    if data.points_id_to_strand_id is not None:
        num_strands = data.points_id_to_strand_id.max() + 1
        hues = np.linspace(start=0, stop=180, num=num_strands)
        vertices_hue = hues[data.points_id_to_strand_id]
        vertices_hue = vertices_hue.astype(np.uint8)
        vertices_sv = np.ones((vertices_hue.shape[0], 2), dtype=np.uint8) * 255
        vertices_hsv = np.concatenate((vertices_hue[:, None], vertices_sv), axis=1)
        vertices_hsv = vertices_hsv[None, :, :]
        vertices_rgb = cv2.cvtColor(vertices_hsv, cv2.COLOR_HSV2RGB)
        vertex_colors = vertices_rgb.squeeze(0)

    # Save result as ply file
    if args.edges:
        save_ply_edges(vertices, vertex_colors, edges, file_path=output_path)
    else:
        save_ply_faces(vertices, vertex_colors, edges, file_path=output_path)

    print(f"Saved to {output_path}")
