"""
Script used to evaluate the reconstruction results from different methods.
The argument -pt could be: gs, usc_hair_salon, strand_integration, neural_haircut.
"""

import os

from arguments import ArgumentParser

from data import eval_data_loading_callbacks
from loss import compute_metrics

if __name__ == "__main__":
    parser = ArgumentParser("Evaluation of reconstruction results")
    parser.add_argument(
        "--source_data_path", "-s", type=str, help="Path to the gt data", required=True
    )
    parser.add_argument(
        "--pred_data_path",
        "-p",
        type=str,
        help="Path to prediction data",
        required=True,
    )
    parser.add_argument(
        "--pred_data_type",
        "-pt",
        default="gs",
        type=str,
        help="Type of the prediction data",
    )
    parser.add_argument("--vis3d", action="store_true", help="Visualize the 3D data")
    args = parser.parse_args()

    if not args.pred_data_type in eval_data_loading_callbacks:
        raise ValueError(f"Evaluation data type {args.pred_data_type} not supported")

    gt_path = os.path.join(args.source_data_path, "hair_eval_data.npz")
    gt_data = eval_data_loading_callbacks["gt"](gt_path)
    print(f"Loaded GT data from {gt_path}")

    eval_data = eval_data_loading_callbacks[args.pred_data_type](args.pred_data_path)
    print(f"Loaded evaluation data from {args.pred_data_path}")

    if args.vis3d:
        import pyvista as pv

        plotter = pv.Plotter()
        eval_poly = pv.PolyData(eval_data.points)
        gt_poly = pv.PolyData(gt_data.points)
        plotter.add_mesh(eval_poly, color="red")
        plotter.add_mesh(gt_poly, color="green")
        plotter.add_axes()
        plotter.show()

    _, _, table = compute_metrics(
        eval_data, gt_data, bidirectional=True, return_table=True
    )
    print(table)
