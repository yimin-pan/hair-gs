from typing import List, Dict
from multiprocessing import Pool, Manager

import numpy as np
from scipy.spatial import cKDTree

from utils import SH2RGB
from data import HairEvalData
from scene.hair_gaussian_model import HairGaussianModel


def pct_matched_points(
    p1: HairEvalData,
    p2: HairEvalData,
    dist_th: float,
    angle_th: float,
    bidirectional: bool = False,
    compute_strand_consistency: bool = False,
    metric_dict: Dict[str, List] = {},
    metric_key: str = "precision",
):
    """
    Compute the percentage of matched points from p1 to p2 given a distance and angle threshold.
    Note that the compute_strand_matching expects the points in HairEvalData to be arranged in strands.
    """

    th_key = f"{dist_th}m&{angle_th}°"
    cos_sim_th = np.cos(np.deg2rad(angle_th))
    n_points = p1.points.shape[0]
    position_tree = cKDTree(p2.points)
    p1_matched_points = position_tree.query_ball_point(
        p1.points, workers=-1, r=dist_th
    )  # if this takes forever => you have run out of RAM
    strand_stats = {}
    count = 0
    # iterate over points in p1
    for i in range(n_points):
        if compute_strand_consistency:
            curr_point_strand_id = p1.points_id_to_strand_id[i]
            if curr_point_strand_id not in strand_stats:
                strand_stats[curr_point_strand_id] = {
                    "matched_points_strand_id": [],
                    "strand_points": 0,
                }
            strand_stats[curr_point_strand_id]["strand_points"] += 1
        # match by dist
        curr_p1_matched_points = p1_matched_points[i]
        if len(curr_p1_matched_points) > 0:
            # match by angle
            curr_p1_matched_points = np.array(curr_p1_matched_points)
            p2_directions = p2.directions[curr_p1_matched_points]
            curr_point_dir = p1.directions[i].reshape(1, 3)
            dir_dot_prod = (curr_point_dir @ p2_directions.T).squeeze()
            if bidirectional:
                dir_dot_prod = np.abs(dir_dot_prod)
            dir_matching_mask = dir_dot_prod >= cos_sim_th
            if np.any(dir_matching_mask):
                count += 1
                # accumulate strand matching statistics
                if compute_strand_consistency:
                    curr_p1_matched_points = curr_p1_matched_points[dir_matching_mask]
                    curr_p1_matched_points_strand_ids = p2.points_id_to_strand_id[
                        curr_p1_matched_points
                    ]
                    u = np.unique(curr_p1_matched_points_strand_ids)
                    strand_stats[curr_point_strand_id][
                        "matched_points_strand_id"
                    ].extend(u)

    matching_ratio = count / n_points
    metric_dict[metric_key][th_key] = matching_ratio
    # compute strand matching ratio
    if compute_strand_consistency:
        strand_matching_count = 0
        num_strands = len(strand_stats)
        for v in strand_stats.values():
            num_points = v["strand_points"]
            matched_strands = v["matched_points_strand_id"]
            if len(matched_strands) > 0:
                matched_strands = np.array(matched_strands)
                _, counts = np.unique(matched_strands, return_counts=True)
                max_count = np.max(counts)
                strand_matching_count += max_count / num_points
        strand_matching_ratio = strand_matching_count / num_strands
        metric_dict["strand_consistency"][th_key] = strand_matching_ratio


def compute_metrics(
    pred: HairEvalData,
    gt: HairEvalData,
    dist_ths: List[float] = [2e-3, 3e-3, 4e-3, 4e-3],
    angle_ths: List[float] = [20, 30, 40, 90],
    metrics: List[str] = ["precision", "recall", "f1", "strand_consistency"],
    bidirectional: bool = False,
    processes: int = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute the precision, recall and f1-score metrics for the predicted oriented point cloud in the given thresholds.
    The directions are assumed to be already normalized, the distance thresholds are given in meters and the angle thresholds in degrees.
    The method used to estimate the metrics depends on the type of the input oriented point clouds.
    This is a parallelized version, if you are running out of memory due to copying, reduce the number of processes.
    """
    compute_strand_consistency = (
        "strand_consistency" in metrics
        and pred.points_id_to_strand_id is not None
        and gt.points_id_to_strand_id is not None
    )
    # thresholds
    thresholds = []
    for dist_th, angle_th in zip(dist_ths, angle_ths):
        thresholds.append(f"{dist_th}m&{angle_th}°")
    # prepare shared memory dictionary
    manager = Manager()
    metrics_dict = manager.dict()
    for metric in metrics:
        metrics_dict[metric] = manager.dict()
    # prepare arguments
    list_arg_tuples = []
    if "precision" in metrics:
        for dist_th, angle_th in zip(dist_ths, angle_ths):
            list_arg_tuples.append(
                (
                    pred,
                    gt,
                    dist_th,
                    angle_th,
                    bidirectional,
                    False,
                    metrics_dict,
                    "precision",
                )
            )
    if "recall" in metrics:
        for dist_th, angle_th in zip(dist_ths, angle_ths):
            list_arg_tuples.append(
                (
                    gt,
                    pred,
                    dist_th,
                    angle_th,
                    bidirectional,
                    compute_strand_consistency,
                    metrics_dict,
                    "recall",
                )
            )
    # compute metrics
    with Pool(processes=8 if processes is None else processes) as pool:
        pool.starmap(pct_matched_points, list_arg_tuples)
    if (
        "f1" in metrics_dict
        and "precision" in metrics_dict
        and "recall" in metrics_dict
    ):
        for th_key in thresholds:
            precision = metrics_dict["precision"][th_key]
            recall = metrics_dict["recall"][th_key]
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if precision + recall > 0
                else 0
            )
            metrics_dict["f1"][th_key] = f1
    # change the keys name
    final_metrics_dict = {}
    for metric_key, metric_dict in metrics_dict.items():
        new_metric_key = metric_key + "(b)" if bidirectional else metric_key
        metric_list = []
        for th_key in thresholds:
            if th_key in metric_dict:
                metric_list.append(metric_dict[th_key])
        final_metrics_dict[new_metric_key] = np.array(metric_list)
    return final_metrics_dict, thresholds
