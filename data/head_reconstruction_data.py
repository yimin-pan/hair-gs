"""
Class for saving and loading head reconstruction data. This information is used during the optimization process.
"""

from typing import NamedTuple

import numpy as np

from .hair_data import HairData
from .head_data import HeadData


class HeadReconstruction(NamedTuple):
    head_verts: np.ndarray
    scalp_verts: np.ndarray


def save_head_reconstruction_data_npz(
    file_path: str,
    hair_data: HairData,
    head_data: HeadData,
):
    head_reconstruction = HeadReconstruction(
        head_verts=head_data.verts,
        scalp_verts=hair_data.verts[hair_data.strand_root_idx],
    )
    np.savez(
        file_path,
        head_verts=head_reconstruction.head_verts,
        scalp_verts=head_reconstruction.scalp_verts,
    )


def load_head_reconstruction_data_npz(path: str) -> HeadReconstruction:
    data = np.load(path)
    head_verts = data["head_verts"]
    scalp_verts = data["scalp_verts"]
    return HeadReconstruction(head_verts=head_verts, scalp_verts=scalp_verts)
