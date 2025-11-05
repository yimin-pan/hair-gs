"""
Utility and auxiliar function that needs speed up are implemented here in Cython.
"""
import cython
import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound, cdivision
from libc.stdint cimport int64_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef compute_strands(np.ndarray[np.int32_t, ndim=2] endpoint_pairs, np.ndarray[np.float32_t, ndim=2] endpoints, np.ndarray[np.float32_t, ndim=1] ref):
    cdef int[:,::1] endpoint_pairs_ = endpoint_pairs
    # Create a map from endpoint id to endpoint_pairs row id
    cdef int[:,::1] id_to_row_id_ = -np.ones((endpoint_pairs.max() + 1, 2), dtype=np.int32)  # each can map to 2 rows at max
    cdef int endpoint_pairs_num_rows = endpoint_pairs_.shape[0]
    cdef int i, j, id, col_
    for i in range(endpoint_pairs_num_rows):
        for j in range(2):
            id = endpoint_pairs_[i][j]
            col_ = 0 if id_to_row_id_[id][0] == -1 else 1
            id_to_row_id_[id][col_] = i
    # Compute strand endpoints (tip/root)
    cdef np.ndarray[np.int32_t, ndim=1] endpoints_ids, strand_endpoint_id
    cdef np.ndarray[np.int64_t, ndim=1] counts
    endpoints_ids, counts = np.unique(endpoint_pairs, return_counts=True)
    strand_endpoint_id = endpoints_ids[counts == 1]
    cdef int[::1] strand_endpoint_id_ = strand_endpoint_id
    # Aggregate strand points and directions (bidirectional)
    cdef list all_segments_id = []
    cdef list all_strand_roots_id = []
    cdef int counter, strand_endpoint_shape, curr_point_id, next_point_id, strand_start_id, row_id
    counter = 0
    strand_endpoint_shape = strand_endpoint_id_.shape[0]
    cdef int[::1] visited_ = np.zeros(strand_endpoint_id.max() + 1, dtype=np.int32)
    cdef int[:,::1] strand_endpoint_pair_ = -np.ones((strand_endpoint_id.shape[0] // 2, 2),
                                    dtype=np.int32)  # array containing the pair of strand endpoints
    cdef np.ndarray[np.int32_t, ndim=1] strand_endpoint_id_to_row = np.zeros(strand_endpoint_id.max() + 1,
                                         dtype=np.int32)  # map strand endpoint id to visited strand_endpoint_pair row id
    strand_endpoint_id_to_row[strand_endpoint_id] = np.arange(strand_endpoint_shape)
    cdef int[::1] strand_endpoint_id_to_row_ = strand_endpoint_id_to_row
    cdef list strand_segments_id
    cdef np.ndarray[np.int32_t, ndim=1] endpoint_pair_row
    for i in range(strand_endpoint_shape):
        strand_segments_id = []
        strand_start_id = strand_endpoint_id_[i]
        curr_point_id = strand_start_id
        row_id = id_to_row_id_[curr_point_id][0]  # strand endpoint appears only in one row
        while row_id != -1:
            next_point_id = endpoint_pairs_[row_id][0] if endpoint_pairs_[row_id][0] != curr_point_id else endpoint_pairs_[row_id][1]
            strand_segments_id.append([curr_point_id, next_point_id])
            curr_point_id = next_point_id
            row_id = id_to_row_id_[curr_point_id][0] if id_to_row_id_[curr_point_id][0] != row_id else \
            id_to_row_id_[curr_point_id][1]
            all_segments_id.append(strand_segments_id)
        all_strand_roots_id.append(strand_start_id)
        if not visited_[strand_start_id]:
            strand_endpoint_pair_[counter][0] = strand_endpoint_id_to_row_[strand_start_id]
            strand_endpoint_pair_[counter][1] = strand_endpoint_id_to_row_[curr_point_id]
            visited_[strand_start_id] = True
            visited_[curr_point_id] = True
            counter += 1
    # Calculate the correct direction for each strand based on the distance to the reference point
    cdef np.ndarray[np.int32_t, ndim=2] strand_endpoint_pair = np.asarray(strand_endpoint_pair_)
    cdef np.ndarray[np.int32_t, ndim=1] all_strand_roots_id_np = np.array(all_strand_roots_id)
    cdef np.ndarray[np.float32_t, ndim=2] strand_root_pos = endpoints[all_strand_roots_id_np]
    cdef np.ndarray[np.float32_t, ndim=1] dists = np.linalg.norm(strand_root_pos - ref, axis=1, keepdims=True).astype(np.float32).squeeze()
    cdef np.ndarray[np.float32_t, ndim=2] strand_endpoint_dists = dists[strand_endpoint_pair]
    cdef np.ndarray[np.int32_t, ndim=1] selected_dir = np.where(strand_endpoint_dists[:, 0] <= strand_endpoint_dists[:, 1], strand_endpoint_pair[:, 0],
                            strand_endpoint_pair[:, 1])
    # Select correct points and directions
    cdef np.ndarray[list, ndim=1] all_segments_id_np_list = np.array(all_segments_id, dtype=list)
    all_segments_id_np_list = all_segments_id_np_list[selected_dir]
    cdef np.ndarray[np.int32_t, ndim=2] all_segments_id_np = np.concatenate(all_segments_id_np_list)
    all_strand_roots_id_np = all_strand_roots_id_np[selected_dir]
    return all_segments_id_np, all_strand_roots_id_np


# Disable bounds checking and negative index wraparound for speed
@boundscheck(False)
@wraparound(False)
@cdivision(True)
def filter_strand_list_segments(object strands_list not None):
    """
    Filters strands_list to extract pairs of consecutive 2D integers.

    Parameters:
    - strands_list: 1D NumPy array of objects, each being a 2D NumPy array of integers with shape (N, 2).

    Returns:
    - NumPy array of shape (total_pairs, 2, 2), where each pair consists of two consecutive 2D integers from the strands.
    """
    cdef Py_ssize_t n_strands = len(strands_list)
    cdef Py_ssize_t total_pairs = 0
    cdef Py_ssize_t j, i
    cdef Py_ssize_t strand_len
    cdef object strand_obj
    cdef np.ndarray strand
    cdef int64_t[:, :] strand_view

    # First Pass: Count the total number of pairs
    for j in range(n_strands):
        strand_obj = strands_list[j]
        strand_len = strand_obj.shape[0]
        if strand_len >= 2:
            total_pairs += (strand_len - 1)

    # Allocate Output Array
    # Shape: (total_pairs, 2, 2)
    cdef np.ndarray[np.int64_t, ndim=3] output = np.empty((total_pairs, 2, 2), dtype=np.int64)
    cdef int64_t[:, :, :] out_view = output
    cdef Py_ssize_t current = 0

    # Second Pass: Populate the Output Array
    for j in range(n_strands):
        strand = strands_list[j]
        strand_len = strand.shape[0]
        if strand_len >= 2:
            strand_view = strand
            for i in range(strand_len - 1):
                out_view[current, 0, 0] = strand_view[i, 0]
                out_view[current, 0, 1] = strand_view[i, 1]
                out_view[current, 1, 0] = strand_view[i + 1, 0]
                out_view[current, 1, 1] = strand_view[i + 1, 1]
                current += 1

    return output
