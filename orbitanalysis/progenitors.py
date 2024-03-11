import time

import numpy as np
from orbitanalysis.utils import recenter_coordinates, myin1d


def get_tracked_ids(snapshot, n=100):

    slices = list(
        zip(snapshot.region_offsets[:-1], snapshot.region_offsets[1:]))
    region_coords = np.concatenate([
        recenter_coordinates(
            snapshot.coordinates[start:end]-p, snapshot.box_size)
        for (start, end), p in zip(slices, snapshot.region_positions)], axis=0)

    rads = np.sqrt(np.einsum('...i,...i', region_coords, region_coords))
    tracked_ids = [snapshot.ids[np.argsort(rads[start:end])[:n]+start]
                   for start, end in slices]
    offsets = np.cumsum([0] + [len(ids) for ids in tracked_ids])[:-1]
    return np.hstack(tracked_ids), offsets


def find_main_progenitors(halo_pids, halo_offsets, tracked_ids,
                          tracked_offsets):

    tracked_ids_, unique_inds = np.unique(tracked_ids, return_index=True)
    tracked_ids = -np.ones(len(tracked_ids), dtype=int)
    tracked_ids[unique_inds] = tracked_ids_

    halo_diffs = np.diff(halo_offsets)
    halo_lens = np.append(halo_diffs, len(halo_pids)-halo_offsets[-1])
    tracked_diffs = np.diff(tracked_offsets)
    tracked_lens = np.append(
        tracked_diffs, len(tracked_ids)-tracked_offsets[-1])

    halo_number = np.hstack([
        n * np.ones(hlen, dtype=int) for n, hlen in enumerate(halo_lens)])

    intersect_inds = np.where(np.in1d(tracked_ids, halo_pids, kind='table'))[0]
    tracked_ids_present = tracked_ids[intersect_inds]

    inds = myin1d(halo_pids, tracked_ids_present, kind='table')
    halo_numbers_progen_ = halo_number[inds]
    halo_numbers_progen = -np.ones(len(tracked_ids), dtype=int)
    halo_numbers_progen[intersect_inds] = halo_numbers_progen_
    halo_numbers_progen_split = np.split(
        halo_numbers_progen, np.cumsum(tracked_lens))[:-1]
    halo_numbers_progen_split_noneg = [
        hnums[hnums != -1] for hnums in halo_numbers_progen_split]
    halo_number_counts = [
        np.unique(hnums, return_counts=True) for hnums in
        halo_numbers_progen_split_noneg]
    haloids_new = []
    for hnums_u, counts in halo_number_counts:
        if len(hnums_u) == 0:
            haloids_new.append(-1)
        else:
            haloids_new.append(hnums_u[np.argmax(counts)])

    return haloids_new
