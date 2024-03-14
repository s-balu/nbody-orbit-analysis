import numpy as np
from orbitanalysis.utils import recenter_coordinates, myin1d


def get_central_particle_ids(snapshot, halo_positions, n=100):

    """
    Find the IDs of the n closest particles to the halo centers.

    Parameters
    ----------
    snapshot : object
        An object with the following attributes:

        * ids : (N,) ndarray - a list of the IDs of all particles in all
                central regions, arranged in blocks.
        * coordinates : (N, 3) ndarray - the corresponding coordinates.
        * region_offsets : (n_halos,) ndarray - the indices of the start of
                           each region block.
        * box_size : float or (3,) array_like - the simulation box side
                     length(s).
    halo_positions : (n_halos,) array_like
        The coordinates of the ceneters of the halos/central regions.
    n : int, optional
        The number of central particles.

    Returns
    -------

    central_ids : (N,) ndarray
        The central IDs arranged in blocks. Will contain n * n_halos elements
        if all central regions encompass >= n particles.
    offsets : (n_halos,) ndarray
        The indices corresponding to the start of each block in central_ids.

    """

    slices = list(
        zip(snapshot.region_offsets[:-1], snapshot.region_offsets[1:]))
    region_coords = np.concatenate([
        recenter_coordinates(
            snapshot.coordinates[start:end]-p, snapshot.box_size)
        for (start, end), p in zip(slices, halo_positions)], axis=0)

    rads = np.sqrt(np.einsum('...i,...i', region_coords, region_coords))
    central_ids = [snapshot.ids[np.argsort(rads[start:end])[:n]+start]
                   for start, end in slices]
    offsets = np.cumsum([0] + [len(ids) for ids in central_ids])[:-1]
    return np.hstack(central_ids), offsets


def find_main_progenitors(halo_pids, halo_offsets, tracked_pids,
                          tracked_offsets):

    """
    Find the main progenitors of a set of halos by tracking their central
    particles.

    Parameters
    ----------
    halo_pids : (N,) ndarray
        Particles in halos, arranged in blocks.
    halo_offsets : (n_halos,) ndarray
        The indices corresponding to the start of each block in halo_pids.
    tracked_pids : (M,) ndarray
        The central IDs of the descendant halos that are being tracked.
    tracked_offsets : (n_descendants,) ndarray
        The indices corresponding to the start of each block in tracked_pids.

    Returns
    -------

    """

    tracked_pids_, unique_inds = np.unique(tracked_pids, return_index=True)
    tracked_pids = -np.ones(len(tracked_pids), dtype=int)
    tracked_pids[unique_inds] = tracked_pids_

    halo_diffs = np.diff(halo_offsets)
    halo_lens = np.append(halo_diffs, len(halo_pids)-halo_offsets[-1])
    tracked_diffs = np.diff(tracked_offsets)
    tracked_lens = np.append(
        tracked_diffs, len(tracked_pids)-tracked_offsets[-1])

    halo_number = np.hstack([
        n * np.ones(hlen, dtype=int) for n, hlen in enumerate(halo_lens)])

    intersect_inds = np.where(
        np.in1d(tracked_pids, halo_pids, kind='table'))[0]
    tracked_pids_present = tracked_pids[intersect_inds]

    inds = myin1d(halo_pids, tracked_pids_present, kind='table')
    halo_numbers_progen_ = halo_number[inds]
    halo_numbers_progen = -np.ones(len(tracked_pids), dtype=int)
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
