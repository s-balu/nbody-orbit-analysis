import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

from orbitanalysis.utils import recenter_coordinates


def get_tracked_ids(snapshot):

    slices = list(
        zip(snapshot.region_offsets[:-1], snapshot.region_offsets[1:]))
    region_coords = np.concatenate([
        recenter_coordinates(
            snapshot.coordinates[start:end]-p, snapshot.box_size)
        for (start, end), p in zip(slices, snapshot.region_positions)], axis=0)

    rads = np.sqrt(np.einsum('...i,...i', region_coords, region_coords))
    return [snapshot.ids[np.argsort(rads[start:end])[:100]+start]
            for start, end in slices]


def find_main_progenitors(halo_pids, tracked_ids, npool):

    lengths = np.array([len(pids) for pids in halo_pids])
    halonums = np.arange(len(halo_pids))
    halo_pids_flat = np.hstack(halo_pids)

    tracked_ids_flat = np.hstack(tracked_ids)
    present_all = np.in1d(halo_pids_flat, tracked_ids_flat, kind='table')
    pcounts_all = np.split(present_all, np.cumsum(lengths))[:-1]
    npresent_all = np.array([np.count_nonzero(pc) for pc in pcounts_all])
    occupied = np.argwhere(npresent_all != 0).flatten()
    unoccupied = np.argwhere(npresent_all == 0).flatten()

    halo_pids_occupied = [halo_pids[i] for i in occupied]
    if len(halo_pids) == 0:
        return None
    halo_pids_flat = np.hstack(halo_pids_occupied)
    lengths = np.delete(lengths, unoccupied)
    halonums = np.delete(halonums, unoccupied)

    def get_progenitor(ii):
        ids = tracked_ids[ii]
        present = np.in1d(halo_pids_flat, ids, kind='table')
        pcounts = np.split(present, np.cumsum(lengths))[:-1]
        npresent = [np.count_nonzero(pc) for pc in pcounts]

        if np.all(npresent == 0):
            return -1
        else:
            return halonums[np.argmax(npresent)]
    if (npool == 1) or (npool is None):
        haloids_new = []
        for jj in range(len(tracked_ids)):
            haloids_new.append(get_progenitor(jj))
    else:
        haloids_new = Pool(npool).map(
            get_progenitor, np.arange(len(tracked_ids)))
    haloids_new = np.array(haloids_new)

    return haloids_new
