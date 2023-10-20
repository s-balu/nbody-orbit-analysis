import numpy as np
import time
import gc
import h5py
from pathos.multiprocessing import ProcessingPool as Pool

from orbitanalysis.utils import myin1d


def track_orbits(load_halo_particle_ids, load_snapshot_object, snapshot,
                 catalogue, haloids, n_radii, mode='pericentric',
                 initial_snapshot_number=0, tree=None, savefile=None,
                 npool=None, verbose=True):
    """
    """

    tstart = time.time()

    coords, vels, radial_vels_next, ids_central = region_frame(
        snapshot, verbose)
    ids_next = snapshot.ids
    region_offsets_next = snapshot.region_offsets

    initialize_savefile(savefile, snapshot.ids, coords, vels, snapshot.masses,
                        snapshot.region_offsets, snapshot.redshift, haloids,
                        snapshot.cutout_positions,
                        snapshot.cutout_radii / n_radii,
                        snapshot.snapshot_number,
                        snapshot.snapshot_number-initial_snapshot_number+1,
                        snapshot.snapshot_path, snapshot.particle_type,
                        n_radii, verbose)

    for ii, snapnum in enumerate(np.flip(np.arange(
            initial_snapshot_number, snapshot.snapshot_number))):

        if verbose:
            print('-' * 30, '\n')

        haloids_new_, catalogue, snapnum = check_for_and_find_main_progenitors(
            load_halo_particle_ids, snapshot, catalogue, snapnum, ids_central,
            tree is None, npool, verbose)
        if tree is not None:
            haloids_new_ = tree[ii+1]
        if haloids_new_ is None:
            break
        has_progen = np.argwhere(haloids_new_ > -1).flatten()
        haloids_new = haloids_new_[has_progen]

        slices = np.array(list(zip(
            region_offsets_next[:-1], region_offsets_next[1:])))
        ids_next = np.concatenate([
            ids_next[start:end] for start, end in slices[has_progen]])
        radial_vels_next = np.concatenate([
            radial_vels_next[start:end] for start, end in slices[has_progen]])
        lens = [end-start for start, end in slices[has_progen]]
        region_offsets_next = np.concatenate([[0], np.cumsum(lens)])

        snapshot = load_snapshot_object(
            snapshot, catalogue, snapnum, haloids_new, n_radii, verbose)

        coords, vels, radial_vels, ids_central = region_frame(
            snapshot, verbose)

        radii = -np.ones(len(haloids))
        radii[has_progen] = snapshot.cutout_radii / n_radii
        positions = -np.ones((len(haloids), 3))
        positions[has_progen, :] = snapshot.cutout_positions

        orbiting_ids_at_snapshot_, orbiting_offsets = \
            compare_radial_velocities(
                mode, snapshot.ids, ids_next, radial_vels, radial_vels_next,
                snapshot.region_offsets, region_offsets_next, verbose)

        save_to_file(savefile, orbiting_ids_at_snapshot_, orbiting_offsets,
                     snapshot.ids, coords, vels, snapshot.masses,
                     snapshot.region_offsets, snapshot.redshift, positions,
                     radii, snapnum, haloids_new_, ii+1, verbose)

        ids_next, radial_vels_next, region_offsets_next = \
            snapshot.ids, radial_vels, snapshot.region_offsets

    if verbose:
        print('Finished orbiting decomposition (took {} s)\n'.format(
            time.time() - tstart))


def region_frame(snapshot, verbose):

    if verbose:
        print('Transforming to region frames...')
        t0 = time.time()

    cslices = list(zip(snapshot.region_offsets[:-1],
                       snapshot.region_offsets[1:], snapshot.cutout_positions))
    vslices = list(zip(snapshot.region_offsets[:-1],
                       snapshot.region_offsets[1:]))
    region_coords = np.concatenate([
        snapshot.coords[start:end]-p for start, end, p in cslices], axis=0)
    del snapshot.coords
    gc.collect()
    region_vels = np.concatenate([
        snapshot.vels[start:end] - np.mean(snapshot.vels[start:end], axis=0)
        for start, end in vslices], axis=0)
    del snapshot.vels
    gc.collect()
    rads = np.sqrt(np.einsum('...i,...i', region_coords, region_coords))
    radial_vels = np.einsum('...i,...i', region_vels, region_coords) / rads
    ids_tracked = [snapshot.ids[np.argsort(rads[start:end])[:100]+start]
                   for start, end in vslices]

    if verbose:
        print('Transformed to region frames (took {} s)\n'.format(
            time.time() - t0))

    return region_coords, region_vels, radial_vels, ids_tracked


def find_main_progenitors(halo_pids, tracked_ids, npool):

    lengths = np.array([len(pids) for pids in halo_pids])
    halonums = np.arange(len(halo_pids))
    halo_pids_flat = np.hstack(halo_pids)

    tracked_ids_flat = np.hstack(tracked_ids)
    present_tot = np.in1d(halo_pids_flat, tracked_ids_flat, kind='table')
    pcounts_tot = np.split(present_tot, np.cumsum(lengths))[:-1]
    npresent_tot = np.array([np.count_nonzero(pc) for pc in pcounts_tot])
    unoccupied = np.argwhere(npresent_tot == 0).flatten()

    for i in sorted(unoccupied, reverse=True):
        del halo_pids[i]
    if len(halo_pids) == 0:
        return None
    halo_pids_flat = np.hstack(halo_pids)
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
    if npool is None:
        haloids_new = []
        for jj in range(len(tracked_ids)):
            haloids_new.append(get_progenitor(jj))
    else:
        haloids_new = Pool(npool).map(
            get_progenitor, np.arange(len(tracked_ids)))
    haloids_new = np.array(haloids_new)

    return haloids_new


def check_for_and_find_main_progenitors(func, snapshot, catalogue, snapnum,
                                        ids_central, return_main_progenitors,
                                        npool, verbose):
    if verbose:
        print('Finding main progenitors at snapshot {}...'.format(snapnum))
        t0 = time.time()

    halo_pids, catalogue_ = func(snapshot, catalogue, snapnum)
    if return_main_progenitors:
        if halo_pids is not None:
            main_progenitors = find_main_progenitors(
                halo_pids, ids_central, npool)
            if verbose:
                print('Found main progenitors (took {} s)\n'.format(
                    time.time() - t0))
            return main_progenitors, catalogue_, snapnum
        else:
            return None, None, snapnum + 1
    else:
        return None, catalogue_, snapnum


def compare_radial_velocities(mode, ids, ids_next, radial_vels,
                              radial_vels_next, offsets, offsets_next,
                              verbose):

    if verbose:
        print('Starting comparison...')
        t0 = time.time()

    slices = list(zip(offsets[:-1], offsets[1:]))
    slices_next = list(zip(offsets_next[:-1], offsets_next[1:]))

    orbiting_ids = []
    for sl, sl_next in zip(slices, slices_next):
        departed = np.setdiff1d(ids[slice(*sl)], ids_next[slice(*sl_next)])
        inds_departed = np.where(np.in1d(ids[slice(*sl)], departed))[0]
        radial_vels_ = np.delete(radial_vels[slice(*sl)], inds_departed)
        ids_ = np.delete(ids[slice(*sl)], inds_departed)

        inds_match = myin1d(ids_next[slice(*sl_next)], ids_)
        radial_vels_match = radial_vels_next[slice(*sl_next)][inds_match]

        if mode == 'pericentric':
            cond = (radial_vels_ < 0) & (radial_vels_match > 0)
        elif mode == 'apocentric':
            cond = (radial_vels_ > 0) & (radial_vels_match < 0)
        else:
            raise ValueError("Orbit detection mode not recognized. Please "
                             "select either 'pericentric' or 'apocentric'.")
        orbinds = np.argwhere(cond).flatten()
        orbiting_ids.append(ids_[orbinds])

    if verbose:
        print('Finished comparison (took {} s)\n'.format(time.time() - t0))

    lens = [len(orbids) for orbids in orbiting_ids]
    offsets = np.concatenate([[0], np.cumsum(lens)])
    return np.concatenate(orbiting_ids), offsets


def initialize_savefile(savefile, ids, coords, vels, masses, offsets, redshift,
                        hids, pos, radii, snapshot_number, nsnaps, snapdir,
                        particle_type, n_radii, verbose):

    if verbose:
        print('Initializing savefile...')
        t0 = time.time()

    hf = h5py.File(savefile, 'w')

    gsnap_new = hf.create_group('Snapshot{}'.format('%0.3d' % snapshot_number))
    gsnap_new.create_dataset('IDs', data=ids)
    gsnap_new.create_dataset('Coordinates', data=coords)
    gsnap_new.create_dataset('Velocities', data=vels)
    if isinstance(masses, np.ndarray):
        gsnap_new.create_dataset('Masses', data=masses)
    gsnap_new.create_dataset('Offsets', data=offsets)

    redshifts = np.empty(nsnaps)
    haloids = np.empty((nsnaps, len(hids)), dtype=np.int32)
    positions = np.empty((nsnaps, len(hids), 3))
    radiis = np.empty((nsnaps, len(hids)))

    redshifts[0] = redshift
    haloids[0, :] = hids
    radiis[0, :] = radii
    positions[0, :, :] = pos

    hf.create_dataset('Redshifts', data=redshifts)
    hf.create_dataset('HaloIDs', data=haloids)
    hf.create_dataset('Positions', data=positions)
    hf.create_dataset('Radii', data=radiis)
    hf.create_dataset('ParticleMasses', data=np.zeros(nsnaps))
    if not isinstance(masses, np.ndarray):
        hf['ParticleMasses'][0] = masses

    head = hf.create_group('Options')
    attrs = [('SnapDir', snapdir),
             ('PartType', particle_type),
             ('NumRadii', n_radii)]
    for attr in list(attrs):
        head.attrs[attr[0]] = attr[1]

    hf.close()

    if verbose:
        print('Savefile initialized (took {} s)\n'.format(time.time() - t0))


def save_to_file(savefile, orbiting_ids_at_snapshot, orbiting_offsets, ids,
                 coords, vels, masses, offsets, redshift, pos, radii,
                 snapshot_number, progens, index, verbose):

    if verbose:
        print('Saving to file...')
        t0 = time.time()

    hf = h5py.File(savefile, 'r+')

    gsnap = hf['Snapshot{}'.format('%0.3d' % (snapshot_number + 1))]
    gsnap.create_dataset('OrbitingIDs', data=orbiting_ids_at_snapshot)
    gsnap.create_dataset('OrbitingOffsets', data=orbiting_offsets)

    gsnap_new = hf.create_group('Snapshot{}'.format('%0.3d' % snapshot_number))
    gsnap_new.create_dataset('IDs', data=ids)
    gsnap_new.create_dataset('Coordinates', data=coords)
    gsnap_new.create_dataset('Velocities', data=vels)
    if isinstance(masses, np.ndarray):
        gsnap_new.create_dataset('Masses', data=masses)
    gsnap_new.create_dataset('Offsets', data=offsets)

    hf['Redshifts'][index] = redshift
    hf['HaloIDs'][index, :] = progens
    hf['Radii'][index, :] = radii
    hf['Positions'][index, :, :] = pos
    if not isinstance(masses, np.ndarray):
        hf['ParticleMasses'][index] = masses

    hf.close()

    if verbose:
        print('Saved to file (took {} s)\n'.format(time.time() - t0))
