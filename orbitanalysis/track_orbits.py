import numpy as np
import time
import gc
import h5py
from pathos.multiprocessing import ProcessingPool as Pool

from orbitanalysis.collate import collate_orbit_history
from orbitanalysis.utils import myin1d, recenter_coordinates


def track_orbits(load_halo_particle_ids, load_snapshot_object, regions,
                 snapshot, catalogue, haloids, n_radii, savefile,
                 mode='pericentric', initial_snapshot_number=0, tree=None,
                 collate=True, save_properties=True, npool=None, verbose=True):
    """
    """

    tstart = time.time()

    # intermediate file
    ext = savefile.split('.')[-1]
    if ext == 'h5' or ext == 'hdf5':
        savefile_ = savefile[:-(len(ext) + 1)] + '_onthefly.' + ext
    else:
        savefile_ = savefile + '_onthefly'

    coords, vels, bulk_vels, radial_vels_next, ids_central = region_frame(
        snapshot, verbose)
    ids_next = snapshot.ids
    region_offsets_next = snapshot.region_offsets

    initialize_savefile(savefile_, snapshot.ids, coords, vels, snapshot.masses,
                        snapshot.region_offsets, snapshot.redshift, haloids,
                        snapshot.region_positions,
                        snapshot.region_radii / n_radii,
                        bulk_vels,
                        snapshot.snapshot_number,
                        snapshot.snapshot_number-initial_snapshot_number+1,
                        snapshot.snapshot_path, snapshot.particle_type,
                        n_radii, save_properties, verbose)

    snapnums_reversed = np.flip(
        np.arange(initial_snapshot_number, snapshot.snapshot_number))
    for ii, snapnum in enumerate(snapnums_reversed):

        if verbose:
            print('-' * 30, '\n')

        # get halo IDs for main progenitors at current snapshot
        haloids_new_, catalogue, snapnum = check_for_and_find_main_progenitors(
            load_halo_particle_ids, snapshot, catalogue, snapnum, ids_central,
            tree is None, npool, verbose)
        if tree is not None:
            haloids_new_ = tree[ii+1]
        if haloids_new_ is None:
            break
        has_progen = np.argwhere(haloids_new_ > -1).flatten()
        haloids_new = haloids_new_[has_progen]

        # discard ids and radial vels of halos in next snapshot (snapnum + 1)
        # that have no progenitor
        slices = np.array(list(zip(
            region_offsets_next[:-1], region_offsets_next[1:])))
        ids_next = np.concatenate([
            ids_next[start:end] for start, end in slices[has_progen]])
        radial_vels_next = np.concatenate([
            radial_vels_next[start:end] for start, end in slices[has_progen]])
        lens = [0] + [end-start for start, end in slices[has_progen]]
        region_offsets_next = np.cumsum(lens)

        # get regions at current snapshot
        region_positions, region_radii = regions(catalogue, haloids_new)
        snapshot = load_snapshot_object(
            snapshot, snapnum, region_positions, n_radii * region_radii,
            verbose)

        coords, vels, bulk_vels, radial_vels, ids_central = region_frame(
            snapshot, verbose)

        radii = -np.ones(len(haloids))
        radii[has_progen] = snapshot.region_radii / n_radii
        positions = -np.ones((len(haloids), 3))
        positions[has_progen, :] = snapshot.region_positions
        region_vels = -np.ones((len(haloids), 3))
        region_vels[has_progen, :] = bulk_vels

        # compare radial velocities between current and next snapshot
        orbiting_ids_at_snapshot_, orbiting_offsets = \
            compare_radial_velocities(
                mode, snapshot.ids, ids_next, radial_vels, radial_vels_next,
                snapshot.region_offsets, region_offsets_next, verbose)

        save_to_file(savefile_, orbiting_ids_at_snapshot_, orbiting_offsets,
                     snapshot.ids, coords, vels, snapshot.masses,
                     snapshot.region_offsets, snapshot.redshift, positions,
                     radii, region_vels, snapnum, haloids_new_, ii+1,
                     save_properties, verbose)

        ids_next, radial_vels_next, region_offsets_next = \
            snapshot.ids, radial_vels, snapshot.region_offsets

    if collate:
        collate_orbit_history(savefile, save_properties, verbose)

    if verbose:
        print('Finished orbiting decomposition (took {} s)\n'.format(
            time.time() - tstart))


def region_frame(snapshot, verbose):

    if verbose:
        print('Transforming to region frames...')
        t0 = time.time()

    slices = list(
        zip(snapshot.region_offsets[:-1], snapshot.region_offsets[1:]))
    region_coords = np.concatenate([
        recenter_coordinates(
            snapshot.coordinates[start:end]-p, snapshot.box_size)
        for (start, end), p in zip(slices, snapshot.region_positions)], axis=0)
    del snapshot.coordinates
    gc.collect()
    region_vels, region_bulk_vels = [], []
    if isinstance(snapshot.masses, np.ndarray):
        for start, end in slices:
            bulk_vel = np.sum(
                snapshot.masses[start:end][:, np.newaxis] *
                snapshot.velocities[start:end], axis=0) / \
                       np.sum(snapshot.masses[start:end])
            region_vels.append(snapshot.velocities[start:end] - bulk_vel)
            region_bulk_vels.append(bulk_vel)
    else:
        for start, end in slices:
            bulk_vel = np.mean(snapshot.velocities[start:end], axis=0)
            region_vels.append(snapshot.velocities[start:end] - bulk_vel)
            region_bulk_vels.append(bulk_vel)
    region_vels = np.concatenate(region_vels, axis=0)
    del snapshot.velocities
    gc.collect()
    rads = np.sqrt(np.einsum('...i,...i', region_coords, region_coords))
    radial_vels = np.einsum('...i,...i', region_vels, region_coords) / rads
    ids_tracked = [snapshot.ids[np.argsort(rads[start:end])[:100]+start]
                   for start, end in slices]

    if verbose:
        print('Transformed to region frames (took {} s)\n'.format(
            time.time() - t0))

    return region_coords, region_vels, np.array(region_bulk_vels), \
        radial_vels, ids_tracked


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

    lens = [0] + [len(orbids) for orbids in orbiting_ids]
    return np.concatenate(orbiting_ids), np.cumsum(lens)


def initialize_savefile(savefile, ids, coords, vels, masses, offsets, redshift,
                        hids, pos, radii, bulk_vel, snapshot_number, nsnaps,
                        snapdir, particle_type, n_radii, save_properties,
                        verbose):

    if verbose:
        print('Initializing savefile...')
        t0 = time.time()

    with h5py.File(savefile, 'w') as hf:

        gsnap_new = hf.create_group(
            'Snapshot{}'.format('%0.3d' % snapshot_number))
        gsnap_new.create_dataset('IDs', data=ids)
        gsnap_new.create_dataset('Coordinates', data=coords)
        if save_properties:
            gsnap_new.create_dataset('Velocities', data=vels)
            if isinstance(masses, np.ndarray):
                gsnap_new.create_dataset('Masses', data=masses)
        gsnap_new.create_dataset('Offsets', data=offsets)

        redshifts = np.empty(nsnaps)
        haloids = np.empty((nsnaps, len(hids)), dtype=np.int32)
        positions = np.empty((nsnaps, len(hids), 3))
        radiis = np.empty((nsnaps, len(hids)))
        bulk_velocities = np.empty((nsnaps, len(hids), 3))

        redshifts[0] = redshift
        haloids[0, :] = hids
        radiis[0, :] = radii
        positions[0, :, :] = pos
        bulk_velocities[0, :, :] = bulk_vel

        hf.create_dataset('Redshifts', data=redshifts)
        hf.create_dataset('HaloIDs', data=haloids)
        hf.create_dataset('Positions', data=positions)
        hf.create_dataset('Radii', data=radiis)
        hf.create_dataset('BulkVelocities', data=bulk_velocities)
        hf.create_dataset('ParticleMasses', data=np.zeros(nsnaps))
        if not isinstance(masses, np.ndarray):
            hf['ParticleMasses'][0] = masses

        head = hf.create_group('Options')
        opts = [('SnapDir', snapdir),
                ('PartType', particle_type),
                ('NumRadii', n_radii)]
        for attr, val in opts:
            head.attrs[attr] = val

    if verbose:
        print('Savefile initialized (took {} s)\n'.format(time.time() - t0))


def save_to_file(savefile, orbiting_ids_at_snapshot, orbiting_offsets, ids,
                 coords, vels, masses, offsets, redshift, pos, radii, bulk_vel,
                 snapshot_number, progens, index, save_properties, verbose):

    if verbose:
        print('Saving to file...')
        t0 = time.time()

    with h5py.File(savefile, 'r+') as hf:

        gsnap = hf['Snapshot{}'.format('%0.3d' % (snapshot_number + 1))]
        gsnap.create_dataset('OrbitingIDs', data=orbiting_ids_at_snapshot)
        gsnap.create_dataset('OrbitingOffsets', data=orbiting_offsets)

        gsnap_new = hf.create_group(
            'Snapshot{}'.format('%0.3d' % snapshot_number))
        gsnap_new.create_dataset('IDs', data=ids)
        gsnap_new.create_dataset('Coordinates', data=coords)
        if save_properties:
            gsnap_new.create_dataset('Velocities', data=vels)
            if isinstance(masses, np.ndarray):
                gsnap_new.create_dataset('Masses', data=masses)
        gsnap_new.create_dataset('Offsets', data=offsets)

        hf['Redshifts'][index] = redshift
        hf['HaloIDs'][index, :] = progens
        hf['Radii'][index, :] = radii
        hf['Positions'][index, :, :] = pos
        hf['BulkVelocities'][index, :, :] = bulk_vel
        if not isinstance(masses, np.ndarray):
            hf['ParticleMasses'][index] = masses

    if verbose:
        print('Saved to file (took {} s)\n'.format(time.time() - t0))
