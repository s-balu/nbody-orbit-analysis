import numpy as np
import time
import h5py

from orbitanalysis.utils import myin1d, recenter_coordinates, vector_norm


def track_orbits(main_branches, snapshot_numbers, load_snapshot_object,
                 regions, n_radii, savefile, mode='pericentric', verbose=True):

    """
    """

    if len(main_branches) != len(snapshot_numbers):
        raise ValueError(
            "Number of halo main branch nodes does not equal the number of "
            "snapshot numbers supplied. Must have len(main_branches) == "
            "len(snapshot_numbers).")
    if (mode != 'pericentric') and (mode != 'apocentric'):
        raise ValueError(
            "Orbit detection mode not recognized. Please specify either "
            "'pericentric' or 'apocentric'.")

    tstart = time.time()

    initialize_savefile(
        savefile, snapshot_numbers[1:], main_branches, n_radii, verbose)

    for i, (halo_ids, snapshot_number) in enumerate(
            zip(main_branches, snapshot_numbers)):

        if verbose:
            print('-' * 30, '\n')

        halo_exists = np.argwhere(halo_ids != -1).flatten()
        halo_ids_ = halo_ids[halo_exists]

        region_positions, region_radii = regions(
            snapshot_number, halo_ids_, n_radii)

        snapshot = load_snapshot_object(
            snapshot_number, region_positions, region_radii)

        coords, radial_vels, bulk_vels = region_frame(snapshot, verbose)

        if i > 0:

            progen_exists = np.argwhere(halo_ids_prev != -1).flatten()
            progen_inds = myin1d(halo_exists, progen_exists)
            noprogen = np.setdiff1d(halo_exists, progen_exists)
            noprogen_inds = myin1d(halo_exists, noprogen)

            region_slices = np.array(list(zip(
                snapshot.region_offsets[:-1], snapshot.region_offsets[1:])))
            region_slices_desc = region_slices[progen_inds]
            region_slices_noprogen = region_slices[noprogen_inds]
            # coords_ = np.concatenate(
            #     [coords[slice(*sl)] for sl in region_slices[progen_inds]],
            #     axis=0)
            radial_vels_desc = np.concatenate(
                [radial_vels[slice(*sl)] for sl in region_slices_desc], axis=0)

            orbiting_ids, orbiting_inds, orbiting_offsets, entered_ids, \
                entered_inds, entered_offsets, departed_ids, \
                departed_offsets = compare_radial_velocities(
                    snapshot.ids, ids_prev, radial_vels_desc, radial_vels_prev,
                    snapshot.region_offsets, region_offsets_prev, mode,
                    verbose)

            angles = 2*np.pi*np.ones(len(orbiting_inds))

            halo_positions = -np.ones((len(halo_ids), 3))
            halo_positions[progen_exists] = region_positions
            halo_radii = -np.ones(len(halo_ids))
            halo_radii[progen_exists] = region_radii / n_radii
            halo_velocities = -np.ones((len(halo_ids), 3))
            halo_velocities[progen_exists] = bulk_vels

            save_to_file(
                savefile, orbiting_ids, orbiting_offsets, entered_ids,
                entered_offsets, departed_ids, departed_offsets, angles,
                snapshot.redshift, halo_positions, halo_radii, halo_velocities,
                main_branches[-1][progen_exists], snapshot_number, i-1,
                verbose)
        else:

            ids_angle = snapshot.ids
            coords_angle = coords
            angle_offsets = snapshot.region_offsets

        ids_prev, radial_vels_prev, region_offsets_prev, halo_ids_prev = \
            snapshot.ids, radial_vels, snapshot.region_offsets, halo_ids

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

    rads = np.sqrt(np.einsum('...i,...i', region_coords, region_coords))
    radial_vels = np.einsum('...i,...i', region_vels, region_coords) / rads

    if verbose:
        print('Transformed to region frames (took {} s)\n'.format(
            time.time() - t0))

    return region_coords, radial_vels, np.array(region_bulk_vels)


def compare_radial_velocities(ids, ids_prev, radial_vels, radial_vels_prev,
                              region_offsets, region_offsets_prev, mode,
                              verbose):

    if verbose:
        print('Identifying {}ers...'.format(mode[:8]))
        t0 = time.time()

    slices_prev = list(zip(region_offsets_prev[:-1], region_offsets_prev[1:]))
    slices = list(zip(region_offsets[:-1], region_offsets[1:]))

    orbiting_ids = []
    orbiting_inds = []
    entered_ids = []
    entered_inds = []
    departed_ids = []
    for sl_prev, sl in zip(slices_prev, slices):
        departed = np.setdiff1d(ids_prev[slice(*sl_prev)], ids[slice(*sl)])
        inds_departed = np.where(
            np.in1d(ids_prev[slice(*sl_prev)], departed))[0]
        radial_vels_prev_ = np.delete(
            radial_vels_prev[slice(*sl_prev)], inds_departed)
        ids_prev_ = np.delete(ids_prev[slice(*sl_prev)], inds_departed)

        inds_match = myin1d(ids[slice(*sl)], ids_prev_)
        radial_vels_match = radial_vels[slice(*sl)][inds_match]

        if mode == 'pericentric':
            cond = (radial_vels_prev_ < 0) & (radial_vels_match > 0)
        elif mode == 'apocentric':
            cond = (radial_vels_prev_ > 0) & (radial_vels_match < 0)
        orbinds = np.argwhere(cond).flatten()
        orbids = ids_prev_[orbinds]
        orbiting_ids.append(orbids)
        orbiting_inds.append(myin1d(ids[slice(*sl)], orbids))

        entids = np.setdiff1d(ids[slice(*sl)], ids_prev[slice(*sl_prev)])
        entered_ids.append(entids)
        entered_inds.append(myin1d(ids[slice(*sl)], entids))
        departed_ids.append(departed)

    if verbose:
        print('Finished identifying {}ers (took {} s)\n'.format(
            mode[:8], time.time() - t0))

    orbiting_lens = [0] + [len(orbids) for orbids in orbiting_ids]
    entered_lens = [0] + [len(entered) for entered in entered_ids]
    departed_lens = [0] + [len(departed) for departed in departed_ids]
    return np.concatenate(orbiting_ids), np.concatenate(orbiting_inds), \
        np.cumsum(orbiting_lens), np.concatenate(entered_ids), \
        np.concatenate(entered_inds), np.cumsum(entered_lens), \
        np.concatenate(departed_ids), np.cumsum(departed_lens)


def initialize_savefile(savefile, snapshot_numbers, main_branches, n_radii,
                        verbose):

    with h5py.File(savefile, 'w') as hf:

        hf.create_dataset('snapshot_numbers', data=snapshot_numbers)
        hf.create_dataset('redshifts', data=np.empty(len(snapshot_numbers)))
        hf.create_dataset('main_branches', data=main_branches)
        hf.create_dataset('halo_radii', data=np.empty(np.shape(main_branches)))
        hf.create_dataset(
            'halo_positions', data=np.empty(np.shape(main_branches) + (3,)))
        hf.create_dataset(
            'halo_velocities', data=np.empty(np.shape(main_branches) + (3,)))
        hf.attrs['multiple_of_radii'] = n_radii

    if verbose:
        print('Savefile initialized\n')


def save_to_file(savefile, orbiting_ids, orbiting_offsets, infalling_ids,
                 infalling_offsets, departed_ids, departed_offsets,
                 angles, redshift, halo_positions, halo_radii, halo_velocities,
                 halo_ids_final, snapshot_number, index, verbose):

    if verbose:
        print('Saving to file...')
        t0 = time.time()

    with h5py.File(savefile, 'r+') as hf:

        gsnap = hf.create_group(
            'snapshot_{}'.format('%0.3d' % snapshot_number))
        gsnap.create_dataset('orbiting_offsets', data=orbiting_offsets)
        gsnap.create_dataset('orbiting_IDs', data=orbiting_ids)
        gsnap.create_dataset('angles', data=angles)

        gsnap.create_dataset('infalling_offsets', data=infalling_offsets)
        gsnap.create_dataset('infalling_IDs', data=infalling_ids)

        gsnap.create_dataset('departed_offsets', data=departed_offsets)
        gsnap.create_dataset('departed_IDs', data=departed_ids)

        gsnap.create_dataset('halo_ids_final', data=halo_ids_final)

        hf['redshifts'][index] = redshift
        hf['halo_radii'][index, :] = halo_radii
        hf['halo_positions'][index, :, :] = halo_positions
        hf['halo_velocities'][index, :, :] = halo_velocities

    if verbose:
        print('Saved to file (took {} s)\n'.format(time.time() - t0))
