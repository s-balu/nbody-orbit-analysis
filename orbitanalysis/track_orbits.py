import numpy as np
import time
import h5py

from orbitanalysis.utils import myin1d, recenter_coordinates


def track_orbits(snapshot_numbers, main_branches, regions, load_snapshot_data,
                 savefile, mode='pericentric', verbose=True):

    """
    Track the orbits of particles in gravitating systems.

    Parameters
    ----------
    snapshot_numbers : (n_snap,) array_like
        The snapshot numbers over which to track the orbits. Can be in
        increasing or decreasing order.
    main_branches : (n_snap, n_halo) array_like
        An array of dimension (number of snapshots) x (number of halos)
        containing the IDs of the main branch progenitors for the halos
        selected at the final snapshot. The array must be in the same order as
        `snapshot_numbers`. Wherever a progenitor does not exist (before the
        beginning of a branch) a value of -1 must be placed.
    regions : function
        A function that takes

        * a snapshot_number,
        * a list of halo IDs,

        and returns the coordinates of the centers of the halos, and the radii
        of the regions in which to track the orbits.
    load_snapshot_data : function
        A function that takes

        * a snapshot number,
        * the coordinates of the halo centers,
        * the radii of the regions encompassing each halo within which to track
          the orbits,

        and returns a dict with the following elements:

        * ids : (N,) ndarray - a list of the IDs of all particles in all
                regions, arranged in blocks.
        * coordinates : (N, 3) ndarray - the corresponding coordinates.
        * velocities : (N, 3) ndarray - the corresponding velocities.
        * masses : (N,) ndarray or float - the corresponding masses. Can be
                   supplied as an array arranged in blocks, or as a float if
                   all particles have the same mass.
        * region_offsets : (n_halos,) ndarray - the indices of the start of
                           each region block.
        * box_size : float or (3,) array_like - the simulation box side
                     length(s) when using a periodic box (optional).
    savefile : str
        The filename at which to save the result of the orbit tracking. The
        data is saved in HDF5 format.
    mode : {'pericentric', 'apocentric'}, optional
        The orbit detection mode. 'pericentric' counts the number of
        pericenters while 'apocentric' counts the number of apocenters.
    verbose : bool, optional
        Print status information and task durations during the orbit tracking.

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

    main_branches = np.asarray(main_branches)
    if main_branches.ndim == 1:
        main_branches = main_branches[:, np.newaxis]
    snapshot_numbers = np.asarray(snapshot_numbers)
    order = np.argsort(snapshot_numbers)
    snapshot_numbers = snapshot_numbers[order]
    main_branches = main_branches[order]

    initialize_savefile(
        savefile, snapshot_numbers[1:], main_branches[1:], verbose)

    for i, (halo_ids, snapshot_number) in enumerate(
            zip(main_branches, snapshot_numbers)):

        if verbose:
            print('-' * 30, '\n')
            print('Snapshot {}\n'.format('%03d' % snapshot_number))

        halo_exists = np.argwhere(halo_ids != -1).flatten()
        halo_ids_ = halo_ids[halo_exists]

        region_positions, region_radii = regions(
            snapshot_number, halo_ids_)

        snapshot = load_snapshot_data(
            snapshot_number, region_positions, region_radii)
        region_offsets = list(snapshot['region_offsets']) + [
            len(snapshot['ids'])]
        region_slices = np.array(
            list(zip(region_offsets[:-1], region_offsets[1:])))

        rhats, radial_vels, bulk_vels = region_frame(
            snapshot, region_slices, region_positions, verbose)

        if i > 0:

            progen_inds = myin1d(halo_exists, progen_exists)
            region_slices_desc = region_slices[progen_inds]

            orbiting_ids, orbiting_inds, orbiting_offsets, entered_ids, entered_offsets, \
                departed_ids, departed_offsets, matched_ids, matched_offsets, \
                angle_changes = compare_radial_velocities(
                    snapshot['ids'], ids_prev, radial_vels, radial_vels_prev,
                    rhats, rhats_prev, region_slices_desc,
                    region_slices_prev, mode, verbose)

            angles, matched_slices, orbiting_angle_ids, orbiting_angles, \
                orbiting_angle_slices, orbiting_angle_changes = calc_angles(
                    matched_ids, angle_changes, matched_offsets, progen_exists,
                    ids_angle_prev, orbiting_ids, orbiting_inds, orbiting_offsets,
                    progen_exists_prev, angles, angle_slices_prev,
                    orbiting_angle_ids, orbiting_angles, orbiting_angle_slices,
                    verbose)


            region_positions_ = -np.ones((len(halo_ids), 3))
            region_positions_[halo_exists] = region_positions
            region_radii_ = -np.ones(len(halo_ids))
            region_radii_[halo_exists] = region_radii
            bulk_velocities_ = -np.ones((len(halo_ids), 3))
            bulk_velocities_[halo_exists] = bulk_vels
            if 'box_size' in snapshot:
                box_size = snapshot['box_size']
            else:
                box_size = None

            save_to_file(
                savefile, orbiting_ids, orbiting_offsets, entered_ids,
                entered_offsets, departed_ids, departed_offsets,
                orbiting_angle_changes, region_positions_, region_radii_,
                bulk_velocities_, main_branches[-1][progen_exists],
                snapshot_number, box_size, i-1, verbose)

            progen_exists_prev = progen_exists
            ids_angle_prev = matched_ids
            angle_slices_prev = matched_slices

        else:

            progen_exists_prev = np.array([])
            ids_angle_prev = np.array([])
            angles = np.array([])
            angle_slices_prev = np.array([])
            orbiting_angle_ids = np.array([])
            orbiting_angles = np.array([])
            orbiting_angle_slices = np.array([])

        ids_prev = snapshot['ids']
        rhats_prev = rhats
        radial_vels_prev = radial_vels
        region_slices_prev = region_slices
        progen_exists = halo_exists

    if verbose:
        print('Finished orbiting decomposition (took {} s)\n'.format(
            time.time() - tstart))


def region_frame(snapshot, region_slices, region_positions, verbose):

    """
    Transform coordinates and velocities to region frames and compute radial
    velocities.
    """

    if verbose:
        print('Transforming to region frames...')
        t0 = time.time()

    region_coords = np.empty(np.shape(snapshot['coordinates']))
    if 'box_size' in snapshot:
        for sl, pos in zip(region_slices, region_positions):
            region_coords[slice(*sl), :] = recenter_coordinates(
                snapshot['coordinates'][slice(*sl)]-pos, snapshot['box_size'])
    else:
        for sl, pos in zip(region_slices, region_positions):
            region_coords[slice(*sl), :] = snapshot['coordinates'][
                slice(*sl)] - pos

    region_vels = np.empty(np.shape(snapshot['velocities']))
    region_bulk_vels = []
    if isinstance(snapshot['masses'], np.ndarray):
        for sl in region_slices:
            bulk_vel = np.sum(
                snapshot['masses'][slice(*sl)][:, np.newaxis] *
                snapshot['velocities'][slice(*sl)], axis=0) / \
                       np.sum(snapshot['masses'][slice(*sl)])
            region_vels[slice(*sl), :] = snapshot['velocities'][slice(*sl)] - \
                bulk_vel
            region_bulk_vels.append(bulk_vel)
    else:
        for sl in region_slices:
            bulk_vel = np.mean(snapshot['velocities'][slice(*sl)], axis=0)
            region_vels[slice(*sl), :] = snapshot['velocities'][slice(*sl)] - \
                bulk_vel
            region_bulk_vels.append(bulk_vel)

    rads = np.sqrt(np.einsum('...i,...i', region_coords, region_coords))
    rhats = region_coords / rads[:, np.newaxis]
    radial_vels = np.einsum('...i,...i', region_vels, rhats)

    if verbose:
        print('Transformed to region frames (took {} s)\n'.format(
            time.time() - t0))

    return rhats, radial_vels, np.array(region_bulk_vels)


def compare_radial_velocities(ids, ids_prev, radial_vels, radial_vels_prev,
                              rhat, rhat_prev, region_slices,
                              region_slices_prev, mode, verbose):

    """
    Identify sign flips in the radial velocity between snapshots.
    """

    if verbose:
        print('Identifying {}ers...'.format(mode[:8]))
        t0 = time.time()

    orbiting_ids = []
    orbiting_inds = []
    entered_ids = []
    departed_ids = []
    matched_ids = []
    angles = []
    for sl_prev, sl in zip(region_slices_prev, region_slices):
        departed = np.setdiff1d(ids_prev[slice(*sl_prev)], ids[slice(*sl)])
        inds_departed = np.where(
            np.in1d(ids_prev[slice(*sl_prev)], departed, kind='table'))[0]
        ids_prev_ = np.delete(ids_prev[slice(*sl_prev)], inds_departed)
        radial_vels_prev_ = np.delete(
            radial_vels_prev[slice(*sl_prev)], inds_departed)
        rhat_prev_ = np.delete(
            rhat_prev[slice(*sl_prev)], inds_departed, axis=0)

        inds_match = myin1d(ids[slice(*sl)], ids_prev_, kind='table')
        ids_match = ids[slice(*sl)][inds_match]
        radial_vels_match = radial_vels[slice(*sl)][inds_match]
        rhat_match = rhat[slice(*sl)][inds_match]

        if mode == 'pericentric':
            cond = (radial_vels_prev_ < 0) & (radial_vels_match > 0)
        elif mode == 'apocentric':
            cond = (radial_vels_prev_ > 0) & (radial_vels_match < 0)
        orbinds = np.argwhere(cond).flatten()
        orbiting_inds.append(orbinds)
        orbids = ids_prev_[orbinds]
        orbiting_ids.append(orbids)

        entids = np.setdiff1d(ids[slice(*sl)], ids_prev[slice(*sl_prev)])
        entered_ids.append(entids)
        departed_ids.append(departed)
        matched_ids.append(ids_match)

        angles.append(
            np.arccos(np.einsum('...i,...i', rhat_prev_, rhat_match)))

    if verbose:
        print('Finished identifying {}ers (took {} s)\n'.format(
            mode[:8], time.time() - t0))

    orbiting_lens = [0] + [len(orbids) for orbids in orbiting_ids]
    entered_lens = [0] + [len(entered) for entered in entered_ids]
    departed_lens = [0] + [len(departed) for departed in departed_ids]
    matched_lens = [0] + [len(matched) for matched in matched_ids]

    return np.concatenate(orbiting_ids), np.concatenate(orbiting_inds),\
        np.cumsum(orbiting_lens), \
        np.concatenate(entered_ids), np.cumsum(entered_lens), \
        np.concatenate(departed_ids), np.cumsum(departed_lens), \
        np.concatenate(matched_ids), np.cumsum(matched_lens), \
        np.concatenate(angles)


def calc_angles(matched_ids, angle_changes, matched_offsets, progen_exists,
                ids_angle_prev, orbiting_ids, orbiting_inds, orbiting_offsets,
                progen_exists_prev, angles_prev, angle_slices_prev,
                orbiting_angle_ids_prev, orbiting_angles_prev,
                orbiting_angle_slices_prev, verbose):

    """
    Return the angles by which particles identified as going through pericenter
    have advanced since their last detected pericenter, or since entering the
    region.

    Particles orbiting in subhalos will generally only advance a small
    angle between pericenters, allowing these spurious detections to be removed
    by making a cut in the angle.
    """

    if verbose:
        print('Calculating angles...')
        t0 = time.time()

    matched_slices = np.array(
        list(zip(matched_offsets[:-1], matched_offsets[1:])))
    orbiting_slices = np.array(
        list(zip(orbiting_offsets[:-1], orbiting_offsets[1:])))
    angles = np.empty(len(matched_ids))
    orbiting_angle_ids = []
    orbiting_angles = []
    orbiting_angle_changes = np.empty(len(orbiting_ids))
    j = 0
    for p, asl, osl in zip(
            progen_exists, matched_slices, orbiting_slices):

        ids_angle_halo = matched_ids[slice(*asl)]
        angles_halo = angle_changes[slice(*asl)]

        if p in progen_exists_prev:
            aslp = angle_slices_prev[j]
            intersect_inds = np.where(np.in1d(
                ids_angle_halo, ids_angle_prev[slice(*aslp)], kind='table'))[0]

            inds_angle = myin1d(
                ids_angle_prev[slice(*aslp)],
                ids_angle_halo[intersect_inds], kind='table')
            angles_halo[intersect_inds] += angles_prev[
                slice(*aslp)][inds_angle]

        angles[slice(*asl)] = angles_halo

        orb_angles_halo = angles_halo[orbiting_inds[slice(*osl)]]
        orb_angle_changes_halo = np.copy(orb_angles_halo)

        if p in progen_exists_prev:

            oaslp = orbiting_angle_slices_prev[j]

            orbiting_angle_ids_prev_halo = orbiting_angle_ids_prev[
                slice(*oaslp)]
            orbiting_angles_prev_halo = orbiting_angles_prev[
                slice(*oaslp)]

            intersect_bools = np.in1d(
                orbiting_ids[slice(*osl)],
                orbiting_angle_ids_prev_halo, kind='table')
            intersect_inds = np.where(intersect_bools)[0]
            nointersect_inds = np.where(~intersect_bools)[0]

            inds_orb_angle = myin1d(
                orbiting_angle_ids_prev_halo,
                orbiting_ids[slice(*osl)][intersect_inds], kind='table')

            orbiting_angle_ids_halo = np.copy(
                orbiting_angle_ids_prev_halo)
            orbiting_angles_halo = np.copy(orbiting_angles_prev_halo)
            orbiting_angles_halo[inds_orb_angle] = orb_angles_halo[
                intersect_inds]
            orb_angle_changes_halo[intersect_inds] -= \
                orbiting_angles_prev_halo[inds_orb_angle]

            orbiting_angle_ids_new_halo = orbiting_ids[slice(*osl)][
                nointersect_inds]
            orbiting_angles_new_halo = orb_angles_halo[
                nointersect_inds]
            orbiting_angle_ids_halo = np.append(
                orbiting_angle_ids_halo, orbiting_angle_ids_new_halo)
            orbiting_angles_halo = np.append(
                orbiting_angles_halo, orbiting_angles_new_halo)

            j += 1

        else:

            orbiting_angle_ids_halo = orbiting_ids[slice(*osl)]
            orbiting_angles_halo = np.copy(orb_angles_halo)

        orbiting_angle_ids.append(orbiting_angle_ids_halo)
        orbiting_angles.append(orbiting_angles_halo)
        orbiting_angle_changes[slice(*osl)] = orb_angle_changes_halo

    orbiting_angle_lens = [0] + [
        len(angids) for angids in orbiting_angle_ids]
    orbiting_angle_ids = np.concatenate(orbiting_angle_ids)
    orbiting_angles = np.concatenate(orbiting_angles)
    orbiting_angle_offsets = np.cumsum(orbiting_angle_lens)
    orbiting_angle_slices = np.array(
        list(zip(orbiting_angle_offsets[:-1],
                 orbiting_angle_offsets[1:])))

    if verbose:
        print('Finished calculating angles (took {} s)\n'.format(
            time.time()-t0))

    return angles, matched_slices, orbiting_angle_ids, orbiting_angles, \
        orbiting_angle_slices, orbiting_angle_changes


def initialize_savefile(savefile, snapshot_numbers, main_branches, verbose):

    with h5py.File(savefile, 'w') as hf:

        hf.create_dataset('snapshot_numbers', data=snapshot_numbers)
        hf.create_dataset('main_branches', data=main_branches)
        hf.create_dataset(
            'region_radii', data=np.empty(np.shape(main_branches)))
        hf.create_dataset(
            'region_positions', data=np.empty(np.shape(main_branches) + (3,)))
        hf.create_dataset(
            'bulk_velocities', data=np.empty(np.shape(main_branches) + (3,)))

    if verbose:
        print('Savefile initialized\n')


def save_to_file(savefile, orbiting_ids, orbiting_offsets, infalling_ids,
                 infalling_offsets, departed_ids, departed_offsets,
                 angles, region_positions, region_radii, bulk_velocities,
                 halo_ids_final, snapshot_number, box_size, index, verbose):

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

        hf['region_radii'][index, :] = region_radii
        hf['region_positions'][index, :, :] = region_positions
        hf['bulk_velocities'][index, :, :] = bulk_velocities

        if index == 0 and box_size is not None:
            hf.attrs['box_size'] = box_size

    if verbose:
        print('Saved to file (took {} s)\n'.format(time.time() - t0))
