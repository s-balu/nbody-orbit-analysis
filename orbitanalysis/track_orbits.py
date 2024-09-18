import numpy as np
import time
import h5py

from orbitanalysis.utils import myin1d, recenter_coordinates


def track_orbits(snapshot_numbers, main_branches, regions, load_snapshot_data,
                 savefile, mode='pericentric', checkpoint=False, resume=False,
                 verbose=True):

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

    if resume:
        with h5py.File(savefile, 'r') as hf:
            snapshot_number_resume = int(list(hf.keys())[-1].split('_')[1])

        sind = np.argwhere(
            snapshot_numbers == snapshot_number_resume).flatten()[0]
        snapshot_numbers = snapshot_numbers[sind:]
        main_branches = main_branches[sind:]

    istart, started = 0, False
    for i, (halo_ids, snapshot_number) in enumerate(
            zip(main_branches, snapshot_numbers)):

        if verbose:
            if resume:
                print('Resuming from file...\n')
            print('-' * 30, '\n')
            print('Snapshot {}\n'.format('%03d' % snapshot_number))

        halo_exists = np.argwhere(halo_ids != -1).flatten()
        if len(halo_exists) == 0:
            if started is False:
                istart = i + 1
            continue
        halo_ids_ = halo_ids[halo_exists]

        region_positions, region_radii = regions(
            snapshot_number, halo_ids_)

        snapshot = load_snapshot_data(
            snapshot_number, region_positions, region_radii)
        if len(snapshot['coordinates']) == 0:
            if started is False:
                istart = i + 1
            continue
        else:
            started = True
        region_offsets = list(snapshot['region_offsets']) + [
            len(snapshot['ids'])]
        region_slices = np.array(
            list(zip(region_offsets[:-1], region_offsets[1:])))

        rhats, radial_vels, bulk_vels = region_frame(
            snapshot, region_slices, region_positions, verbose)

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

        if i > istart:

            progen_inds = myin1d(halo_exists, progen_exists)
            region_slices_desc = region_slices[progen_inds]

            ids_dict = compare_radial_velocities(
                snapshot['ids'], ids_prev, radial_vels, radial_vels_prev,
                rhats, rhats_prev, region_slices_desc, region_slices_prev,
                mode, verbose)

            angles_dict = calc_angles(
                ids_dict, angles_dict, progen_exists, progen_exists_prev,
                ids_angle_prev, angle_slices_prev, verbose)

            save_to_file(
                savefile, ids_dict, angles_dict, region_positions_,
                region_radii_, bulk_velocities_, halo_ids,
                main_branches[-1][progen_exists], snapshot_number, checkpoint,
                verbose)

            progen_exists_prev = progen_exists
            ids_angle_prev = ids_dict['matched_ids']
            angle_slices_prev = angles_dict['matched_slices']

        else:

            if resume:

                ids_dict, angles_dict, ids_angle_prev, angle_slices_prev, \
                    progen_exists_prev = read_resume_data(
                        savefile, snapshot_number_resume, main_branches[-1])

            else:

                initialize_savefile(savefile, mode, box_size, verbose)

                ids_dict, angles_dict = init_dicts(
                    region_offsets, snapshot['ids'])

                save_to_file(
                    savefile, ids_dict, angles_dict, region_positions_,
                    region_radii_, bulk_velocities_, halo_ids,
                    main_branches[-1][halo_exists], snapshot_number,
                    checkpoint, verbose)

                progen_exists_prev = np.array([])
                ids_angle_prev = np.array([])
                angle_slices_prev = np.array([])

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
            np.in1d(ids_prev[slice(*sl_prev)], departed))[0]
        ids_prev_ = np.delete(ids_prev[slice(*sl_prev)], inds_departed)
        radial_vels_prev_ = np.delete(
            radial_vels_prev[slice(*sl_prev)], inds_departed)
        rhat_prev_ = np.delete(
            rhat_prev[slice(*sl_prev)], inds_departed, axis=0)

        inds_match = myin1d(ids[slice(*sl)], ids_prev_)
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

    out_dict = {}
    out_dict['orbiting_ids'] = np.concatenate(orbiting_ids)
    out_dict['orbiting_inds'] = np.concatenate(orbiting_inds)
    out_dict['orbiting_offsets'] = np.cumsum(orbiting_lens)
    out_dict['entered_ids'] = np.concatenate(entered_ids)
    out_dict['entered_offsets'] = np.cumsum(entered_lens)
    out_dict['departed_ids'] = np.concatenate(departed_ids)
    out_dict['departed_offsets'] = np.cumsum(departed_lens)
    out_dict['matched_ids'] = np.concatenate(matched_ids)
    out_dict['matched_offsets'] = np.cumsum(matched_lens)
    out_dict['angle_changes'] = np.concatenate(angles)

    return out_dict


def calc_angles(ids_dict, angles_dict, progen_exists, progen_exists_prev,
                ids_angle_prev, angle_slices_prev, verbose):

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
        list(zip(ids_dict['matched_offsets'][:-1],
                 ids_dict['matched_offsets'][1:])))
    orbiting_slices = np.array(
        list(zip(ids_dict['orbiting_offsets'][:-1],
                 ids_dict['orbiting_offsets'][1:])))
    angles = np.empty(len(ids_dict['matched_ids']), dtype=np.float16)
    orbiting_angle_ids = []
    orbiting_angles = []
    orbiting_angle_changes = np.empty(
        len(ids_dict['orbiting_ids']), dtype=np.float16)
    j = 0
    for p, asl, osl in zip(
            progen_exists, matched_slices, orbiting_slices):

        ids_angle_halo = ids_dict['matched_ids'][slice(*asl)]
        angles_halo = ids_dict['angle_changes'][slice(*asl)]

        if p in progen_exists_prev:
            aslp = angle_slices_prev[j]
            intersect_inds = np.where(np.in1d(
                ids_angle_halo, ids_angle_prev[slice(*aslp)]))[0]

            inds_angle = myin1d(
                ids_angle_prev[slice(*aslp)],
                ids_angle_halo[intersect_inds])
            angles_halo[intersect_inds] += angles_dict['angles'][
                slice(*aslp)][inds_angle]

        angles[slice(*asl)] = angles_halo

        orb_angles_halo = angles_halo[ids_dict['orbiting_inds'][slice(*osl)]]
        orb_angle_changes_halo = np.copy(orb_angles_halo)

        if p in progen_exists_prev:

            oaslp = angles_dict['orbiting_angle_slices'][j]

            orbiting_angle_ids_prev_halo = angles_dict['orbiting_angle_ids'][
                slice(*oaslp)]
            orbiting_angles_prev_halo = angles_dict['orbiting_angles'][
                slice(*oaslp)]

            intersect_bools = np.in1d(
                ids_dict['orbiting_ids'][slice(*osl)],
                orbiting_angle_ids_prev_halo)
            intersect_inds = np.where(intersect_bools)[0]
            nointersect_inds = np.where(~intersect_bools)[0]

            inds_orb_angle = myin1d(
                orbiting_angle_ids_prev_halo,
                ids_dict['orbiting_ids'][slice(*osl)][intersect_inds])

            orbiting_angle_ids_halo = np.copy(
                orbiting_angle_ids_prev_halo)
            orbiting_angles_halo = np.copy(orbiting_angles_prev_halo)
            orbiting_angles_halo[inds_orb_angle] = orb_angles_halo[
                intersect_inds]
            orb_angle_changes_halo[intersect_inds] -= \
                orbiting_angles_prev_halo[inds_orb_angle]

            orbiting_angle_ids_new_halo = ids_dict['orbiting_ids'][
                slice(*osl)][nointersect_inds]
            orbiting_angles_new_halo = orb_angles_halo[
                nointersect_inds]
            orbiting_angle_ids_halo = np.append(
                orbiting_angle_ids_halo, orbiting_angle_ids_new_halo)
            orbiting_angles_halo = np.append(
                orbiting_angles_halo, orbiting_angles_new_halo)

            j += 1

        else:

            orbiting_angle_ids_halo = ids_dict['orbiting_inds'][slice(*osl)]
            orbiting_angles_halo = np.copy(orb_angles_halo)

        orbiting_angle_ids.append(orbiting_angle_ids_halo.astype(np.uint32))
        orbiting_angles.append(orbiting_angles_halo.astype(np.float16))
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

    out_dict = {}
    out_dict['angles'] = angles
    out_dict['matched_slices'] = matched_slices
    out_dict['orbiting_angle_ids'] = orbiting_angle_ids
    out_dict['orbiting_angles'] = orbiting_angles
    out_dict['orbiting_angle_slices'] = orbiting_angle_slices
    out_dict['orbiting_angle_changes'] = orbiting_angle_changes

    return out_dict


def init_dicts(region_offsets, ids):

    ids_dict = {}
    ids_dict['orbiting_offsets'] = np.array([])
    ids_dict['orbiting_ids'] = np.array([])
    ids_dict['entered_offsets'] = region_offsets
    ids_dict['entered_ids'] = ids
    ids_dict['departed_offsets'] = np.array([])
    ids_dict['departed_ids'] = np.array([])
    ids_dict['matched_ids'] = np.array([])

    angles_dict = {}
    angles_dict['angles'] = np.array([])
    angles_dict['orbiting_angle_ids'] = np.array([])
    angles_dict['orbiting_angles'] = np.array([])
    angles_dict['orbiting_angle_slices'] = np.array([])
    angles_dict['orbiting_angle_changes'] = np.array([], dtype=np.float16)
    angles_dict['matched_slices'] = np.array([])

    return ids_dict, angles_dict


def read_resume_data(savefile, snapshot_number_resume, halo_ids_final):

    ids_dict, angles_dict = {}, {}

    with h5py.File(savefile, 'r') as hf:

        hfs = hf['snapshot_{}'.format('%03d' % snapshot_number_resume)]
        ids_dict['orbiting_offsets'] = hfs['orbiting_offsets'][:]
        ids_dict['orbiting_ids'] = hfs['orbiting_IDs'][:]
        ids_dict['entered_offsets'] = hfs['infalling_offsets'][:]
        ids_dict['entered_ids'] = hfs['infalling_IDs'][:]
        ids_dict['departed_offsets'] = hfs['departed_offsets'][:]
        ids_dict['departed_ids'] = hfs['departed_IDs'][:]

        progen_exists = myin1d(
            halo_ids_final, hfs['halo_ids_final'][:])

    with h5py.File(savefile+'.checkpoint', 'r') as hf:

        matched_ids = hf['matched_ids'][:]
        matched_slices = hf['matched_slices'][:]

        angles_dict['angles'] = hf['angles'][:]
        angles_dict['orbiting_angle_ids'] = hf['orbiting_angle_ids'][:]
        angles_dict['orbiting_angles'] = hf['orbiting_angles'][:]
        angles_dict['orbiting_angle_slices'] = hf['orbiting_angle_slices'][:]

    return ids_dict, angles_dict, matched_ids, matched_slices, progen_exists

def initialize_savefile(savefile, mode, box_size, verbose):

    with h5py.File(savefile, 'w') as hf:

        hf.attrs['mode'] = mode
        if box_size is not None:
            hf.attrs['box_size'] = box_size

    if verbose:
        print('Savefile initialized\n')


def save_to_file(savefile, ids_dict, angles_dict, region_positions, region_radii,
                 bulk_velocities, halo_ids, halo_ids_final, snapshot_number,
                 checkpoint, verbose):

    if verbose:
        print('Saving to file...')
        t0 = time.time()

    with h5py.File(savefile, 'r+') as hf:

        gsnap = hf.create_group(
            'snapshot_{}'.format('%0.3d' % snapshot_number))
        gsnap.create_dataset(
            'orbiting_offsets', data=ids_dict['orbiting_offsets'])
        gsnap.create_dataset('orbiting_IDs', data=ids_dict['orbiting_ids'])
        gsnap.create_dataset(
            'angles', data=angles_dict['orbiting_angle_changes'])

        gsnap.create_dataset(
            'infalling_offsets', data=ids_dict['entered_offsets'])
        gsnap.create_dataset('infalling_IDs', data=ids_dict['entered_ids'])

        gsnap.create_dataset(
            'departed_offsets', data=ids_dict['departed_offsets'])
        gsnap.create_dataset('departed_IDs', data=ids_dict['departed_ids'])

        gsnap.create_dataset('halo_ids', data=halo_ids)
        gsnap.create_dataset('halo_ids_final', data=halo_ids_final)
        gsnap.create_dataset('region_radii', data=region_radii)
        gsnap.create_dataset('region_positions', data=region_positions)
        gsnap.create_dataset('bulk_velocities', data=bulk_velocities)

    if checkpoint:

        with h5py.File(savefile+'.checkpoint', 'w') as hf:

            hf.create_dataset('matched_ids', data=ids_dict['matched_ids'])
            hf.create_dataset(
                'matched_slices', data=angles_dict['matched_slices'])
            hf.create_dataset('angles', data=angles_dict['angles'])
            hf.create_dataset(
                'orbiting_angle_ids',
                data=angles_dict['orbiting_angle_ids'])
            hf.create_dataset(
                'orbiting_angles', data=angles_dict['orbiting_angles'])
            hf.create_dataset(
                'orbiting_angle_slices', data=angles_dict[
                    'orbiting_angle_slices'])

    if verbose:
        print('Saved to file (took {} s)\n'.format(time.time() - t0))
