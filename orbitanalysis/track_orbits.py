import numpy as np
import time
import h5py
from pathos.multiprocessing import Pool

from orbitanalysis.utils import myin1d, recenter_coordinates, hubble_parameter


def track_orbits(snapshot_numbers, main_branches, regions, load_snapshot_data,
                 savefile, mode='pericentric', checkpoint=False, resume=False,
                 npool=1, verbose=True):

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
        * redshift : float - the redshift of the snapshot.
        * H0 : float - the Hubble parameter at z = 0.
        * Omega_m : float - the matter density parameter.
        * Omega_L : float - the dark energy density parameter.
        * Omega_k : float (optional) - the curvature density parameter. If not
                    provided, Omega_k = 0 will be assumed.
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
        print('Resuming from file...\n')
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
            print('-' * 30, '\n')
            print('Snapshot {}\n'.format('%03d' % snapshot_number))

        halo_exists = np.argwhere(halo_ids != -1).flatten()
        if len(halo_exists) == 0:
            if started is False:
                istart = i + 1
            continue
        halo_ids_ = halo_ids[halo_exists]

        region_positions, region_radii, region_bulk_vels = regions(
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

        if 'Omega_k' not in snapshot:
            snapshot['Omega_k'] = 0
        H = hubble_parameter(
            snapshot['redshift'], snapshot['H0'], snapshot['Omega_m'],
            snapshot['Omega_L'], snapshot['Omega_k'])

        if i == 0 and not resume:
            if 'box_size' in snapshot:
                box_size = snapshot['box_size']
            else:
                box_size = None
            initialize_savefile(savefile, mode, box_size, verbose)

        def track(j):

            hind = halo_exists[j]
            region_slice = region_slices[j]
            region_position = region_positions[j]

            rhs, radial_vels, bulk_vels = region_frame(
                snapshot, region_slice, region_position,
                None if region_bulk_vels is None else region_bulk_vels[j], H)

            sl = slice(*region_slice)
            npart = np.diff(region_slice)[0]

            if i > istart:

                if hind in progen_exists:

                    ind = np.argwhere(progen_exists == hind).flatten()[0]
                    sl_prev = slice(*region_slices_prev[ind])

                    apsis_dict = \
                        compare_radial_velocities(
                        snapshot['ids'][sl], ids_prev[sl_prev], radial_vels,
                        radial_vels_prev[sl_prev], rhs, rhats_prev[sl_prev],
                        mode)

                    angs, apsis_angs = calc_angles(
                        npart, angles_prev[sl_prev], apsis_dict)
                    
                    return rhs, radial_vels, bulk_vels, angs, \
                        apsis_dict['apsis_ids'], apsis_angs
                
                else:
                    angs = np.zeros(npart, dtype=np.float16)
            
            else:
                angs = np.zeros(npart, dtype=np.float16)
            
            return rhs, radial_vels, bulk_vels, angs, None, None
    
        if npool is None:
            result = []
            for idx in range(len(halo_exists)):
                result.append(track(idx))
        else:
            result = Pool(npool).map(track, np.arange(len(halo_exists)))

        rhats, radial_velocities, bulk_velocities, angles = [], [], [], []
        apsis_ids, apsis_angles = [], []
        for x in result:
            rhats.append(x[0])
            radial_velocities.append(x[1])
            bulk_velocities.append(x[2])
            angles.append(x[3])
            if x[4] is not None:
                apsis_ids.append(x[4])
                apsis_angles.append(x[5])

        angles = np.concatenate(angles)

        if i > istart:

            apsis_offsets = np.cumsum(
                [0]+[len(apsids) for apsids in apsis_ids])
            apsis_ids = np.concatenate(apsis_ids)
            apsis_angles = np.concatenate(apsis_angles)

            region_positions_ = -np.ones((len(halo_ids), 3))
            region_positions_[halo_exists] = region_positions
            region_radii_ = -np.ones(len(halo_ids))
            region_radii_[halo_exists] = region_radii
            bulk_velocities_ = -np.ones((len(halo_ids), 3))
            bulk_velocities_[halo_exists] = np.array(bulk_velocities)

            save_to_file(
                savefile, apsis_ids, apsis_offsets, apsis_angles,
                region_positions_, region_radii_, bulk_velocities_, halo_ids,
                main_branches[-1][progen_exists], snapshot_number, mode,
                checkpoint, angles, verbose)
        
        else:
            if resume:
                with h5py.File(savefile+'.checkpoint', 'r') as hf:
                    angles = hf['angles'][:]
        
        rhats_prev = np.concatenate(rhats)
        radial_vels_prev = np.concatenate(radial_velocities)

        ids_prev = snapshot['ids']
        angles_prev = angles
        region_slices_prev = region_slices
        progen_exists = halo_exists

    if verbose:
        print('Finished pericenter detection (took {} s)\n'.format(
            time.time() - tstart))


def region_frame(snapshot, region_slice, region_position, region_bulk_vel, H):

    """
    Transform coordinates and velocities to region frames and compute radial
    velocities.
    """

    if 'box_size' in snapshot:
        region_coords = recenter_coordinates(
            snapshot['coordinates'][slice(*region_slice)]-region_position,
            snapshot['box_size'])
    else:
        region_coords = snapshot['coordinates'][slice(*region_slice)] - \
            region_position

    if region_bulk_vel is None:
        region_bulk_vel_cat = False
    else:
        region_bulk_vel_cat = True

    if isinstance(snapshot['masses'], np.ndarray):
        if not region_bulk_vel_cat:
            bulk_vel = np.sum(
                snapshot['masses'][slice(*region_slice)][:, np.newaxis] *
                snapshot['velocities'][slice(*region_slice)], axis=0) / \
                    np.sum(snapshot['masses'][slice(*region_slice)])
        else:
            bulk_vel = region_bulk_vel
        region_vels = snapshot['velocities'][slice(*region_slice)] - \
            bulk_vel + H * region_coords / (1 + snapshot['redshift'])
    else:
        if not region_bulk_vel_cat:
            bulk_vel = np.mean(
                snapshot['velocities'][slice(*region_slice)], axis=0)
        else:
            bulk_vel = region_bulk_vel
        region_vels = snapshot['velocities'][slice(*region_slice)] - \
            bulk_vel + H * region_coords / (1 + snapshot['redshift'])

    rads = np.sqrt(np.einsum('...i,...i', region_coords, region_coords))
    rhats = region_coords / rads[:, np.newaxis]
    radial_vels = np.einsum('...i,...i', region_vels, rhats)

    return rhats, radial_vels, bulk_vel


def compare_radial_velocities(ids, ids_prev, radial_vels, radial_vels_prev,
                              rhat, rhat_prev, mode):

    """
    Identify sign flips in the radial velocity between snapshots.
    """

    departed = np.setdiff1d(ids_prev, ids)
    inds_departed = np.where(np.in1d(ids_prev, departed))[0]
    ids_prev_ = np.delete(ids_prev, inds_departed)
    radial_vels_prev_ = np.delete(radial_vels_prev, inds_departed)
    rhat_prev_ = np.delete(rhat_prev, inds_departed, axis=0)

    inds_match = myin1d(ids, ids_prev_)
    ids_match = ids[inds_match]
    radial_vels_match = radial_vels[inds_match]
    rhat_match = rhat[inds_match]

    if mode == 'pericentric':
        cond = (radial_vels_prev_ < 0) & (radial_vels_match > 0)
    elif mode == 'apocentric':
        cond = (radial_vels_prev_ > 0) & (radial_vels_match < 0)
    apsis_inds = np.argwhere(cond).flatten()
    apsis_ids = ids_prev_[apsis_inds]

    out_dict = {}
    out_dict['apsis_inds'] = apsis_inds
    out_dict['apsis_ids'] = apsis_ids
    out_dict['ids_match'] = ids_match
    out_dict['inds_match'] = inds_match
    out_dict['inds_departed'] = inds_departed
    out_dict['angle_changes'] = np.arccos(
        np.einsum('...i,...i', rhat_prev_, rhat_match))

    return out_dict
        

def calc_angles(npart, angles_prev, apsis_dict):

    """
    Return the angles by which particles identified as going through pericenter
    have advanced since their last detected pericenter, or since entering the
    region.

    Particles orbiting in subhalos will generally only advance a small
    angle between pericenters, allowing these spurious detections to be removed
    by making a cut in the angle.
    """

    angles_prev_ = np.delete(angles_prev, apsis_dict['inds_departed'])
    angles_ = angles_prev_ + apsis_dict['angle_changes']
    apsis_angles = np.copy(angles_[apsis_dict['apsis_inds']])

    angles_[apsis_dict['apsis_inds']] = 0.0

    angles = np.zeros(npart)
    angles[apsis_dict['inds_match']] = angles_

    return angles.astype(np.float16), apsis_angles.astype(np.float16)


def initialize_savefile(savefile, mode, box_size, verbose):

    with h5py.File(savefile, 'w') as hf:

        hf.attrs['mode'] = mode
        if box_size is not None:
            hf.attrs['box_size'] = box_size

    if verbose:
        print('Savefile initialized\n')


def save_to_file(savefile, apsis_ids, apsis_offsets, apsis_angles,
                 region_positions, region_radii, bulk_velocities, halo_ids,
                 halo_ids_final, snapshot_number, mode, checkpoint, angles,
                 verbose):

    if verbose:
        print('Saving to file...')
        t0 = time.time()

    with h5py.File(savefile, 'r+') as hf:

        gsnap = hf.create_group('snapshot_{}'.format('%0.3d'%snapshot_number))

        gsnap.create_dataset('region_offsets', data=apsis_offsets)
        gsnap.create_dataset('{}er_IDs'.format(mode[:-3]), data=apsis_ids)
        gsnap.create_dataset('angles', data=apsis_angles)

        gsnap.create_dataset('halo_ids', data=halo_ids)
        gsnap.create_dataset('halo_ids_final', data=halo_ids_final)
        gsnap.create_dataset('region_radii', data=region_radii)
        gsnap.create_dataset('region_positions', data=region_positions)
        gsnap.create_dataset('bulk_velocities', data=bulk_velocities)

    if checkpoint:

        with h5py.File(savefile+'.checkpoint', 'w') as hf:

            hf.create_dataset('angles', data=angles)

    if verbose:
        print('Saved to file (took {} s)\n'.format(time.time() - t0))
