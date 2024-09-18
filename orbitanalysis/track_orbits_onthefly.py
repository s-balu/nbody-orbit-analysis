import numpy as np
import time
import h5py

from orbitanalysis.utils import myin1d, recenter_coordinates


def track_orbits(snapshot_number, progenitor_links, regions, load_snapshot_data,
                 savefile, mode='pericentric', verbose=True):

    """

    """

    if (mode != 'pericentric') and (mode != 'apocentric'):
        raise ValueError(
            "Orbit detection mode not recognized. Please specify either "
            "'pericentric' or 'apocentric'.")

    ids, rhats, radial_vels, region_positions, region_radii, bulk_velocities, \
          region_slices = [], [], [], [], [], [], []
    for s, halo_ids_ in zip(
            [snapshot_number, snapshot_number-1], progenitor_links):

        halo_exists = np.argwhere(halo_ids_ != -1).flatten()
        halo_ids = halo_ids_[halo_exists]

        region_pos, region_rad = regions(s, halo_ids)
        region_pos_ = repack(region_pos, len(halo_ids_), halo_exists)
        region_rad_ = repack(region_rad, len(halo_ids_), halo_exists)
        region_positions.append(region_pos_)
        region_radii.append(region_rad_)

        snapshot = load_snapshot_data(s, region_pos, region_rad)
        ids.append(snapshot['ids'])

        offsets = list(snapshot['region_offsets']) + [len(snapshot['ids'])]
        slices = np.array(list(zip(offsets[:-1], offsets[1:])))
        slices_ = repack(slices, len(halo_ids_), halo_exists)
        region_slices.append(slices_)

        x = region_frame(snapshot, slices_, region_pos_, verbose)
        rhats.append(x[0])
        radial_vels.append(x[1])
        bulk_velocities.append(x[2])

        if 'box_size' in snapshot:
            box_size = snapshot['box_size']
        else:
            box_size = None

    ids_dict = compare_radial_velocities(
        ids[0], ids[1], radial_vels[0], radial_vels[1], rhats[0], rhats[1],
        region_slices[0], region_slices[1], mode, verbose)

    save_to_file(
        savefile, ids_dict, region_positions, region_radii, bulk_velocities,
        progenitor_links, snapshot_number, box_size, mode, verbose)


def repack(arr, length, inds):

    shape = np.array(np.shape(arr))
    shape[0] = length
    arr_ = -np.ones(tuple(shape), dtype=arr.dtype)
    arr_[inds] = arr

    return arr_


def region_frame(snapshot, region_slices, region_positions, verbose):

    """
    Transform coordinates and velocities to region frames and compute radial
    velocities.
    """

    if verbose:
        print('Transforming to region frames...')
        t0 = time.time()

    region_coords = np.empty(
        np.shape(snapshot['coordinates']), dtype=snapshot['coordinates'].dtype)
    if 'box_size' in snapshot:
        for sl, pos in zip(region_slices, region_positions):
            region_coords[slice(*sl), :] = recenter_coordinates(
                snapshot['coordinates'][slice(*sl)]-pos, snapshot['box_size'])
    else:
        for sl, pos in zip(region_slices, region_positions):
            region_coords[slice(*sl), :] = snapshot['coordinates'][
                slice(*sl)] - pos

    region_vels = np.empty(
        np.shape(snapshot['velocities']), dtype=snapshot['velocities'].dtype)
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
        print('Transformed to region frames in {} s\n'.format(
            time.time()-t0))

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

        if np.diff(sl_prev) > 0:

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

        else:

            entered_ids.append(ids[slice(*sl)])
            orbiting_inds.append(np.array([], dtype=ids.dtype))
            orbiting_ids.append(np.array([], dtype=ids.dtype))
            departed_ids.append(np.array([], dtype=ids.dtype))
            matched_ids.append(np.array([], dtype=ids.dtype))
            angles.append(np.array([], dtype=ids.dtype))

    if verbose:
        print('Identified {}ers in {} s\n'.format(mode[:8], time.time()-t0))

    orbiting_lens = [0] + [len(orbids) for orbids in orbiting_ids]
    entered_lens = [0] + [len(entered) for entered in entered_ids]
    departed_lens = [0] + [len(departed) for departed in departed_ids]
    matched_lens = [0] + [len(matched) for matched in matched_ids]

    out_dict = {}
    out_dict['{}er_ids'.format(mode[:8])] = np.concatenate(orbiting_ids)
    out_dict['{}er_inds'.format(mode[:8])] = np.concatenate(orbiting_inds)
    out_dict['{}er_offsets'.format(mode[:8])] = np.cumsum(orbiting_lens)
    out_dict['entered_ids'] = np.concatenate(entered_ids)
    out_dict['entered_offsets'] = np.cumsum(entered_lens)
    out_dict['departed_ids'] = np.concatenate(departed_ids)
    out_dict['departed_offsets'] = np.cumsum(departed_lens)
    out_dict['matched_ids'] = np.concatenate(matched_ids)
    out_dict['matched_offsets'] = np.cumsum(matched_lens)
    out_dict['angle_changes'] = np.concatenate(angles)

    return out_dict


def save_to_file(savefile, ids_dict, region_positions, region_radii,
                 bulk_velocities, progenitor_links, snapshot_number,
                 box_size, mode, verbose):

    if verbose:
        print('Saving to file...')
        t0 = time.time()

    # region_positions_ = -np.ones(
    #     (*np.shape(progenitor_links), 3), dtype=region_positions[0].dtype)
    # region_radii_ = -np.ones(
    #     np.shape(progenitor_links), dtype=region_radii[0].dtype)
    # bulk_velocities_ = -np.ones(
    #     (*np.shape(progenitor_links), 3), dtype=bulk_velocities[0].dtype)
    # progen_exist = np.argwhere(progenitor_links[1] != -1).flatten()
    # for i, inds in enumerate(
    #         [np.arange(len(progenitor_links[0])), progen_exist]):
    #     region_positions_[i, inds] = region_positions[i]
    #     region_radii_[i, inds] = region_radii[i]
    #     bulk_velocities_[i, inds] = bulk_velocities[i]

    with h5py.File(savefile.format('%0.3d' % snapshot_number), 'w') as hf:

        tag = mode[:8] + 'er'
        hf.create_dataset(tag+'_offsets', data=ids_dict[tag+'_offsets'])
        hf.create_dataset(tag+'_IDs', data=ids_dict[tag+'_ids'])
        hf.create_dataset('angles', data=ids_dict['angle_changes'])

        hf.create_dataset('entered_offsets', data=ids_dict['entered_offsets'])
        hf.create_dataset('entered_IDs', data=ids_dict['entered_ids'])

        hf.create_dataset(
            'departed_offsets', data=ids_dict['departed_offsets'])
        hf.create_dataset('departed_IDs', data=ids_dict['departed_ids'])

        hf.create_dataset('progenitor_links', data=progenitor_links)
        hf.create_dataset('region_radii', data=region_radii)
        hf.create_dataset('region_positions', data=region_positions)
        hf.create_dataset('bulk_velocities', data=bulk_velocities)

        if box_size is not None:
            hf.attrs['box_size'] = box_size

    if verbose:
        print('Saved to file in {} s\n'.format(time.time()-t0))
