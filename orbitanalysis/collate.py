import os
import numpy as np
import h5py
import time

from orbitanalysis.utils import myin1d, magnitude


def collate_orbit_history(savefile, save_properties=True, verbose=True):

    if verbose:
        print('Collating data...')
        t0 = time.time()

    ext = savefile.split('.')[-1]
    if ext == 'h5' or ext == 'hdf5':
        readfile = savefile[:-(len(ext) + 1)] + '_onthefly.' + ext
    else:
        readfile = savefile + '_onthefly'

    hf = h5py.File(readfile, 'r')
    snapshot_numbers = np.sort([
        int(x[-3:]) for x in list(hf.keys()) if x[:8] == 'Snapshot'])[1:]
    nsnap = len(snapshot_numbers)
    halo_ids = np.flip(hf['HaloIDs'][:nsnap], axis=0)

    orbiting_offsets_prev = None
    infalling_offsets_prev = None
    ids_prev = None
    coords_prev = None
    angles_prev = None
    orbit_counts_prev = None

    hf_new = h5py.File(savefile, 'w')
    for si, snapshot_number in enumerate(snapshot_numbers):

        sstr = 'Snapshot{}'.format('%0.3d' % snapshot_number)
        hfsnap = hf[sstr]
        orbiting_ids_new = hfsnap['OrbitingIDs'][()]
        ids_all = hfsnap['IDs'][()]
        coords_all = hfsnap['Coordinates'][()]
        if save_properties:
            vels_all = hfsnap['Velocities'][()]
            if 'Masses' in hfsnap:
                masses_all = hfsnap['Masses'][()]
            else:
                masses_all = None
        else:
            vels_all, masses_all = None, None
        offsets_all = hfsnap['Offsets'][()]

        slices, hinds, norb = get_slices(hfsnap, orbiting_offsets_prev,
                                         infalling_offsets_prev, halo_ids, si)

        nmax = np.sum(norb) + len(orbiting_ids_new)
        orbit_counts = np.empty(nmax, dtype=np.int32)
        angles = np.empty(len(ids_all), dtype=np.float32)

        idx = 0
        orbiting_lengths = []
        for hi, (orbslice_prev, orbslice_new, intslice_prev, intslice_new,
                 countslice) in enumerate(slices):

            hasprev = False if np.all(np.isnan(orbslice_prev)) else True

            slon, slin = slice(*orbslice_new), slice(*intslice_new)

            if hasprev:
                slop, slip = slice(*orbslice_prev), slice(*intslice_prev)
                oids = ids_prev[slop]
                departed = np.setdiff1d(
                    oids, ids_all[slin], assume_unique=True)
                departed_inds = np.where(np.in1d(oids, departed))[0]
                oids = np.delete(oids, departed_inds)
                oids = np.concatenate((oids, orbiting_ids_new[slon]))
                oids, inds, c = np.unique(
                    oids, return_index=True, return_counts=True)
            else:
                oids = orbiting_ids_new[slon]
            n = len(oids)

            orbit_counts[idx:idx + n] = np.ones(n, dtype=np.int32)
            if hasprev:
                inds_repeats = inds[c > 1]
                ocounts = np.delete(
                    orbit_counts_prev[slice(*countslice)], departed_inds)
                ocounts[inds_repeats] += 1
                orbit_counts[np.sort(np.argwhere(
                    inds < len(ocounts)).flatten()) + idx] = ocounts

            orbinds = myin1d(ids_all[slin], oids)
            infinds = list(set(range(len(ids_all[slin]))).difference(orbinds))
            order = np.concatenate((orbinds, infinds)).astype(int)
            ids_all[slin] = ids_all[slin][order]

            coords_all[slin] = coords_all[slin][order]
            if save_properties:
                vels_all[slin] = vels_all[slin][order]
                if 'Masses' in hfsnap:
                    masses_all[slin] = masses_all[slin][order]

            angles[slin] = np.zeros(
                np.diff(intslice_new)[0], dtype=np.float32)
            if hasprev:
                entered = np.setdiff1d(
                    ids_all[slin], ids_prev[slip], assume_unique=True)
                entered_inds = np.in1d(ids_all[slin], entered)
                mask = np.ones(np.diff(intslice_new)[0], dtype=bool)
                mask[entered_inds, ] = False
                order = myin1d(ids_prev[slip], ids_all[slin][mask])
                angles_new = np.arccos(np.einsum(
                    '...i,...i', coords_prev[slip][order],
                    coords_all[slin][mask]) / (
                        magnitude(coords_prev[slip][order]) * magnitude(
                    coords_all[slin][mask])))
                angles[slin][mask] = angles_prev[slip][order] + angles_new

            idx += n
            orbiting_lengths.append(n)

        orbiting_offsets_prev = offsets_all
        infalling_offsets_prev = offsets_all[:-1] + np.array(orbiting_lengths)
        ids_prev = ids_all
        coords_prev = coords_all
        angles_prev = angles
        orbit_counts_prev = orbit_counts[:int(np.sum(orbiting_lengths))]

        gsnap = hf_new.create_group(sstr)
        write_data(gsnap,
                   ids_all,
                   orbit_counts[:int(np.sum(orbiting_lengths))],
                   offsets_all,
                   offsets_all[:-1] + np.array(orbiting_lengths),
                   coords_all,
                   vels_all,
                   masses_all,
                   angles,
                   save_properties)

        print('Snapshot {} done'.format(snapshot_number))

    masses = np.flip(hf['ParticleMasses'][:nsnap], axis=0)
    if np.all(masses == 0.0):
        masses = None
    write_global_data(hf, hf_new,
                      snapshot_numbers,
                      np.flip(hf['Redshifts'][:nsnap]),
                      halo_ids,
                      np.flip(hf['Positions'][:nsnap], axis=0),
                      np.flip(hf['Radii'][:nsnap], axis=0),
                      masses
                      )

    hf.close()
    hf_new.close()

    os.remove(readfile)

    if verbose:
        print('Collation done (took {} s)'.format(time.time() - t0))


###############################################################################


def get_slices(hfsnap, orbiting_offsets_prev, infalling_offsets_prev, halo_ids,
               si):

    hids, hids_prev = halo_ids[si], halo_ids[si-1]
    inds = np.argwhere(hids > -1).flatten()
    hids_prev = hids_prev[inds]
    hids = hids[inds]

    orbiting_offsets_new = hfsnap['OrbitingOffsets'][()]
    offsets_all = hfsnap['Offsets'][()]

    orbslices_new = np.array(list(
        zip(orbiting_offsets_new[:-1], orbiting_offsets_new[1:])))
    intslices_new = np.array(list(zip(offsets_all[:-1], offsets_all[1:])))

    if si > 0:

        norb = infalling_offsets_prev - orbiting_offsets_prev[:-1]
        counts_offsets_prev = np.concatenate([[0], np.cumsum(norb)])
        countslices = np.array(list(
            zip(counts_offsets_prev[:-1], counts_offsets_prev[1:])))

        orbslices_prev = np.empty((len(orbslices_new), 2), dtype=np.int32)
        intslices_prev = np.empty((len(orbslices_new), 2), dtype=np.int32)
        inds_noprev = np.argwhere(hids_prev == -1).flatten()
        mask = np.ones(len(orbslices_new), dtype=bool)
        mask[inds_noprev, ] = False
        orbslices_prev[mask, :] = np.array(list(
            zip(orbiting_offsets_prev[:-1], infalling_offsets_prev)))
        intslices_prev[mask, :] = np.array(list(
            zip(orbiting_offsets_prev[:-1], orbiting_offsets_prev[1:])))
        if len(inds_noprev) != 0:
            tmp = np.ones((len(inds_noprev), 2))
            tmp[:] = np.nan
            orbslices_prev[inds_noprev] = tmp
            intslices_prev[inds_noprev] = tmp
    else:
        norb = [0]
        orbslices_prev = np.ones((len(hids), 2))
        intslices_prev = np.ones((len(hids), 2))
        orbslices_prev[:] = np.nan
        intslices_prev[:] = np.nan

        countslices = orbslices_prev

    return zip(orbslices_prev, orbslices_new, intslices_prev, intslices_new,
               countslices), inds, norb


def write_data(gsnap, ids, counts, orb_offsets, inf_offsets, coords, vels,
               masses, angles, save_properties):

    gsnap.create_dataset('IDs', data=ids)
    gsnap.create_dataset('Counts', data=counts)
    gsnap.create_dataset('OrbitingOffsets', data=orb_offsets)
    gsnap.create_dataset('InfallingOffsets', data=inf_offsets)
    if save_properties:
        gsnap.create_dataset('Coordinates', data=coords)
        gsnap.create_dataset('Velocities', data=vels)
        if masses is not None:
            gsnap.create_dataset('Masses', data=masses)
    gsnap.create_dataset('Angles', data=angles)


def write_global_data(hf, hf_new, snapshot_numbers, redshifts, halo_indices,
                      positions, radii, masses):

    hf_new.create_dataset('SnapshotNumbers', data=snapshot_numbers)
    hf_new.create_dataset('Redshifts', data=redshifts)
    hf_new.create_dataset('HaloIndices', data=halo_indices)
    hf_new.create_dataset('Positions', data=positions)
    hf_new.create_dataset('Radii', data=radii)
    if masses is not None:
        hf_new.create_dataset('ParticleMasses', data=masses)

    head = hf_new.create_group('Options')
    for attr in list(hf['Options'].attrs):
        head.attrs[attr] = hf['Options'].attrs[attr]
