import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from orbitanalysis.utils import myin1d, magnitude


def filter_pericenters(hf, index, angle_condition=np.pi/2,
                       final_snapshot=None):

    snapshot_numbers = hf['SnapshotNumbers'][:]
    if final_snapshot is not None:
        snap_ind = np.where(snapshot_numbers == final_snapshot)[0][0]
        snapshot_numbers = snapshot_numbers[:snap_ind+1]

    peri_ids = []
    peri_angles = []
    for ii, snapshot_number in enumerate(snapshot_numbers):
        sdata = hf['Snapshot{}'.format('%0.3d' % snapshot_number)]

        orb_offsets = sdata['OrbitingOffsets'][()]
        inf_offsets = sdata['InfallingOffsets'][()]
        orb_ranges = list(zip(orb_offsets[:-1], inf_offsets))
        counts_offsets = np.hstack([[0], np.cumsum(
            inf_offsets - orb_offsets[:-1])])
        count_ranges = list(zip(counts_offsets[:-1], counts_offsets[1:]))

        orb_slice = slice(*orb_ranges[index])
        count_slice = slice(*count_ranges[index])

        ids_orb = sdata['IDs'][orb_slice]
        angles_orb = sdata['PhaseAngles'][orb_slice]
        counts = sdata['Counts'][count_slice]
        if ii > 0:
            departed = np.setdiff1d(ids_orb_prev, ids_orb)
            departed_inds = np.in1d(ids_orb_prev, departed)
            ids_orb_prev_ = np.delete(ids_orb_prev, departed_inds)
            counts_prev_ = np.delete(counts_prev, departed_inds)

            matched_inds = myin1d(ids_orb, ids_orb_prev_)
            first_peri_ids = np.setdiff1d(
                ids_orb, ids_orb_prev_, assume_unique=True)
            first_peri_inds = myin1d(ids_orb, first_peri_ids)

            peri_inds = np.where(counts[matched_inds] > counts_prev_)[0]
            nth_peri_ids = ids_orb[matched_inds[peri_inds]]
            new_peri_ids = np.concatenate([first_peri_ids, nth_peri_ids])
            new_peri_inds = np.concatenate(
                [first_peri_inds, matched_inds[peri_inds]])
            new_peri_angles = angles_orb[new_peri_inds]

            peri_ids.append(new_peri_ids)
            peri_angles.append(new_peri_angles)

        ids_orb_prev = ids_orb
        counts_prev = counts

    filter_ids = []
    for ii in range(1, len(peri_ids)+1):
        ids1, angles1 = peri_ids[-ii], peri_angles[-ii]
        filter_ids_ii = []
        for jj in range(ii+1, len(peri_ids)+1):
            ids2, angles2 = peri_ids[-jj], peri_angles[-jj]

            diff = np.setdiff1d(ids1, ids2)
            diff_inds = myin1d(ids1, diff)
            ids1_ = np.delete(ids1, diff_inds)
            angles1_ = np.delete(angles1, diff_inds)
            matched_inds = myin1d(ids2, ids1_)

            filter_inds = np.where(
                (angles1_ - angles2[matched_inds]) < angle_condition)[0]
            filter_ids_ii.append(ids1_[filter_inds])

            ids1, angles1 = diff, angles1[diff_inds]

        filter_inds_remaining = np.where(angles1 < angle_condition)[0]
        filter_ids_ii.append(ids1[filter_inds_remaining])

        filter_ids.append(np.concatenate(filter_ids_ii))
    filter_ids.append(np.array([]))
    filter_ids = [filter_ids[-ii] for ii in range(1, len(filter_ids)+1)]

    return filter_ids


def read_orbiting_decomposition(filename, snapshot_number, index,
                                angle_condition=np.pi/2, filter_ids=None,
                                position=None, snapshot=None):

    hf = h5py.File(filename, 'r')
    sdata = hf['Snapshot{}'.format('%0.3d' % snapshot_number)]

    orb_offsets = sdata['OrbitingOffsets'][()]
    inf_offsets = sdata['InfallingOffsets'][()]
    counts_offsets = np.hstack([[0], np.cumsum(
        inf_offsets - orb_offsets[:-1])])
    orb_slices = list(zip(orb_offsets[:-1], inf_offsets))
    inf_slices = list(zip(inf_offsets, orb_offsets[1:]))
    count_slices = list(zip(counts_offsets[:-1], counts_offsets[1:]))
    orb_slice = slice(*orb_slices[index])
    inf_slice = slice(*inf_slices[index])
    count_slice = slice(*count_slices[index])

    if angle_condition > 0.0:
        if filter_ids is None:
            filter_ids = filter_pericenters(hf, index, angle_condition)
        snap_ind = np.where(hf['SnapshotNumbers'][:] == snapshot_number)[0][0]
        filter_ids_snap = np.concatenate(filter_ids[:snap_ind+1])
        filter_ids_unique, filter_counts = np.unique(
            filter_ids_snap, return_counts=True)

    if 'ParticleMasses' in hf:
        particle_mass = hf['ParticleMasses'][index]
    else:
        particle_mass = None

    ids_orb = sdata['IDs'][orb_slice]
    if snapshot is None:
        coords_orb = sdata['Coordinates'][orb_slice]
        vels_orb = sdata['Velocities'][orb_slice]
        if 'Masses' in sdata:
            masses_orb = sdata['Masses'][orb_slice]
        else:
            masses_orb = np.repeat(particle_mass, len(ids_orb))
    else:
        inds_orb = myin1d(snapshot.ids, ids_orb)
        coords_orb = snapshot.coords[inds_orb] - position
        vels_orb = snapshot.vels[inds_orb]
        if isinstance(snapshot.masses, np.ndarray):
            masses_orb = snapshot.masses[inds_orb]
        else:
            masses_orb = np.repeat(snapshot.masses, len(ids_orb))

    ids_inf = sdata['IDs'][inf_slice]
    if snapshot is None:
        coords_inf = sdata['Coordinates'][inf_slice]
        vels_inf = sdata['Velocities'][inf_slice]
        if 'Masses' in sdata:
            masses_inf = sdata['Masses'][inf_slice]
        else:
            masses_inf = np.repeat(particle_mass, len(ids_inf))
    else:
        inds_inf = myin1d(snapshot.ids, ids_inf)
        coords_inf = snapshot.coords[inds_inf] - position
        vels_inf = snapshot.vels[inds_inf]
        if isinstance(snapshot.masses, np.ndarray):
            masses_inf = snapshot.masses[inds_inf]
        else:
            masses_inf = np.repeat(snapshot.masses, len(ids_inf))

    counts = sdata['Counts'][count_slice]

    if angle_condition > 0.0:
        departed = np.setdiff1d(filter_ids_unique, ids_orb, assume_unique=True)
        departed_inds = myin1d(filter_ids_unique, departed)
        filter_ids_unique_ = np.delete(filter_ids_unique, departed_inds)
        filter_counts_ = np.delete(filter_counts, departed_inds)
        inds_filter = myin1d(ids_orb, filter_ids_unique_)
        counts[inds_filter] -= filter_counts_

    ids_norb, coords_norb, vels_norb, masses_norb = {}, {}, {}, {}
    norbs = np.unique(counts[counts > 0])
    for n in norbs:
        norb_inds = np.argwhere(counts == n).flatten()
        ids_norb[n] = ids_orb[norb_inds]
        coords_norb[n] = coords_orb[norb_inds]
        vels_norb[n] = vels_orb[norb_inds]
        masses_norb[n] = masses_orb[norb_inds]

    if angle_condition > 0.0:
        inds_remove = np.argwhere(counts <= 0).flatten()
        ids_inf = np.append(ids_inf, ids_orb[inds_remove])
        coords_inf = np.append(coords_inf, coords_orb[inds_remove], axis=0)
        vels_inf = np.append(vels_inf, vels_orb[inds_remove], axis=0)
        masses_inf = np.append(masses_inf, masses_orb[inds_remove])

        ids_orb = np.delete(ids_orb, inds_remove)
        coords_orb = np.delete(coords_orb, inds_remove, axis=0)
        vels_orb = np.delete(vels_orb, inds_remove, axis=0)
        masses_orb = np.delete(masses_orb, inds_remove)
        counts = np.delete(counts, inds_remove)

    return (ids_orb, ids_norb, ids_inf), \
        (coords_orb, coords_norb, coords_inf), \
        (vels_orb, vels_norb, vels_inf), \
        (masses_orb, masses_norb, masses_inf), \
        counts


def plot_position_space(coords, counts, colormap='inferno_r', savefile=None):

    coords_orb, coords_norb, coords_inf = coords

    clrmap = mpl.colormaps[colormap]

    fig, axs = plt.subplots(
        ncols=4, figsize=(22.5, 7), gridspec_kw={
            'width_ratios': [20, 20, 20, 1]})
    cbar_ax = axs[3]
    norm = plt.Normalize(vmin=1, vmax=np.max(counts))
    mpl.colorbar.ColorbarBase(
        cbar_ax, cmap=clrmap, norm=norm, orientation='vertical')
    cbar_ax.set_title(r'$N_{\rm orbits}$', fontsize=18)
    cbar_ax.tick_params(labelsize=18)
    cbar_ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # orbiting & infalling
    for ii in [0, 2]:
        axs[ii].scatter(coords_inf[:, 0], coords_inf[:, 1], color='grey',
                        alpha=0.4, marker='.', s=0.2)
    for ii in [0, 1]:
        for n in np.unique(counts):
            axs[ii].scatter(coords_norb[n][:, 0], coords_norb[n][:, 1],
                            color=clrmap(norm(n-1)), alpha=0.4, marker='.',
                            s=0.2)

    lim = max(np.abs(np.array(list(axs[0].get_xlim()) +
                              list(axs[0].get_ylim()))))
    titles = ['orbiting + infalling', 'orbiting', 'infalling']
    for ax, title in zip(axs, titles):
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(r'$x$', fontsize=18)
        ax.set_ylabel(r'$y$', fontsize=18)
        ax.tick_params(labelsize=18)
        ax.set_title(title, fontsize=18)

    fig.tight_layout()
    fig.savefig(savefile, dpi=300)
    plt.close(fig)


def plot_phase_space(coords, vels, counts, colormap='inferno_r',
                     savefile=None):

    coords_orb, coords_norb, coords_inf = coords
    vels_orb, vels_norb, vels_inf = vels

    r_orb = magnitude(coords_orb)
    vr_orb = np.einsum('...i,...i', vels_orb, coords_orb) / r_orb
    r_inf = magnitude(coords_inf)
    vr_inf = np.einsum('...i,...i', vels_inf, coords_inf) / r_inf
    r_norb, vr_norb = {}, {}
    for n in np.unique(counts):
        r_norb[n] = magnitude(coords_norb[n])
        vr_norb[n] = np.einsum('...i,...i', vels_norb[n], coords_norb[n]) / \
                     r_norb[n]

    clrmap = mpl.colormaps[colormap]

    fig, axs = plt.subplots(
        ncols=4, figsize=(22.5, 7), gridspec_kw={
            'width_ratios': [20, 20, 20, 1]})
    cbar_ax = axs[3]
    norm = plt.Normalize(vmin=1, vmax=np.max(counts))
    mpl.colorbar.ColorbarBase(
        cbar_ax, cmap=clrmap, norm=norm, orientation='vertical')
    cbar_ax.set_title(r'$N_{\rm orbits}$', fontsize=18)
    cbar_ax.tick_params(labelsize=18)
    cbar_ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # orbiting & infalling
    for ii in [0, 2]:
        axs[ii].scatter(r_inf, vr_inf, color='grey', alpha=0.4,
                        marker='.', s=0.2)
    for ii in [0, 1]:
        for n in np.unique(counts):
            axs[ii].scatter(r_norb[n], vr_norb[n], color=clrmap(norm(n-1)),
                            alpha=0.4, marker='.', s=0.2)

    xlim = max(np.abs(np.array(list(axs[0].get_xlim()))))
    ylim = max(np.abs(np.array(list(axs[0].get_ylim()))))
    titles = ['orbiting + infalling', 'orbiting', 'infalling']
    for ax, title in zip(axs, titles):
        ax.set_xlim(0, xlim)
        ax.set_ylim(-ylim, ylim)
        ax.set_xlabel(r'$r$', fontsize=18)
        ax.set_ylabel(r'$v_r$', fontsize=18)
        ax.tick_params(labelsize=18)
        ax.set_title(title, fontsize=18)

    fig.tight_layout()
    fig.savefig(savefile, dpi=300)
    plt.close(fig)
