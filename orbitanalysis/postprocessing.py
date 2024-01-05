import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from orbitanalysis.utils import myin1d, magnitude


class OrbitDecomposition:

    def __init__(self, filename):

        self.datafile = h5py.File(filename, 'r+')

        self.halo_indices = self.datafile['HaloIndices'][:]
        if 'ParticleMasses' in self.datafile:
            self.particle_masses = self.datafile['ParticleMasses']
        self.halo_positions = self.datafile['Positions'][:]
        self.halo_radii = self.datafile['Radii'][:]
        self.redshifts = self.datafile['Redshifts'][:]
        self.snapshot_numbers = self.datafile['SnapshotNumbers'][:]

        self.number_of_radii = self.datafile['Options'].attrs['NumRadii']
        self.particle_type = self.datafile['Options'].attrs['PartType']
        self.snapshot_directory = self.datafile['Options'].attrs['SnapDir']

    def correct_counts_and_save_to_file(self, angle_condition=np.pi/2):

        for halo_index in self.halo_indices[-1]:
            self.correct_counts(halo_index, angle_condition, save_to_file=True)

    def correct_counts(self, halo_index, angle_condition=np.pi/2,
                       snapshot_number=None, filtered_ids=None,
                       save_to_file=False):

        if filtered_ids is None:
            filtered_ids = self.filter_pericenters(
                halo_index, angle_condition, final_snapshot=snapshot_number)

        hind = np.argwhere(self.halo_indices[-1] == halo_index).flatten()[0]

        if snapshot_number is None:
            no_progen = np.count_nonzero(self.halo_indices[:, hind] == -1)
            snapshot_numbers = self.snapshot_numbers[no_progen:]
        else:
            snapshot_numbers = np.array([snapshot_number])

        for snap_num in snapshot_numbers:

            if snapshot_number is None:
                snap_ind = np.argwhere(
                    snapshot_numbers == snap_num).flatten()[0]
                filtered_ids_snap = np.concatenate(filtered_ids[:snap_ind + 1])
            else:
                filtered_ids_snap = np.concatenate(filtered_ids)

            filtered_ids_unique, filter_counts = np.unique(
                filtered_ids_snap, return_counts=True)

            sdata = self.datafile['Snapshot{}'.format('%0.3d' % snap_num)]
            orb_offsets = sdata['OrbitingOffsets'][()]
            inf_offsets = sdata['InfallingOffsets'][()]
            orb_ranges = list(zip(orb_offsets[:-1], inf_offsets))
            counts_offsets = np.hstack([[0], np.cumsum(
                inf_offsets - orb_offsets[:-1])])
            count_ranges = list(
                zip(counts_offsets[:-1], counts_offsets[1:]))

            orb_slice = slice(*orb_ranges[hind])
            count_slice = slice(*count_ranges[hind])

            ids_orb = sdata['IDs'][orb_slice]

            departed = np.setdiff1d(
                filtered_ids_unique, ids_orb, assume_unique=True)
            departed_inds = myin1d(filtered_ids_unique, departed)
            filter_ids_unique_ = np.delete(
                filtered_ids_unique, departed_inds)
            filter_counts_ = np.delete(filter_counts, departed_inds)
            inds_filter = myin1d(ids_orb, filter_ids_unique_)

            if save_to_file:
                if 'CorrectedCounts' not in sdata:
                    sdata.create_dataset(
                        'CorrectedCounts', data=np.copy(sdata['Counts'][:]))

                counts_corrected = np.copy(
                    sdata['Counts'][count_slice][inds_filter])
                counts_corrected -= filter_counts_
                counts_corrected[counts_corrected < 0] = 0

                temp = np.copy(sdata['CorrectedCounts'])
                temp[count_slice][inds_filter] = counts_corrected
                sdata['CorrectedCounts'][:] = temp
            else:
                counts = np.copy(sdata['Counts'][count_slice])
                counts[inds_filter] -= filter_counts_
                counts[counts < 0] = 0
                return counts

    def filter_pericenters(self, halo_index, angle_condition,
                           final_snapshot=None):

        hind = np.argwhere(self.halo_indices[-1] == halo_index).flatten()[0]
        no_progen = np.count_nonzero(self.halo_indices[:, hind] == -1)
        snapshot_numbers = self.snapshot_numbers[no_progen:]

        if final_snapshot is not None:
            snap_ind = np.where(snapshot_numbers == final_snapshot)[0][0]
            snapshot_numbers = snapshot_numbers[:snap_ind+1]

        peri_ids = []
        peri_angles = []
        for ii, snapshot_number in enumerate(snapshot_numbers):
            sdata = self.datafile[
                'Snapshot{}'.format('%0.3d' % snapshot_number)]

            orb_offsets = sdata['OrbitingOffsets'][()]
            inf_offsets = sdata['InfallingOffsets'][()]
            orb_ranges = list(zip(orb_offsets[:-1], inf_offsets))
            counts_offsets = np.hstack([[0], np.cumsum(
                inf_offsets - orb_offsets[:-1])])
            count_ranges = list(zip(counts_offsets[:-1], counts_offsets[1:]))

            orb_slice = slice(*orb_ranges[hind])
            count_slice = slice(*count_ranges[hind])

            ids_orb = sdata['IDs'][orb_slice]
            angles_orb = sdata['Angles'][orb_slice]
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

        filtered_ids = []
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

            filtered_ids.append(np.concatenate(filter_ids_ii))
        filtered_ids.append(np.array([], dtype=type(filtered_ids[0][0])))
        filtered_ids = [
            filtered_ids[-ii] for ii in range(1, len(filtered_ids)+1)]

        return filtered_ids

    def get_halo_decomposition_at_snapshot(self, snapshot_number, halo_index,
                                           use_corrected=False,
                                           angle_condition=None,
                                           filtered_ids=None,
                                           snapshot_data=None):

        hind = np.argwhere(self.halo_indices[-1] == halo_index).flatten()[0]

        sdata = self.datafile['Snapshot{}'.format('%0.3d' % snapshot_number)]

        orb_offsets = sdata['OrbitingOffsets'][()]
        inf_offsets = sdata['InfallingOffsets'][()]
        counts_offsets = np.hstack([[0], np.cumsum(
            inf_offsets - orb_offsets[:-1])])
        orb_slices = list(zip(orb_offsets[:-1], inf_offsets))
        inf_slices = list(zip(inf_offsets, orb_offsets[1:]))
        count_slices = list(zip(counts_offsets[:-1], counts_offsets[1:]))
        orb_slice = slice(*orb_slices[hind])
        inf_slice = slice(*inf_slices[hind])
        count_slice = slice(*count_slices[hind])

        snap_ind = np.where(self.snapshot_numbers == snapshot_number)[0][0]
        self.halo_radius = self.halo_radii[snap_ind, hind]
        self.halo_position = self.halo_positions[snap_ind, hind]
        self.redshift = self.redshifts[snap_ind]
        if hasattr(self, 'particle_masses'):
            particle_mass = self.particle_masses[snap_ind]
            self.masses_orb = particle_mass
            self.masses_inf = particle_mass

        self.ids_orb = sdata['IDs'][orb_slice]
        if snapshot_data is None:
            self.coords_orb = sdata['Coordinates'][orb_slice]
            self.vels_orb = sdata['Velocities'][orb_slice]
            if 'Masses' in sdata:
                self.masses_orb = sdata['Masses'][orb_slice]
        else:
            self.inds_orb = myin1d(snapshot_data.ids, self.ids_orb)
            self.coords_orb = snapshot_data.coordinates[self.inds_orb] - \
                self.halo_position
            self.vels_orb = snapshot_data.velocities[self.inds_orb]
            if isinstance(snapshot_data.masses, np.ndarray):
                self.masses_orb = snapshot_data.masses[self.inds_orb]
            else:
                self.masses_orb = snapshot_data.masses

        self.ids_inf = sdata['IDs'][inf_slice]
        if snapshot_data is None:
            self.coords_inf = sdata['Coordinates'][inf_slice]
            self.vels_inf = sdata['Velocities'][inf_slice]
            if 'Masses' in sdata:
                self.masses_inf = sdata['Masses'][inf_slice]
        else:
            self.inds_inf = myin1d(snapshot_data.ids, self.ids_inf)
            self.coords_inf = snapshot_data.coordinates[self.inds_inf] - \
                self.halo_position
            self.vels_inf = snapshot_data.velocities[self.inds_inf]
            if isinstance(snapshot_data.masses, np.ndarray):
                self.masses_inf = snapshot_data.masses[self.inds_inf]
            else:
                self.masses_inf = snapshot_data.masses

        angle_condition = 0.0 if angle_condition is None else angle_condition
        if not use_corrected and (
                angle_condition > 0.0 or filtered_ids is not None):
            self.raw_counts = sdata['Counts'][count_slice]
            self.counts = self.correct_counts(
                halo_index, angle_condition, snapshot_number,
                filtered_ids=filtered_ids)
        elif use_corrected:
            self.raw_counts = sdata['Counts'][count_slice]
            self.counts = sdata['CorrectedCounts'][count_slice]
        else:
            self.counts = sdata['Counts'][count_slice]

        inds_remove = np.argwhere(self.counts == 0).flatten()
        if len(inds_remove) != 0:
            self.ids_inf = np.append(self.ids_inf, self.ids_orb[inds_remove])
            self.coords_inf = np.append(
                self.coords_inf, self.coords_orb[inds_remove], axis=0)
            self.vels_inf = np.append(
                self.vels_inf, self.vels_orb[inds_remove], axis=0)
            if isinstance(self.masses_inf, np.ndarray):
                self.masses_inf = np.append(
                    self.masses_inf, self.masses_orb[inds_remove])

            self.ids_orb = np.delete(self.ids_orb, inds_remove)
            self.coords_orb = np.delete(self.coords_orb, inds_remove, axis=0)
            self.vels_orb = np.delete(self.vels_orb, inds_remove, axis=0)
            if isinstance(self.masses_orb, np.ndarray):
                self.masses_orb = np.delete(self.masses_orb, inds_remove)
            self.counts = np.delete(self.counts, inds_remove)

    def plot_position_space(self, projection='xy', colormap='inferno_r',
                            counts_to_plot='all', xlabel=r'$x/R_{200}$',
                            ylabel=r'$y/R_{200}$', display=False,
                            savefile=None):

        if counts_to_plot is 'all':
            counts_to_plot = np.unique(self.counts)

        clrmap = mpl.colormaps[colormap]

        fig, axs = plt.subplots(
            ncols=4, figsize=(22.5, 7), gridspec_kw={
                'width_ratios': [20, 20, 20, 1]})
        cbar_ax = axs[3]
        norm = plt.Normalize(vmin=1, vmax=np.max(counts_to_plot))
        mpl.colorbar.ColorbarBase(
            cbar_ax, cmap=clrmap, norm=norm, orientation='vertical')
        cbar_ax.set_title(r'$N_{\rm orbits}$', fontsize=18)
        cbar_ax.tick_params(labelsize=18)
        cbar_ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        if projection == 'xy':
            proj = [0, 1]
        elif projection == 'yz':
            proj = [1, 2]
        elif projection == 'xz':
            proj = [0, 2]
        else:
            raise ValueError(
                "'projection' must be either 'xy', 'yz', or 'xz'.")

        # orbiting & infalling
        for ii in [0, 2]:
            axs[ii].scatter(self.coords_inf[:, proj[0]]/self.halo_radius,
                            self.coords_inf[:, proj[1]]/self.halo_radius,
                            color='grey', alpha=0.4, marker='.', s=0.2)
        for ii in [0, 1]:
            for n in counts_to_plot[counts_to_plot > 0]:
                norb = np.where(self.counts == n)[0]
                axs[ii].scatter(self.coords_orb[norb][:, proj[0]] /
                                self.halo_radius,
                                self.coords_orb[norb][:, proj[1]] /
                                self.halo_radius,
                                color=clrmap(norm(n-1)), alpha=0.4, marker='.',
                                s=0.2)

        lim = max(np.abs(np.array(list(axs[0].get_xlim()) +
                                  list(axs[0].get_ylim()))))
        titles = ['orbiting + infalling', 'orbiting', 'infalling']
        for ax, title in zip(axs, titles):
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xlabel(xlabel, fontsize=18)
            ax.set_ylabel(ylabel, fontsize=18)
            ax.tick_params(labelsize=18)
            ax.set_title(title, fontsize=18)
        fig.tight_layout()

        if savefile is not None:
            fig.savefig(savefile, dpi=300)
        if display:
            plt.show()
        else:
            plt.close(fig)

    def plot_phase_space(self, colormap='inferno_r', counts_to_plot='all',
                         radius_label=r'$r/R_{200}$',
                         radial_velocity_label=r'$v_r\,\,({\rm km\, s}^{-1})$',
                         display=False, savefile=None):

        r_orb = magnitude(self.coords_orb)
        vr_orb = np.einsum('...i,...i', self.vels_orb, self.coords_orb) / r_orb
        r_inf = magnitude(self.coords_inf)
        vr_inf = np.einsum('...i,...i', self.vels_inf, self.coords_inf) / r_inf

        if counts_to_plot is 'all':
            counts_to_plot = np.unique(self.counts)

        clrmap = mpl.colormaps[colormap]

        fig, axs = plt.subplots(
            ncols=4, figsize=(22.5, 7), gridspec_kw={
                'width_ratios': [20, 20, 20, 1]})
        cbar_ax = axs[3]
        norm = plt.Normalize(vmin=1, vmax=np.max(counts_to_plot))
        mpl.colorbar.ColorbarBase(
            cbar_ax, cmap=clrmap, norm=norm, orientation='vertical')
        cbar_ax.set_title(r'$N_{\rm orbits}$', fontsize=18)
        cbar_ax.tick_params(labelsize=18)
        cbar_ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # orbiting & infalling
        for ii in [0, 2]:
            axs[ii].scatter(r_inf/self.halo_radius, vr_inf, color='grey',
                            alpha=0.4, marker='.', s=0.2)
        for ii in [0, 1]:
            for n in counts_to_plot[counts_to_plot > 0]:
                norb = np.where(self.counts == n)[0]
                axs[ii].scatter(r_orb[norb]/self.halo_radius, vr_orb[norb],
                                color=clrmap(norm(n-1)), alpha=0.4, marker='.',
                                s=0.2)

        xlim = max(np.abs(np.array(list(axs[0].get_xlim()))))
        ylim = max(np.abs(np.array(list(axs[0].get_ylim()))))
        titles = ['orbiting + infalling', 'orbiting', 'infalling']
        for ax, title in zip(axs, titles):
            ax.set_xlim(0, xlim)
            ax.set_ylim(-ylim, ylim)
            ax.set_xlabel(radius_label, fontsize=18)
            ax.set_ylabel(radial_velocity_label, fontsize=18)
            ax.tick_params(labelsize=18)
            ax.set_title(title, fontsize=18)
        fig.tight_layout()

        if savefile is not None:
            fig.savefig(savefile, dpi=300)
        if display:
            plt.show()
        else:
            plt.close(fig)
