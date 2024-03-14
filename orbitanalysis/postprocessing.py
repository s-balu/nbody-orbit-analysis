import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from orbitanalysis.utils import myin1d, vector_norm, recenter_coordinates


class OrbitDecomposition:

    def __init__(self, filename):

        self.filename = filename

        with h5py.File(filename, 'r') as datafile:
            self.main_branches = datafile['main_branches'][:]
            self.region_positions = datafile['region_positions'][:]
            self.region_radii = datafile['region_radii'][:]
            self.bulk_velocities = datafile['bulk_velocities'][:]
            self.snapshot_numbers = datafile['snapshot_numbers'][:]
            if 'box_size' in datafile.attrs:
                self.box_size = datafile.attrs['box_size']

    def get_halo_decomposition_at_snapshot(self, halo_id, snapshot_number,
                                           snapshot_data, angle_cut=0.0):

        """
        Get the orbiting and infalling components of a halo at a particular
        snapshot.

        Parameters
        ----------
        halo_id : int
            The ID of the halo.
        snapshot_number : int
            The snapshot number at which to retrieve the halo decompoaition.
        snapshot_data : dict
            A dictionary with the following elements:

            * ids : (N,) ndarray - list of particle IDs that contains at least
                    all those that belong to the halo region at the specified
                    snapshot number.
            * coordinates : (N, 3) ndarray - the corresponding coordinates.
            * velocities : (N, 3) ndarray - the corresponding velocities.
            * masses : (N,) ndarray or float - the corresponding masses, or a
                       single mass value if all particles have the same mass.
        angle_cut : float, optional
            Particles that advance about the halo center by less than this
            angle between peri/apocenters will not have that peri/apocenter
            counted. This is designed to remove spurious peri/apocenters from
            orbits within subhalos.

        """

        hind = np.argwhere(self.main_branches[-1] == halo_id).flatten()[0]
        sind = np.argwhere(
            self.snapshot_numbers == snapshot_number).flatten()[0]

        self.region_position = self.region_positions[sind, hind]
        self.region_radius = self.region_radii[sind, hind]
        self.bulk_velocity = self.bulk_velocities[sind, hind]

        with h5py.File(self.filename, 'r') as datafile:

            ids_orbiting = np.array([])
            ids_infalling = np.array([])
            ids_departed = np.array([])

            for s in self.snapshot_numbers[:sind+1]:

                sdata = datafile['snapshot_{}'.format('%0.3d' % s)]

                hind_ = np.argwhere(
                    sdata['halo_ids_final'] == halo_id).flatten()
                if len(hind_) == 0:
                    continue
                else:
                    hind_ = hind_[0]

                orbiting_lims = sdata['orbiting_offsets'][hind_:hind_+2]
                orbids = sdata['orbiting_IDs'][slice(*orbiting_lims)]
                angles = sdata['angles'][slice(*orbiting_lims)]
                ids_orbiting = np.append(
                    ids_orbiting, orbids[
                        np.argwhere(angles > angle_cut).flatten()])

                infalling_lims = sdata['infalling_offsets'][hind_:hind_+2]
                infids = sdata['infalling_IDs'][slice(*infalling_lims)]
                ids_infalling = np.append(ids_infalling, infids)

                ids_returned = np.intersect1d(ids_departed, infids)
                if len(ids_returned) > 0:
                    inds_returned = np.where(
                        np.in1d(ids_departed, ids_returned))[0]
                    ids_departed = np.delete(ids_departed, inds_returned)

                departed_lims = sdata['departed_offsets'][hind_:hind_+2]
                ids_departed = np.append(
                    ids_departed, sdata['departed_IDs'][slice(*departed_lims)])

        orb_departed_ids = np.intersect1d(ids_orbiting, ids_departed)
        orb_departed_inds = np.where(
            np.in1d(ids_orbiting, orb_departed_ids))[0]
        ids_orbiting = np.delete(ids_orbiting, orb_departed_inds)
        ids_orbiting, counts = np.unique(ids_orbiting, return_counts=True)

        ids_infalling = np.unique(ids_infalling)
        inf_orb_ids = np.intersect1d(ids_infalling, ids_orbiting)
        inf_orb_inds = np.where(np.in1d(ids_infalling, inf_orb_ids))[0]
        ids_infalling = np.delete(ids_infalling, inf_orb_inds)

        inf_departed_ids = np.intersect1d(ids_infalling, ids_departed)
        inf_departed_inds = np.where(
            np.in1d(ids_infalling, inf_departed_ids))[0]
        ids_infalling = np.delete(ids_infalling, inf_departed_inds)

        self.ids_orbiting = np.intersect1d(snapshot_data['ids'], ids_orbiting)
        inds_contained = myin1d(ids_orbiting, self.ids_orbiting)
        self.counts = counts[inds_contained]

        self.ids_infalling = np.intersect1d(
            snapshot_data['ids'], ids_infalling)

        inds_orbiting = myin1d(snapshot_data['ids'], self.ids_orbiting)
        if hasattr(self, "box_size"):
            self.coords_orbiting = recenter_coordinates(
                snapshot_data['coordinates'][inds_orbiting] -
                self.region_position, self.box_size)
        else:
            self.coords_orbiting = snapshot_data['coordinates'][
                inds_orbiting] - self.region_position
        self.vels_orbiting = snapshot_data['velocities'][inds_orbiting] - \
            self.bulk_velocity
        if isinstance(snapshot_data['masses'], np.ndarray):
            self.masses_orbiting = snapshot_data['masses'][inds_orbiting]
        else:
            self.masses_orbiting = snapshot_data['masses']

        inds_infalling = myin1d(snapshot_data['ids'], self.ids_infalling)
        if hasattr(self, "box_size"):
            self.coords_infalling = recenter_coordinates(
                snapshot_data['coordinates'][inds_infalling] -
                self.region_position, self.box_size)
        else:
            self.coords_infalling = snapshot_data['coordinates'][
                inds_infalling] - self.region_position
        self.vels_infalling = snapshot_data['velocities'][inds_infalling] - \
            self.bulk_velocity
        if isinstance(snapshot_data['masses'], np.ndarray):
            self.masses_infalling = snapshot_data['masses'][inds_infalling]
        else:
            self.masses_infalling = snapshot_data['masses']

    def plot_position_space(self, projection='xy', colormap='rainbow_r',
                            counts_to_plot='all', xlabel=r'$x/R_{\rm region}$',
                            ylabel=r'$y/R_{\rm region}$', display=False,
                            savefile=None):

        """
        Plot the spatial distribution of the particles, colored by number of
        orbits.
        """

        if counts_to_plot == 'all':
            counts_to_plot = np.unique(self.counts)

        clrmap = mpl.colormaps[colormap]

        fig, axs = plt.subplots(
            ncols=4, figsize=(22.5, 7), width_ratios=[20, 20, 20, 1])
        cbar_ax = axs[3]
        max_n = np.max(counts_to_plot)
        norm = mpl.colors.LogNorm(vmin=1, vmax=max_n)
        mpl.colorbar.ColorbarBase(
            cbar_ax, cmap=clrmap, norm=norm, orientation='vertical')
        cbar_ax.set_title(r'$N_{\rm orbits}$', fontsize=18)
        cbar_ax.tick_params()
        ticks = np.unique(
            (10**np.linspace(np.log10(1), np.log10(max_n), 16)).astype(int))
        cbar_ax.set_yticks(ticks)
        cbar_ax.set_yticklabels(
            [r'${}$'.format(x) for x in ticks], fontsize=18)

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
            axs[ii].scatter(self.coords_infalling[:, proj[0]] /
                            self.region_radius,
                            self.coords_infalling[:, proj[1]] /
                            self.region_radius,
                            color='grey', alpha=0.4, marker='.', s=0.2)
        for ii in [0, 1]:
            for n in counts_to_plot[counts_to_plot > 0]:
                norb = np.where(self.counts == n)[0]
                axs[ii].scatter(self.coords_orbiting[norb][:, proj[0]] /
                                self.region_radius,
                                self.coords_orbiting[norb][:, proj[1]] /
                                self.region_radius,
                                color=clrmap(norm(n)), alpha=0.4, marker='.',
                                s=0.2)

        lim = max(np.abs(np.array(list(axs[0].get_xlim()) +
                                  list(axs[0].get_ylim()))))
        titles = ['orbiting + infalling', 'orbiting', 'infalling']
        for jj, (ax, title) in enumerate(zip(axs, titles)):
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xlabel(xlabel, fontsize=18)
            ax.tick_params(labelsize=18)
            ax.set_title(title, fontsize=18)
            if jj == 0:
                ax.set_ylabel(ylabel, fontsize=18)
            else:
                ax.set_yticklabels([])
        fig.subplots_adjust(wspace=0)
        if savefile is not None:
            fig.savefig(savefile, bbox_inches='tight', dpi=300)
        if display:
            plt.show()
        else:
            plt.close(fig)

    def plot_phase_space(self, colormap='rainbow_r', counts_to_plot='all',
                         radius_label=r'$r/R_{\rm region}$',
                         radial_velocity_label=r'$v_r\,\,({\rm km\, s}^{-1})$',
                         logr=True, display=False, savefile=None):

        """
        Plot the phase space distribution of the particles, colored by number
        of orbits.
        """

        r_orb = vector_norm(self.coords_orbiting)
        vr_orb = np.einsum(
            '...i,...i', self.vels_orbiting, self.coords_orbiting) / r_orb
        r_inf = vector_norm(self.coords_infalling)
        vr_inf = np.einsum(
            '...i,...i', self.vels_infalling, self.coords_infalling) / r_inf

        if counts_to_plot == 'all':
            counts_to_plot = np.unique(self.counts)

        clrmap = mpl.colormaps[colormap]

        fig, axs = plt.subplots(
            ncols=4, figsize=(22.5, 7), width_ratios=[20, 20, 20, 1])
        cbar_ax = axs[3]
        max_n = np.max(counts_to_plot)
        norm = mpl.colors.LogNorm(vmin=1, vmax=max_n)
        mpl.colorbar.ColorbarBase(
            cbar_ax, cmap=clrmap, norm=norm, orientation='vertical')
        cbar_ax.set_title(r'$N_{\rm orbits}$', fontsize=18)
        cbar_ax.tick_params()
        ticks = np.unique(
            (10**np.linspace(np.log10(1), np.log10(max_n), 16)).astype(int))
        cbar_ax.set_yticks(ticks)
        cbar_ax.set_yticklabels(
            [r'${}$'.format(x) for x in ticks], fontsize=18)

        # orbiting & infalling
        for ii in [0, 2]:
            axs[ii].scatter(r_inf/self.region_radius, vr_inf, color='grey',
                            alpha=0.4, marker='.', s=0.2)
        for ii in [0, 1]:
            for n in counts_to_plot[counts_to_plot > 0]:
                norb = np.where(self.counts == n)[0]
                axs[ii].scatter(r_orb[norb]/self.region_radius, vr_orb[norb],
                                color=clrmap(norm(n)), alpha=0.4, marker='.',
                                s=0.2)

        xlims = (min(np.abs(np.array(list(axs[0].get_xlim())))),
                 max(np.abs(np.array(list(axs[0].get_xlim())))))
        ylim = max(np.abs(np.array(list(axs[0].get_ylim()))))
        titles = ['orbiting + infalling', 'orbiting', 'infalling']
        for jj, (ax, title) in enumerate(zip(axs, titles)):
            if logr:
                ax.set_xlim(*xlims)
                ax.set_xscale('log')
            else:
                ax.set_xlim(0, xlims[1])
            ax.set_ylim(-ylim, ylim)
            ax.set_xlabel(radius_label, fontsize=18)
            ax.tick_params(labelsize=18)
            ax.set_title(title, fontsize=18)
            if jj == 0:
                ax.set_ylabel(radial_velocity_label, fontsize=18)
            else:
                ax.set_yticklabels([])
        fig.subplots_adjust(wspace=0)
        if savefile is not None:
            fig.savefig(savefile, bbox_inches='tight', dpi=300)
        if display:
            plt.show()
        else:
            plt.close(fig)
