import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt

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

    def collate_orbits(self, halo_ids, snapshot_number, angle_cut=np.pi/2,
                       savefile=None):

        """
        Collate the peri/apocenter information to obtain the complete set of
        IDs for the orbiting and infalling particles, subject to an angle cut.
        If a `savefile` is specified, the orbit information for each snapshot
        will be saved on the fly.

        Parameters
        ----------
        halo_ids : array_like
            The IDs of the halos.
        snapshot_number : int
            The snapshot number at which to retrieve the halo decompoaition.
        angle_cut : float, optional
            Particles that advance about the halo center by less than this
            angle between peri/apocenters will not have that peri/apocenter
            counted. This is designed to remove spurious peri/apocenters from
            orbits within subhalos.
        savefile : str, optional
            The HDF5 filename at which to save the orbit information.

        """

        sind = np.argwhere(
            self.snapshot_numbers == snapshot_number).flatten()[0]

        def correct_ids(oids, iids, dids):

            orb_departed_ids = np.intersect1d(oids, dids)
            orb_departed_inds = np.where(
                np.in1d(oids, orb_departed_ids))[0]
            oids = np.delete(oids, orb_departed_inds)
            oids, c = np.unique(oids, return_counts=True)

            iids = np.unique(iids)
            inf_orb_ids = np.intersect1d(iids, oids)
            inf_orb_inds = np.where(np.in1d(iids, inf_orb_ids))[0]
            iids = np.delete(iids, inf_orb_inds)

            inf_departed_ids = np.intersect1d(iids, dids)
            inf_departed_inds = np.where(
                np.in1d(iids, inf_departed_ids))[0]
            iids = np.delete(iids, inf_departed_inds)

            return oids, c, iids

        ids_orbiting = np.array([], dtype=int)
        ids_infalling = np.array([], dtype=int)
        ids_departed = [np.array([], dtype=int) for _ in halo_ids]
        inds_orbiting = [np.array([], dtype=int) for _ in halo_ids]
        inds_infalling = [np.array([], dtype=int) for _ in halo_ids]

        for s in self.snapshot_numbers[:sind+1]:

            with h5py.File(self.filename, 'r') as datafile:

                sdata = datafile['snapshot_{}'.format('%0.3d' % s)]
                hinds1 = np.where(
                    np.in1d(sdata['halo_ids_final'][:], halo_ids))[0]
                if len(hinds1) == 0:
                    continue
                hids = sdata['halo_ids_final'][hinds1]
                hinds2 = myin1d(halo_ids, hids)
                hinds1 = hinds1[np.argsort(hinds2)]
                hinds2 = np.sort(hinds2)

                orbiting_lims = list(
                    zip(sdata['orbiting_offsets'][hinds1],
                        sdata['orbiting_offsets'][hinds1+1]))
                orbids = np.concatenate(
                    [sdata['orbiting_IDs'][slice(*lims)] for lims in
                     orbiting_lims])
                angles = np.concatenate(
                    [sdata['angles'][slice(*lims)] for lims in orbiting_lims])

                new_orb_lens = [0] + [
                    np.diff(lims)[0] for lims in orbiting_lims]
                new_orb_offsets = np.cumsum(new_orb_lens)
                new_orb_lims = list(
                    zip(new_orb_offsets[:-1], new_orb_offsets[1:]))

                angle_bool = np.zeros(len(angles), dtype=bool)
                angle_inds = np.argwhere(angles > angle_cut).flatten()
                angle_bool[angle_inds] = True
                angle_cut_lens = [0] + [
                    np.sum(angle_bool[slice(*lims)]) for lims in new_orb_lims]
                angle_cut_offsets = np.cumsum(angle_cut_lens)
                angle_cut_lims = np.array(list(
                    zip(angle_cut_offsets[:-1], angle_cut_offsets[1:])))
                N = len(ids_orbiting)
                for hind, lim in zip(hinds2, angle_cut_lims):
                    inds_orbiting[hind] = np.append(
                        inds_orbiting[hind], np.arange(*(lim+N)))
                ids_orbiting = np.append(ids_orbiting, orbids[angle_inds])

                infalling_lims = list(
                    zip(sdata['infalling_offsets'][hinds1],
                        sdata['infalling_offsets'][hinds1+1]))
                infids = np.concatenate(
                    [sdata['infalling_IDs'][slice(*lims)] for lims in
                     infalling_lims])

                new_inf_lens = [0] + [
                    np.diff(lims)[0] for lims in infalling_lims]
                new_inf_offsets = np.cumsum(new_inf_lens)
                new_inf_lims = np.array(list(
                    zip(new_inf_offsets[:-1], new_inf_offsets[1:])))
                N = len(ids_infalling)
                for hind, lim in zip(hinds2, new_inf_lims):
                    inds_infalling[hind] = np.append(
                        inds_infalling[hind], np.arange(*(lim+N)))
                ids_infalling = np.append(ids_infalling, infids)

                for hind1, hind2, ilims in zip(hinds1, hinds2, new_inf_lims):
                    ids_returned = np.intersect1d(
                        ids_departed[hind2], infids[slice(*ilims)])
                    if len(ids_returned) > 0:
                        inds_returned = np.where(
                            np.in1d(ids_departed[hind2], ids_returned))[0]
                        ids_departed[hind2] = np.delete(
                            ids_departed[hind2], inds_returned)

                    departed_lims = sdata['departed_offsets'][hind1:hind1+2]
                    ids_departed[hind2] = np.append(
                        ids_departed[hind2], sdata['departed_IDs'][
                            slice(*departed_lims)])

            if savefile is not None:

                ids_orbiting_, counts, ids_infalling_ = [], [], []
                for oinds, iinds, hind in zip(
                        inds_orbiting, inds_infalling, hinds2):
                    oids_, c_, iids_ = correct_ids(
                        ids_orbiting[oinds], ids_infalling[iinds],
                        ids_departed[hind])
                    ids_orbiting_.append(oids_)
                    counts.append(c_)
                    ids_infalling_.append(iids_)
                orbiting_offsets = np.cumsum(
                    [0] + [len(ids) for ids in ids_orbiting_])
                infalling_offsets = np.cumsum(
                    [0] + [len(ids) for ids in ids_infalling_])
                with h5py.File(savefile, 'a') as hf:
                    hfsnap = hf.create_group('snapshot_{}'.format('%03d' % s))
                    hfsnap.create_dataset(
                        'orbiting_IDs', data=np.concatenate(ids_orbiting_))
                    hfsnap.create_dataset(
                        'infalling_IDs', data=np.concatenate(ids_infalling_))
                    hfsnap.create_dataset(
                        'orbit_counts', data=np.concatenate(counts))
                    hfsnap.create_dataset(
                        'orbiting_offsets', data=orbiting_offsets)
                    hfsnap.create_dataset(
                        'infalling_offsets', data=infalling_offsets)

        if savefile is None:

            ids_orbiting_, counts, ids_infalling_ = [], [], []
            for oinds, iinds, hind in zip(
                    inds_orbiting, inds_infalling, hinds2):
                oids_, c_, iids_ = correct_ids(
                    ids_orbiting[oinds], ids_infalling[iinds],
                    ids_departed[hind])
                ids_orbiting_.append(oids_)
                counts.append(c_)
                ids_infalling_.append(iids_)
        else:
            with h5py.File(savefile, 'a') as hf:
                hf.create_dataset('halo_IDs', data=halo_ids)

        self.orbiting_offsets = np.cumsum(
            [0] + [len(ids) for ids in ids_orbiting_])
        self.infalling_offsets = np.cumsum(
            [0] + [len(ids) for ids in ids_infalling_])
        self.ids_orbiting = np.concatenate(ids_orbiting_)
        self.ids_infalling = np.concatenate(ids_infalling_)
        self.counts = np.concatenate(counts)
        self.halo_ids_collated = halo_ids
        self.snapshot_number_collated = snapshot_number

        return

    def get_particle_data_at_snapshot(self, snapshot_data, halo_ids=None,
                                      snapshot_number=None, angle_cut=0.0,
                                      readfile=None):

        """
        Get the orbiting and infalling components of a halo at a particular
        snapshot.

        Parameters
        ----------
        snapshot_data : dict
            A dictionary with the following elements:

            * ids : (N,) ndarray - list of particle IDs that contains at least
                    all those that belong to the halo region at the specified
                    snapshot number.
            * coordinates : (N, 3) ndarray - the corresponding coordinates.
            * velocities : (N, 3) ndarray - the corresponding velocities.
            * masses : (N,) ndarray or float - the corresponding masses, or a
                       single mass value if all particles have the same mass.
        halo_ids : array_like, optional if `collate_orbits` has been run or if
                   reading from a file.
            The IDs of the halos.
        snapshot_number : int, optional if `collate_orbits` has been run
            The snapshot number at which to retrieve the halo decomposition.
        angle_cut : float, optional
            Particles that advance about the halo center by less than this
            angle between peri/apocenters will not have that peri/apocenter
            counted. This is designed to remove spurious peri/apocenters from
            orbits within subhalos.
        readfile : str, optional
            Read the orbit information from a file generated by
            `collate_orbits`.

        """

        if halo_ids is not None:
            halo_ids = np.atleast_1d(halo_ids)

        if not hasattr(self, "ids_orbiting") and readfile is None:
            self.collate_orbits(halo_ids, snapshot_number, angle_cut)
        elif readfile is not None:
            with h5py.File(readfile, 'r') as hf:
                hfsnap = hf['snapshot_{}'.format('%03d' % snapshot_number)]
                self.ids_orbiting = hfsnap['orbiting_IDs'][:]
                self.ids_infalling = hfsnap['infalling_IDs'][:]
                self.counts = hfsnap['orbit_counts'][:]
                self.orbiting_offsets = hfsnap['orbiting_offsets'][:]
                self.infalling_offsets = hfsnap['infalling_offsets'][:]
                self.halo_ids_collated = hf['halo_IDs'][:]
                self.snapshot_number_collated = snapshot_number
        orbiting_lims = np.array(list(
            zip(self.orbiting_offsets[:-1], self.orbiting_offsets[1:])))
        infalling_lims = np.array(list(
            zip(self.infalling_offsets[:-1], self.infalling_offsets[1:])))

        if halo_ids is not None:
            hinds1 = myin1d(self.halo_ids_collated, halo_ids)
            orbiting_lims = orbiting_lims[hinds1]
            infalling_lims = infalling_lims[hinds1]
            hinds2 = myin1d(self.main_branches[-1], halo_ids)
        else:
            hinds2 = myin1d(self.main_branches[-1], self.halo_ids_collated)
        sind = np.argwhere(
            self.snapshot_numbers == self.snapshot_number_collated).flatten()[
            0]
        region_positions = self.region_positions[sind, hinds2]
        bulk_velocities = self.bulk_velocities[sind, hinds2]

        n_orb, n_inf = len(self.ids_orbiting), len(self.ids_infalling)
        ids_orbiting = np.empty(n_orb)
        ids_infalling = np.empty(n_inf)
        coords_orbiting = np.empty((n_orb, 3))
        coords_infalling = np.empty((n_inf, 3))
        vels_orbiting = np.empty((n_orb, 3))
        vels_infalling = np.empty((n_inf, 3))
        if isinstance(snapshot_data['masses'], np.ndarray):
            masses_orbiting = np.empty(n_orb)
            masses_infalling = np.empty(n_inf)
            mass_is_array = True
        else:
            mass_is_array = False
        counts = np.empty(n_orb)
        ooff, ioff = 0, 0
        self.orbiting_slices, self.infalling_slices = [], []
        for olim, ilim, region_position, bulk_velocity in zip(
                orbiting_lims, infalling_lims, region_positions,
                bulk_velocities):
            olen, ilen = np.diff(olim)[0], np.diff(ilim)[0]
            osl, isl = slice(*olim), slice(*ilim)
            osl_, isl_ = slice(ooff, ooff+olen), slice(ioff, ioff+ilen)
            self.orbiting_slices.append(osl_)
            self.infalling_slices.append(isl_)
            orbids = np.intersect1d(
                snapshot_data['ids'], self.ids_orbiting[osl])
            inds_contained = myin1d(self.ids_orbiting[osl], orbids)
            ids_orbiting[osl_] = orbids
            counts[osl_] = self.counts[osl][inds_contained]

            infids = np.intersect1d(
                snapshot_data['ids'], self.ids_infalling[isl])
            ids_infalling[isl_] = infids

            inds_orbiting = myin1d(snapshot_data['ids'], orbids)
            if hasattr(self, "box_size"):
                coords_orbiting[osl_] = recenter_coordinates(
                    snapshot_data['coordinates'][inds_orbiting] -
                    region_position, self.box_size)
            else:
                coords_orbiting[osl_] = snapshot_data['coordinates'][
                    inds_orbiting] - region_position
            vels_orbiting[osl_] = snapshot_data['velocities'][inds_orbiting] -\
                bulk_velocity
            if mass_is_array:
                masses_orbiting[osl_] = snapshot_data['masses'][inds_orbiting]
            else:
                masses_orbiting = snapshot_data['masses']

            inds_infalling = myin1d(snapshot_data['ids'], infids)
            if hasattr(self, "box_size"):
                coords_infalling[isl_] = recenter_coordinates(
                    snapshot_data['coordinates'][inds_infalling] -
                    region_position, self.box_size)
            else:
                coords_infalling[isl_] = snapshot_data['coordinates'][
                    inds_infalling] - region_position
            vels_infalling[isl_] = snapshot_data['velocities'][inds_infalling]\
                - bulk_velocity
            if mass_is_array:
                masses_infalling[isl_] = snapshot_data['masses'][
                    inds_infalling]
            else:
                masses_infalling = snapshot_data['masses']

            ooff += olen
            ioff += ilen

        self.ids_orbiting = ids_orbiting[:ooff]
        self.ids_infalling = ids_infalling[:ioff]
        self.coords_orbiting = coords_orbiting[:ooff]
        self.coords_infalling = coords_infalling[:ioff]
        self.vels_orbiting = vels_orbiting[:ooff]
        self.vels_infalling = vels_infalling[:ioff]
        if mass_is_array:
            self.masses_orbiting = masses_orbiting[:ooff]
            self.masses_infalling = masses_infalling[:ioff]
        self.counts = counts[:ooff]
        self.halo_ids_partdata = halo_ids if halo_ids is not None else \
            self.halo_ids_collated

        return

    def plot_position_space(self, halo_id, projection='xy',
                            colormap='rainbow_r', counts_to_plot='all',
                            xlabel=r'$x/R_{\rm region}$',
                            ylabel=r'$y/R_{\rm region}$', display=False,
                            savefile=None):

        """
        Plot the spatial distribution of the particles, colored by number of
        orbits.
        """

        ind = np.argwhere(self.halo_ids_partdata == halo_id).flatten()
        if len(ind) == 0:
            raise ValueError('The halo corresponding to the provided ID has '
                             'not been post-processed!')
        else:
            ind = ind[0]
        osl, isl = self.orbiting_slices[ind], self.infalling_slices[ind]
        coords_orbiting = self.coords_orbiting[osl]
        counts = self.counts[osl]
        coords_infalling = self.coords_infalling[isl]

        hind = np.argwhere(self.main_branches[-1] == halo_id).flatten()[0]
        sind = np.argwhere(
            self.snapshot_numbers == self.snapshot_number_collated).flatten()[
            0]
        region_radius = self.region_radii[sind, hind]

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
            axs[ii].scatter(coords_infalling[:, proj[0]] /
                            region_radius,
                            coords_infalling[:, proj[1]] /
                            region_radius,
                            color='grey', alpha=0.4, marker='.', s=0.2)
        for ii in [0, 1]:
            for n in counts_to_plot[counts_to_plot > 0]:
                norb = np.where(counts == n)[0]
                axs[ii].scatter(coords_orbiting[norb][:, proj[0]] /
                                region_radius,
                                coords_orbiting[norb][:, proj[1]] /
                                region_radius,
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

    def plot_phase_space(self, halo_id, colormap='rainbow_r',
                         counts_to_plot='all',
                         radius_label=r'$r/R_{\rm region}$',
                         radial_velocity_label=r'$v_r\,\,({\rm km\, s}^{-1})$',
                         logr=True, display=False, savefile=None):

        """
        Plot the phase space distribution of the particles, colored by number
        of orbits.
        """

        ind = np.argwhere(self.halo_ids_partdata == halo_id).flatten()
        if len(ind) == 0:
            raise ValueError('The halo corresponding to the provided ID has '
                             'not been post-processed!')
        else:
            ind = ind[0]
        osl, isl = self.orbiting_slices[ind], self.infalling_slices[ind]
        coords_orbiting = self.coords_orbiting[osl]
        vels_orbiting = self.vels_orbiting[osl]
        counts = self.counts[osl]
        coords_infalling = self.coords_infalling[isl]
        vels_infalling = self.vels_infalling[isl]

        hind = np.argwhere(self.main_branches[-1] == halo_id).flatten()[0]
        sind = np.argwhere(
            self.snapshot_numbers == self.snapshot_number_collated).flatten()[
            0]
        region_radius = self.region_radii[sind, hind]

        r_orb = vector_norm(coords_orbiting)
        vr_orb = np.einsum(
            '...i,...i', vels_orbiting, coords_orbiting) / r_orb
        r_inf = vector_norm(coords_infalling)
        vr_inf = np.einsum(
            '...i,...i', vels_infalling, coords_infalling) / r_inf

        if counts_to_plot == 'all':
            counts_to_plot = np.unique(counts)

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
            axs[ii].scatter(r_inf/region_radius, vr_inf, color='grey',
                            alpha=0.4, marker='.', s=0.2)
        for ii in [0, 1]:
            for n in counts_to_plot[counts_to_plot > 0]:
                norb = np.where(counts == n)[0]
                axs[ii].scatter(r_orb[norb]/region_radius, vr_orb[norb],
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
