import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

from orbitanalysis.utils import myin1d, vector_norm, recenter_coordinates


class OrbitDecomposition:

    def __init__(self, filename):

        self.filename = filename

        with h5py.File(filename, 'r') as datafile:

            self.main_branches = []
            self.region_positions = []
            self.region_radii = []
            self.bulk_velocities = []
            self.snapshot_numbers = []

            for skey in list(datafile.keys()):

                ds = datafile[skey]
                self.main_branches.append(ds['halo_ids'][:])
                self.region_positions.append(ds['region_positions'][:])
                self.region_radii.append(ds['region_radii'][:])
                self.bulk_velocities.append(ds['bulk_velocities'][:])
                self.snapshot_numbers.append(int(skey.split('_')[1]))
            
            self.main_branches = np.array(self.main_branches)
            self.region_positions = np.array(self.region_positions)
            self.region_radii = np.array(self.region_radii)
            self.bulk_velocities = np.array(self.bulk_velocities)
            self.snapshot_numbers = np.array(self.snapshot_numbers)

            if 'box_size' in datafile.attrs:
                self.box_size = datafile.attrs['box_size']

    def collate_orbits(self, halo_ids=None, snapshot_number=None,
                       angle_cut=np.pi/2, save_final_counts=False,
                       data_type=None, savefile=None, verbose=True):

        """
        Collate the peri/apocenter information to obtain the complete set of
        IDs for the orbiting and infalling particles, subject to an angle cut.
        If a `savefile` is specified, the orbit information for each snapshot
        will be saved on the fly.

        Parameters
        ----------
        halo_ids : array_like, optional
            The IDs of the halos at redshift 0. If left unspecified, all halos
            will be collated.
        snapshot_number : int, optional
            The snapshot number at which to retrieve the orbit decomposition.
        angle_cut : float, optional
            Particles that advance about the halo center by less than this
            angle between peri/apocenters will not have that peri/apocenter
            counted. This is designed to remove spurious peri/apocenters from
            orbits within subhalos.
        save_final_counts: bool, optional
            Additionally save the orbit counts that the particles at each
            snapshot will have by the time of the final snapshot.
        data_type : type, optional
            save the IDs in this data type. Defaults to the data type used by
            the snapshot data.
        savefile : str, optional
            The HDF5 filename at which to save the orbit information.
        verbose : bool, optional
            Print status.

        """

        if verbose:
            t_start = time.time()

        if halo_ids is None:
            halo_ids = self.main_branches[-1]
        else:
            if len(np.intersect1d(self.main_branches[-1], halo_ids)) < len(
                    halo_ids):
                self.missing_halo_ids = np.setdiff1d(
                    halo_ids, self.main_branches[-1])
                raise ValueError(
                    "The input halo ID list contains IDs of halos (at z=0) "
                    "that have not been processed. Refer to the final row of " 
                    "the `main_branches` attribute to see all IDs (at z=0) "
                    "that have been processed.")

        if snapshot_number is None:
            sind = len(self.snapshot_numbers) - 1
        else:
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

            return oids, c.astype(np.uint16), iids

        ids_orbiting = np.array([], dtype=np.uint64)
        ids_infalling = np.array([], dtype=np.uint64)
        ids_departed = [np.array([], dtype=np.uint64) for _ in halo_ids]
        inds_orbiting = [np.array([], dtype=np.uint64) for _ in halo_ids]
        inds_infalling = [np.array([], dtype=np.uint64) for _ in halo_ids]

        init_orb, init_inf, init_dep = False, False, False

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
                hids = hids[np.argsort(hinds2)]
                hinds2 = np.sort(hinds2)

                has_orbiting, has_infalling, has_departed = True, True, True
                if len(sdata['orbiting_IDs']) == 0:
                    has_orbiting = False
                if len(sdata['infalling_IDs']) == 0:
                    has_infalling = False
                if len(sdata['departed_IDs']) == 0:
                    has_departed = False

                if has_orbiting:

                    if not init_orb:
                        if data_type is None:
                            orbtype = sdata['orbiting_IDs'].dtype
                        else:
                            orbtype = data_type
                        ids_orbiting = np.array([], dtype=orbtype)
                        inds_orbiting = [
                            np.array([], dtype=orbtype) for _ in halo_ids]
                        init_orb = True

                    orbiting_lims = list(
                        zip(sdata['orbiting_offsets'][hinds1],
                            sdata['orbiting_offsets'][hinds1+1]))
                    orbids = np.concatenate(
                        [sdata['orbiting_IDs'][slice(*lims)] for lims in
                         orbiting_lims])

                    new_orb_lens = [0] + [
                        np.diff(lims)[0] for lims in orbiting_lims]
                    new_orb_offsets = np.cumsum(new_orb_lens)
                    new_orb_lims = list(
                        zip(new_orb_offsets[:-1], new_orb_offsets[1:]))

                    angles = np.concatenate(
                        [sdata['angles'][slice(*lims)] for lims in 
                         orbiting_lims])
                    angle_bool = np.zeros(len(angles), dtype=bool)
                    angle_inds = np.argwhere(angles > angle_cut).flatten()
                    angle_bool[angle_inds] = True
                    angle_cut_lens = [0] + [
                        np.sum(angle_bool[slice(*lims)]) for lims in 
                        new_orb_lims]
                    angle_cut_offsets = np.cumsum(angle_cut_lens)
                    angle_cut_lims = np.array(list(
                        zip(angle_cut_offsets[:-1], angle_cut_offsets[1:])))
                    N = len(ids_orbiting)
                    for hind, lim in zip(hinds2, angle_cut_lims):
                        inds_orbiting[hind] = np.append(
                            inds_orbiting[hind], np.arange(
                                *(lim+N), dtype=orbtype))
                    ids_orbiting = np.append(ids_orbiting, orbids[angle_inds])

                if has_infalling:

                    if not init_inf:
                        if data_type is None:
                            inftype = sdata['infalling_IDs'].dtype
                        else:
                            inftype = data_type
                        ids_infalling = np.array([], dtype=inftype)
                        inds_infalling = [
                            np.array([], dtype=inftype) for _ in halo_ids]
                        init_inf = True

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
                            inds_infalling[hind], np.arange(
                                *(lim+N), dtype=inftype))
                    ids_infalling = np.append(ids_infalling, infids)

                if has_departed:

                    if not init_dep:
                        if data_type is None:
                            deptype = sdata['departed_IDs'].dtype
                        else:
                            deptype = data_type
                        ids_departed = [
                            np.array([], dtype=deptype) for _ in halo_ids]
                        init_dep = True

                    for hind1, hind2, ilims in zip(
                            hinds1, hinds2, new_inf_lims):
                        ids_returned = np.intersect1d(
                            ids_departed[hind2], infids[slice(*ilims)])
                        if len(ids_returned) > 0:
                            inds_returned = np.where(
                                np.in1d(ids_departed[hind2], ids_returned))[0]
                            ids_departed[hind2] = np.delete(
                                ids_departed[hind2], inds_returned)

                        departed_lims = sdata['departed_offsets'][
                            hind1:hind1+2]
                        ids_departed[hind2] = np.append(
                            ids_departed[hind2], sdata['departed_IDs'][
                                slice(*departed_lims)])

            if savefile is not None:

                ids_, counts = [], []
                for oinds, iinds, dids in zip(
                        inds_orbiting, inds_infalling, ids_departed):
                    oids_, c_, iids_ = correct_ids(
                        ids_orbiting[oinds], ids_infalling[iinds], dids)
                    ids_.append(np.concatenate([oids_, iids_]))
                    counts.append(np.concatenate(
                        [c_, np.zeros(len(iids_), dtype=np.uint16)]))
                offsets = np.cumsum([0] + [len(x) for x in ids_])
                with h5py.File(savefile, 'a') as hf:
                    hfsnap = hf.create_group('snapshot_{}'.format('%03d' % s))
                    hfsnap.create_dataset(
                        'particle_IDs', data=np.concatenate(ids_))
                    hfsnap.create_dataset(
                        'orbit_counts', data=np.concatenate(counts))
                    hfsnap.create_dataset('halo_offsets', data=offsets)

                    hfsnap.create_dataset('halo_IDs_final', data=hids)

            if verbose:
                print('Snapshot {} collated'.format('%03d' % s))

        if savefile is None:

            ids_orbiting_, counts, ids_infalling_ = [], [], []
            for oinds, iinds, dids in zip(
                    inds_orbiting, inds_infalling, ids_departed):
                oids_, c_, iids_ = correct_ids(
                    ids_orbiting[oinds], ids_infalling[iinds], dids)
                ids_orbiting_.append(oids_)
                counts.append(c_)
                ids_infalling_.append(iids_)
        else:
            with h5py.File(savefile, 'a') as hf:
                hf.create_dataset('halo_IDs', data=halo_ids)

            if save_final_counts:
                self.save_final_orbit_counts(savefile, verbose=verbose)

        self.halo_offsets = np.cumsum([0] + [len(x) for x in ids_])
        self.particle_ids = np.concatenate(ids_)
        self.orbit_counts = np.concatenate(counts)
        self.halo_ids_collated = halo_ids
        self.snapshot_number_collated = self.snapshot_numbers[sind]

        if verbose:
            print('Orbits collated in {} s'.format(
                round(time.time()-t_start, 3)))

        return

    def get_orbit_decomposition_at_snapshot(self, snapshot_data,
                                            snapshot_number=None,
                                            halo_ids=None, angle_cut=0.0,
                                            collated_file=None, verbose=True):
        
        """
        Get the particle data and orbit counts for a set of halos at a
        particular snapshot.

        Parameters
        ----------
        snapshot_data : dict
            A dictionary with the following elements:

            * ids : (N,) ndarray - list of particle IDs that contains at least
                    all those that belong to the halo region at the specified
                    snapshot number.
            * coordinates : (N, 3) ndarray, optional - the corresponding
                            coordinates.
            * velocities : (N, 3) ndarray, optional - the corresponding
                           velocities.
            * masses : (N,) ndarray or float, optional - the corresponding
                       masses, or a single mass value if all particles have the
                       same mass.
        halo_ids : array_like, optional if `collate_orbits` has been run or if
                   reading from a file.
            The IDs of the halos at redshift 0.
        snapshot_number : int, optional if `collate_orbits` has been run
            The snapshot number at which to retrieve the halo decomposition.
        angle_cut : float, optional
            Particles that advance about the halo center by less than this
            angle between peri/apocenters will not have that peri/apocenter
            counted. This is designed to remove spurious peri/apocenters from
            orbits within subhalos.
        collated_file : str, optional
            Read the orbit information from a file generated by
            `collate_orbits`.
        verbose : bool, optional
            Print status.

        """

        if collated_file is not None:

            has_final_counts, is_final = False, False
            with h5py.File(collated_file, 'r') as hf:
                hfsnap = hf['snapshot_{}'.format('%03d' % snapshot_number)]
                self.halo_offsets = hfsnap['halo_offsets'][:]
                self.particle_ids = hfsnap['particle_IDs'][:]
                self.orbit_counts = hfsnap['orbit_counts'][:]
                if 'orbit_counts_final' in hfsnap:
                    has_final_counts = True
                    self.orbit_counts_final = hfsnap['orbit_counts_final'][:]
                else:
                    if len(list(hf.keys())) > 2:
                        if 'orbit_counts_final' in hf[list(hf.keys())[-2]]:
                            is_final = True
                self.halo_ids_collated = hfsnap['halo_IDs_final'][:]

        elif not hasattr(self, "orbit_counts"):

            self.collate_orbits(
                halo_ids, snapshot_number, angle_cut, verbose=verbose)

        if halo_ids is None:
            halo_ids = self.halo_ids_collated
        else:
            halo_ids = np.atleast_1d(halo_ids)

        halo_inds = myin1d(self.halo_ids_collated, halo_ids)

        slices = np.array(list(zip(
            self.halo_offsets[:-1], self.halo_offsets[1:])))[halo_inds]
        self.halo_offsets = np.cumsum([0] + [sl[1]-sl[0] for sl in slices])
        
        self.particle_ids = np.concatenate(
            [self.particle_ids[slice(*sl)] for sl in slices])
        self.orbit_counts = np.concatenate(
            [self.orbit_counts[slice(*sl)] for sl in slices])
        if has_final_counts:
            self.orbit_counts_final = np.concatenate(
                [self.orbit_counts_final[slice(*sl)] for sl in slices])
        elif not has_final_counts and is_final:
            self.orbit_counts_final = self.orbit_counts

        if 'masses' in snapshot_data:
            if hasattr(snapshot_data['masses'], "__len__"):
                mass_is_array = True
            else:
                mass_is_array = False
        
        snapshot_data['ids'], inds_u = np.unique(
            snapshot_data['ids'], return_index=True)
        ids_u, inv = np.unique(self.particle_ids, return_inverse=True)
        inds = myin1d(snapshot_data['ids'], ids_u)[inv]

        if 'coordinates' in snapshot_data:
            snapshot_data['coordinates'] = snapshot_data['coordinates'][inds_u]
            self.coordinates = snapshot_data['coordinates'][inds]
        if 'velocities' in snapshot_data:
            snapshot_data['velocities'] = snapshot_data['velocities'][inds_u]
            self.velocities = snapshot_data['velocities'][inds]
        if 'masses' in snapshot_data:
            if mass_is_array:
                snapshot_data['masses'] = snapshot_data['masses'][inds_u]
                self.masses = snapshot_data['masses'][inds]
            else:
                self.masses = snapshot_data['masses']

        return

    def save_final_orbit_counts(self, collated_file, snapshot_numbers=None,
                                verbose=True):

        """
        Save the orbit counts that the particles at each snapshot will have by
        the time of the final snapshot.

        Parameters
        ----------
        collated_file : str, optional
            File generated by `collate_orbits` that contains the complete
            particle ID and orbit count information at each snapshot.
        snapshot_numbers : array_like, optional
            Snapshot numbers for which to save the final orbit counts.
        verbose : bool, optional
            Print status.

        """

        with h5py.File(collated_file, 'r+') as hf:

            skey_final = list(hf.keys())[-1]
            counts_all_final = hf[skey_final]['orbit_counts'][:]
            ids_all_final = hf[skey_final]['particle_IDs'][:]
            offsets_all_final = hf[skey_final]['halo_offsets'][:]
            slices_all_final = np.array(
                list(zip(offsets_all_final[:-1], offsets_all_final[1:])))

            skeys = np.array(list(hf.keys())[1:-1])
            if snapshot_numbers is not None:
                snap_nums_from_file = np.array(
                    [int(skey.split('_')[1]) for skey in skeys])
                snap_inds = np.where(np.in1d(
                    snap_nums_from_file, np.asarray(snapshot_numbers)))
                skeys = skeys[snap_inds]

            for skey in skeys:

                counts_all = hf[skey]['orbit_counts'][:]
                ids_all = hf[skey]['particle_IDs'][:]
                offsets_all = hf[skey]['halo_offsets'][:]
                slices_all = np.array(
                    list(zip(offsets_all[:-1], offsets_all[1:])))
                halo_ids = hf[skey]['halo_IDs_final'][:]

                counts_retro_all = np.empty(len(counts_all), dtype=np.int16)

                slinds = myin1d(hf['halo_IDs'][:], halo_ids)
                for slf, sl in zip(
                        slices_all_final[slinds], slices_all[slinds]):
                    
                    ids_final = ids_all_final[slice(*slf)]
                    counts_final = counts_all_final[slice(*slf)]

                    ids = ids_all[slice(*sl)]
                    counts = counts_all[slice(*sl)]

                    counts_retro = -np.ones(len(counts), dtype=np.int16)
                    ids_departed = np.setdiff1d(ids, ids_final)
                    inds_departed = myin1d(ids, ids_departed)
                    mask = np.ones(len(counts), dtype=bool)
                    mask[inds_departed] = False
                    inds_retained = np.where(mask)[0]
                    inds = myin1d(ids_final, ids[inds_retained])
                    counts_retro[inds_retained] = counts_final[inds]
                    counts_retro_all[slice(*sl)] = counts_retro

                hf[skey].create_dataset(
                    'orbit_counts_final', data=counts_retro_all)

                if verbose:
                    print('Final counts saved for {} {}'.format(
                        *(skey.split('_'))))

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
