import numpy as np
import h5py
import time

from orbitanalysis.utils import myin1d


class Apsides:

    def __init__(self, filename):

        self.filename = filename
        snapshot_numbers = []

        with h5py.File(filename, 'r') as hf:

            skeys = list(hf.keys())
            for skey in skeys:

                snapshot_numbers.append(int(skey.split('_')[1]))

            self.final_halo_ids = hf[skeys[-1]]['halo_IDs'][:]

            self.mode = hf.attrs['mode']
            if 'box_size' in hf.attrs:
                self.box_size = hf.attrs['box_size']

        self.snapshot_numbers = np.array(snapshot_numbers)

    def collate_apsides(self, halo_ids=None, snapshot_number=None,
                        angle_cut=np.pi/4, save_final_counts=False,
                        data_type=None, savefile=None, verbose=True):

        """
        Collate the peri/apocenter information to obtain the complete set of
        orbiting particle IDs and their counts at each snapshot, subject to an
        angle cut.

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
            halo_ids = self.final_halo_ids
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

        init_orb = False
        for s in self.snapshot_numbers[:sind+1]:

            with h5py.File(self.filename, 'r') as hf:

                hfs = hf['snapshot_{}'.format('%0.3d' % s)]

                region_positions = hfs['region_positions'][:]
                region_radii = hfs['region_radii'][:]
                bulk_velocities = hfs['bulk_velocities'][:]
                
                halo_ids_current = hfs['halo_IDs'][:]
                if s != self.snapshot_numbers[-1]:
                    halo_ids_final = hfs['final_descendant_IDs'][:]
                else:
                    halo_ids_final = halo_ids_current
                common = np.intersect1d(halo_ids_final, halo_ids)
                hinds1 = myin1d(halo_ids_final, common)
                hinds2 = myin1d(halo_ids, common)

                if len(hfs['{}er_IDs'.format(self.mode[:-3])]) > 0:

                    if not init_orb:
                        if data_type is None:
                            orbtype = hfs[
                                '{}er_IDs'.format(self.mode[:-3])].dtype
                        else:
                            orbtype = data_type
                        orbiting_ids = [
                            np.array([], dtype=orbtype) for _ in halo_ids]
                        init_orb = True
                
                    hoffsets = hfs['region_offsets'][:]
                    hslices = list(zip(hoffsets[:-1], hoffsets[1:]))

                    for hind1, hind2 in zip(hinds1, hinds2):

                        peri_ids = hfs['{}er_IDs'.format(self.mode[:-3])][
                            slice(*hslices[hind1])]
                        angles = hfs['angles'][slice(*hslices[hind1])]

                        orbiting_ids[hind2] = np.append(
                            orbiting_ids[hind2], peri_ids[angles>angle_cut])
                
                else:
                    continue

            orbiting_ids_unique, counts, lens = [], [], []
            for i, oids in enumerate(orbiting_ids):
                oids_u, c = np.unique(oids, return_counts=True)
                orbiting_ids_unique.append(oids_u)
                counts.append(c)
                if i in hinds2:
                    lens.append(len(oids_u))
            orbiting_ids_unique = np.concatenate(orbiting_ids_unique)
            counts = np.concatenate(counts)
            offsets = np.cumsum([0]+lens)[:-1]

            final_halo_ids = halo_ids_final[hinds1] if s != \
                self.snapshot_numbers[-1] else None
            with h5py.File(savefile, 'a') as hf:

                hfs = hf.create_group('snapshot_{}'.format('%03d' % s))
                hfs.create_dataset('particle_IDs', data=orbiting_ids_unique)
                hfs.create_dataset(
                    '{}er_counts'.format(self.mode[:-3]), data=counts)
                hfs.create_dataset('halo_offsets', data=offsets)
                if final_halo_ids is not None:
                    hfs.create_dataset(
                        'final_descendant_IDs', data=final_halo_ids)
                hfs.create_dataset('halo_IDs', data=halo_ids_current[hinds1])
                hfs.create_dataset(
                    'halo_positions', data=region_positions[hinds1])
                hfs.create_dataset(
                    'halo_velocities', data=bulk_velocities[hinds1])
                hfs.create_dataset(
                    'region_radii', data=region_radii[hinds1])

            if verbose:
                print('Snapshot {} collated'.format('%03d' % s))

        if save_final_counts:
            self.save_final_apsis_counts(savefile, verbose=verbose)

        if verbose:
            print('{}ers collated in {} s'.format(
                self.mode[:-3], round(time.time()-t_start, 3)))

        return

    def save_final_apsis_counts(self, collated_file, snapshot_numbers=None,
                                verbose=True):

        """
        Save the orbit counts that the particles at each snapshot will have by
        the time of the final snapshot.

        Parameters
        ----------
        collated_file : str, optional
            File generated by `collate_apsides` that contains the complete
            particle ID and orbit count information at each snapshot.
        snapshot_numbers : array_like, optional
            Snapshot numbers for which to save the final orbit counts.
        verbose : bool, optional
            Print status.

        """

        with h5py.File(collated_file, 'r+') as hf:

            skeys = np.array(list(hf.keys()))

            ids_final = hf[skeys[-1]]['particle_IDs'][:]
            counts_final = hf[skeys[-1]][
                '{}er_counts'.format(self.mode[:-3])][:]
            halo_ids = hf[skeys[-1]]['halo_IDs'][:]
            offsets_final = list(
                hf[skeys[-1]]['halo_offsets'][:]) + [len(ids_final)]
            slices_final = list(zip(offsets_final[:-1], offsets_final[1:]))

            if snapshot_numbers is None:
                skeys_ = skeys[:-1]
            else:
                snap_nums = np.array(
                    [int(skey.split('_')[-1]) for skey in skeys])
                skeys_ = skeys[
                    np.where(np.in1d(snap_nums, snapshot_numbers))[0]]

            for skey in skeys_:

                ids = hf[skey]['particle_IDs'][:]
                desc_ids = hf[skey]['final_descendant_IDs'][:]
                offsets = list(hf[skey]['halo_offsets'][:]) + [len(ids)]
                slices = list(zip(offsets[:-1], offsets[1:]))

                hinds = myin1d(halo_ids, desc_ids)

                counts_retro = np.empty(len(ids))
                for hind2, hind1 in enumerate(hinds):
                    
                    final_inds = myin1d(
                        ids_final[slice(*slices_final[hind1])],
                        ids[slice(*slices[hind2])])
                    
                    counts_retro[slice(*slices[hind2])] = counts_final[
                        slice(*slices_final[hind1])][final_inds]

                hf[skey].create_dataset(
                    '{}er_counts_final'.format(self.mode[:-3]),
                    data=counts_retro)

                if verbose:
                    print('Final counts saved for {} {}'.format(
                        *(skey.split('_'))))
