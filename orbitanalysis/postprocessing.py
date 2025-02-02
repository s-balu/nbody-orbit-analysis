import numpy as np
import h5py
import time

from orbitanalysis.utils import myin1d


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

            self.mode = datafile.attrs['mode']
            if 'box_size' in datafile.attrs:
                self.box_size = datafile.attrs['box_size']

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

        init_orb = False
        for s in self.snapshot_numbers[:sind+1]:

            with h5py.File(self.filename, 'r') as hf:

                hfs = hf['snapshot_{}'.format('%0.3d' % s)]
                halo_ids_final = hfs['halo_ids_final'][:]
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
            for oids in orbiting_ids:
                oids_u, c = np.unique(oids, return_counts=True)
                orbiting_ids_unique.append(oids_u)
                counts.append(c)
                lens.append(len(oids_u))
            orbiting_ids_unique = np.concatenate(orbiting_ids_unique)
            counts = np.concatenate(counts)
            offsets = np.cumsum([0]+lens)[:-1]

            with h5py.File(savefile, 'a') as hf:

                hfs = hf.create_group('snapshot_{}'.format('%03d' % s))
                hfs.create_dataset('particle_IDs', data=orbiting_ids_unique)
                hfs.create_dataset(
                    '{}er_counts'.format(self.mode[:-3]), data=counts)
                hfs.create_dataset('halo_offsets', data=offsets)
                hfs.create_dataset('halo_IDs_final', data=halo_ids[hinds2])

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
            
            if snapshot_numbers is None:
                skeys_ = skeys[:-1]
            else:
                snap_nums = np.array(
                    [int(skey.split('_')[-1]) for skey in skeys])
                skeys_ = skeys[
                    np.where(np.in1d(snap_nums, snapshot_numbers))[0]]

            for skey in skeys_:

                ids = hf[skey]['particle_IDs'][:]
                final_inds = myin1d(ids_final, ids)

                hf[skey].create_dataset(
                    '{}er_counts_final'.format(self.mode[:-3]),
                    data=counts_final[final_inds])

                if verbose:
                    print('Final counts saved for {} {}'.format(
                        *(skey.split('_'))))
