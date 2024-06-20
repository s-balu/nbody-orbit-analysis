import numpy as np
import h5py
from scipy.spatial import KDTree

from orbitanalysis.track_orbits import track_orbits
from orbitanalysis.postprocessing import OrbitDecomposition
from orbitanalysis.utils import vector_norm, recenter_coordinates

from dmutils.paths import Paths

simdir = Paths().home + '/sims/DM-L25-N128-eps0.004/output'
savedir = simdir + '/orbit_decomposition'

main_branches = np.flip(
    np.loadtxt(savedir + '/main_branches.txt', dtype=int)[:, :], axis=0)
savefile = savedir + '/orbit_decomposition.hdf5'
snapshot_numbers = np.arange(49-len(main_branches), 49)


def regions(snapshot_number, haloids):

    with h5py.File(simdir + '/fof_subhalo_tab_{}.hdf5'.format(
            '%03d' % snapshot_number), 'r') as catalogue_data:

        return catalogue_data['Subhalo/SubhaloPos'][:][haloids], \
            4 * catalogue_data['Group/Group_R_Crit200'][:][
                catalogue_data['Subhalo/SubhaloGroupNr'][:][haloids]]


def load_snapshot_data(snapshot_number, region_positions, region_radii):

    snapshot = {}
    with h5py.File(simdir + '/snapshot_{}.hdf5'.format(
            '%03d' % snapshot_number), 'r') as snapshot_data:

        coordinates = snapshot_data['PartType1/Coordinates'][:]
        box_size = snapshot_data['Parameters'].attrs['BoxSize']

        # region_inds = []
        # for position, radius in zip(
        #         np.atleast_2d(region_positions), np.atleast_1d(region_radii)):
        #     r = vector_norm(
        #         recenter_coordinates(coordinates-position, box_size))
        #     region_inds.append(np.argwhere(r < radius).flatten())
        # region_lens = [len(inds) for inds in region_inds]

        kdtree = KDTree(coordinates, boxsize=box_size*(1+1e-7))
        region_inds = kdtree.query_ball_point(
            region_positions, region_radii)
        region_lens = list(np.atleast_1d(kdtree.query_ball_point(
            region_positions, region_radii, return_length=True)))

        region_offsets = np.cumsum([0] + region_lens)[:-1]
        region_inds = np.hstack(region_inds).astype(int)

        snapshot['ids'] = snapshot_data['PartType1/ParticleIDs'][:][
            region_inds]
        snapshot['coordinates'] = coordinates[region_inds]
        snapshot['velocities'] = snapshot_data['PartType1/Velocities'][:][
            region_inds]
        snapshot['masses'] = snapshot_data['Header'].attrs['MassTable'][1]
        snapshot['region_offsets'] = region_offsets
        snapshot['box_size'] = box_size

    return snapshot


track_orbits(
    snapshot_numbers, main_branches, regions, load_snapshot_data, savefile,
    mode='pericentric', npool=2, verbose=True)

orb_decomp = OrbitDecomposition(savefile)
halo_id = main_branches[-1][1]
snapdata = load_snapshot_data(48, *regions(48, halo_id))
orb_decomp.get_halo_decomposition_at_snapshot(
    halo_id=halo_id, snapshot_number=48, snapshot_data=snapdata,
    angle_cut=np.pi/2)

orb_decomp.plot_position_space(
    projection='xy', colormap='rainbow_r', counts_to_plot='all',
    xlabel=r'$x/(4R_{200})$', ylabel=r'$y/(4R_{200})$', display=False,
    savefile=savedir + '/position_space_anglecut.png')
orb_decomp.plot_phase_space(
    colormap='rainbow_r', counts_to_plot='all', radius_label=r'$r/(4R_{200})$',
    radial_velocity_label=r'$v_r\,\,({\rm km\, s}^{-1})$', logr=False,
    display=False, savefile=savedir + '/phase_space_anglecut.png')
