import numpy as np
import h5py

from orbitanalysis.track_orbits import track_orbits
from orbitanalysis.postprocessing import OrbitDecomposition
from orbitanalysis.utils import vector_norm, recenter_coordinates

main_branches = np.flip(
    np.loadtxt('/path/to/main_branches.txt', dtype=int), axis=0)

final_snapshot_number = 48
snapshot_numbers = np.arange(
    final_snapshot_number+1-len(main_branches), final_snapshot_number+1)

savedir = '/path/to/directory'
savefile = savedir + '/orbit_decomposition.hdf5'


def regions(snapshot_number, haloids):

    """
    A function that takes a snapshot and a list of halo IDs and returns the
    coordinates of the centers of the halos and the radii of the regions within
    which to track orbits. In this case, the positions are those of the minimum
    potential and the radii are four times R_200c.

    """

    with h5py.File('/path/to/halo_catalogue_{}.hdf5'.format(
            '%03d' % snapshot_number), 'r') as catalogue_data:

        return catalogue_data['position_of_minimum_potential'][haloids], \
            4 * catalogue_data['R_200crit'][haloids]


def load_snapshot_data(snapshot_number, region_positions, region_radii):

    """
    A function that takes a snapshot number and a list of region positions and
    radii and returns the required particle data within those regions.
    """

    snapshot = {}
    with h5py.File('/path/to/snapshot_{}.hdf5'.format(
            '%03d' % snapshot_number), 'r') as snapshot_data:

        coordinates = snapshot_data['Coordinates'][:]
        box_size = snapshot_data.attrs['BoxSize']

        region_inds = []
        for position, radius in zip(
                np.atleast_2d(region_positions), np.atleast_1d(region_radii)):
            r = vector_norm(
                recenter_coordinates(coordinates-position, box_size))
            region_inds.append(np.argwhere(r < radius).flatten())
        region_lens = [len(inds) for inds in region_inds]
        region_offsets = np.cumsum([0] + region_lens)[:-1]
        region_inds = np.hstack(region_inds).astype(int)

        snapshot['ids'] = snapshot_data['ParticleIDs'][region_inds]
        snapshot['coordinates'] = coordinates[region_inds]
        snapshot['velocities'] = snapshot_data['Velocities'][region_inds]
        snapshot['masses'] = snapshot_data['Masses'][region_inds]
        snapshot['region_offsets'] = region_offsets
        snapshot['box_size'] = box_size  # for periodic boxes. Optional.

    return snapshot


# track orbits by counting pericenters
track_orbits(
    snapshot_numbers, main_branches, regions, load_snapshot_data, savefile,
    mode='pericentric', verbose=True)

# post-processing
orb_decomp = OrbitDecomposition(savefile)

halo_id = main_branches[-1][0]  # first halo in the list
snapdata = load_snapshot_data(
    final_snapshot_number, *regions(final_snapshot_number, halo_id))

# Read orbit decomposition using an angle cut of pi/2
orb_decomp.get_halo_decomposition_at_snapshot(
    halo_id=halo_id, snapshot_number=final_snapshot_number,
    snapshot_data=snapdata, angle_cut=np.pi/2)

# Plotting
orb_decomp.plot_position_space(
    projection='xy', colormap='rainbow_r', counts_to_plot='all',
    xlabel=r'$x/(4R_{200})$', ylabel=r'$y/(4R_{200})$', display=False,
    savefile=savedir + '/position_space.png')
orb_decomp.plot_phase_space(
    colormap='rainbow_r', counts_to_plot='all', radius_label=r'$r/(4R_{200})$',
    radial_velocity_label=r'$v_r\,\,({\rm km\, s}^{-1})$', logr=True,
    display=False, savefile=savedir + '/phase_space.png')
