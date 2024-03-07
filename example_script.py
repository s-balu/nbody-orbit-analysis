import numpy as np

from simtools.sim_readers import GadgetSnapshot, GadgetCatalogue

from orbitanalysis.track_orbits import track_orbits
from orbitanalysis.postprocessing import OrbitDecomposition

n_radii = 4  # no. of unit radii (R_200 in this case) to track particles out to
mode = 'pericentric'  # use pericentric passages to count orbits

savedir = '/path/to/directory'
savefile = savedir + '/orbit_decomposition.hdf5'

main_branches = np.flip(
    np.loadtxt(savedir + '/main_branches.txt', dtype=int), axis=0)
snapshot_numbers = np.arange(49-len(main_branches), 49)

final_catalogue = GadgetCatalogue(
    path='/path/to/catalogues',
    catalogue_filename='fof_subhalo_tab_{}.hdf5',
    snapshot_number=48,
    particle_type=1,
    verbose=True
    )


def regions(snapshot_number, haloids, nradii):

    catalogue = GadgetCatalogue(
        path='/path/to/catalogues',
        catalogue_filename='fof_subhalo_tab_{}.hdf5',
        snapshot_number=snapshot_number,
        particle_type=1,
        verbose=True
    )

    return catalogue.halo['position_of_minimum_potential'][haloids], \
        nradii*catalogue.group['R_200crit'][
            catalogue.halo['group_number'][haloids]]


def load_snapshot_object(snapshot_number, region_positions, region_radii):

    snapshot = GadgetSnapshot(
        path='/path/to/snapshots',
        snapshot_filename='snapshot_{}.hdf5',
        snapshot_number=snapshot_number,
        particle_type=1,
        region_positions=region_positions,
        region_radii=region_radii,
        read_mode=1,
        buffer=1.0e-7,
        verbose=True
        )

    return snapshot


track_orbits(
    main_branches, snapshot_numbers, load_snapshot_object, regions, n_radii,
    savefile, mode='pericentric', verbose=True)

# post-processing
orb_decomp = OrbitDecomposition(savefile)

halo_id = main_branches[-1][0]  # first halo in the list
snapdata = load_snapshot_object(48, *regions(48, halo_id, n_radii))
orb_decomp.get_halo_decomposition_at_snapshot(
    snapshot_number=48, halo_id=halo_id, snapshot_data=snapdata,
    angle_condition=np.pi/2)

# plotting
orb_decomp.plot_position_space(
    projection='xy', colormap='inferno_r', counts_to_plot='all',
    xlabel=r'$x/R_{200}$', ylabel=r'$y/R_{200}$', display=False,
    savefile=savedir + '/position_space.png')
orb_decomp.plot_phase_space(
    colormap='inferno_r', counts_to_plot='all', radius_label=r'$r/R_{200}$',
    radial_velocity_label=r'$v_r\,\,({\rm km\, s}^{-1})$', display=False,
    savefile=savedir + '/phase_space.png')
