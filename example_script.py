import numpy as np

from simtools.box import Snapshot, Catalogue
from simtools.sim_readers import GadgetSnap, GadgetCat

from orbitanalysis.funcs import load_halo_particle_ids_subfind, \
    load_snapshot_obj_subfind
from orbitanalysis.track_orbits import track_orbits
from orbitanalysis.collate import collate_orbit_history
from orbitanalysis.postprocessing import read_orbiting_decomposition, \
    plot_position_space, plot_phase_space

###############################################################################

snapshot_dir = 'data/DM-L25-N128-eps0.004/snapshots'
catalaogue_dir = 'data/DM-L25-N128-eps0.004/catalogues'
snapshot_filename = 'snapshot_{}.hdf5'
catalogue_filename = 'fof_subhalo_tab_{}.hdf5'
read_mode = 1

particle_type = 1
initial_snapshot_number, final_snapshot_number = 0, 48
groupids_at_snapshot = np.arange(0, 1)

n_radii = 4
mode = 'pericentric'

savedir = snapshot_dir + '/orbiting_decomposition'
savefile_onthefly = savedir + \
    '/orbiting_decomposition_DM-L25-N128_onthefly.hdf5'
savefile = savedir + '/orbiting_decomposition_DM-L25-N128.hdf5'

###############################################################################

final_catalogue = Catalogue(GadgetCat,
                            {'path': catalaogue_dir,
                             'catalogue_filename': catalogue_filename,
                             'snapshot_number': final_snapshot_number,
                             'particle_type': particle_type,
                             'verbose': True}
                            )

haloids_at_final_snapshot = final_catalogue.group['FirstSub'][
    groupids_at_snapshot]
haloids_at_final_snapshot = haloids_at_final_snapshot[
    haloids_at_final_snapshot > -1]

final_snapshot = Snapshot(GadgetSnap,
                          {'path': snapshot_dir,
                           'snapshot_filename': snapshot_filename,
                           'snapshot_number': final_snapshot_number,
                           'particle_type': particle_type,
                           'cutout_positions': final_catalogue.halo['Pos'][
                               haloids_at_final_snapshot],
                           'cutout_radii': n_radii * final_catalogue.group[
                               'R_200'][final_catalogue.halo['HaloGroupNr'][
                                   haloids_at_final_snapshot]],
                           'read_mode': read_mode,
                           'to_physical': False,
                           'buffer': 1.0e-7,
                           'verbose': True}
                          )

track_orbits(load_halo_particle_ids_subfind, load_snapshot_obj_subfind,
             final_snapshot, final_catalogue, haloids_at_final_snapshot,
             n_radii, mode, initial_snapshot_number,
             savefile=savefile_onthefly, verbose=True)

collate_orbit_history(savefile_onthefly, savefile)

IDS, COORDS, VELS, MASSES, counts = read_orbiting_decomposition(
    savefile, 48, 0, angle_condition=np.pi/2, filter_ids=None)

plot_position_space(COORDS, counts, colormap='inferno_r',
                    savefile=savedir + '/position_space.png')
plot_phase_space(COORDS, VELS, counts, colormap='inferno_r',
                 savefile=savedir + '/phase_space.png')
