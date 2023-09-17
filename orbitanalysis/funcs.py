import numpy as np
import time
from simtools.sim_readers import GadgetSnap, GadgetCat, AHFCat, VelociraptorCat


def load_halo_particle_ids_subfind(snapshot, catalogue, s):

    snapshot_ = GadgetSnap(path=snapshot.snapshot_path,
                           snapshot_filename=snapshot.snapshot_filename,
                           snapshot_number=s,
                           particle_type=snapshot.particle_type,
                           load_coords=False,
                           load_vels=False,
                           load_masses=False,
                           read_mode=snapshot.read_mode,
                           npool=snapshot.npool,
                           unit_length_in_cm=snapshot.unit_length,
                           unit_mass_in_g=snapshot.unit_mass,
                           unit_velocity_in_cm_per_s=snapshot.unit_velocity,
                           to_physical=False,
                           buffer=snapshot.buffer,
                           verbose=True
                           )

    catalogue_ = GadgetCat(path=catalogue.catalogue_path,
                           catalogue_filename=catalogue.catalogue_filename,
                           snapshot_number=s,
                           particle_type=catalogue.particle_type,
                           verbose=True
                           )

    if not snapshot_.has_snap or not catalogue_.has_cat:
        return None, None
    else:
        inds_list = [np.arange(offset, offset + length, dtype=np.int32) for
                     offset, length in zip(
                catalogue_.halo['Offset'], catalogue_.halo['Len'])]
        return [snapshot_.ids[inds] for inds in inds_list], catalogue_


def load_snapshot_obj_subfind(snapshot, catalogue, s, hids, n_radii, verbose):

    if verbose:
        print('Loading particle data...')
        t0 = time.time()

    snapshot_ = GadgetSnap(path=snapshot.snapshot_path,
                           snapshot_filename=snapshot.snapshot_filename,
                           snapshot_number=s,
                           particle_type=snapshot.particle_type,
                           cutout_positions=catalogue.halo['Pos'][hids],
                           cutout_radii=n_radii * catalogue.group['R_200'][
                               catalogue.halo['HaloGroupNr'][hids]],
                           read_mode=snapshot.read_mode,
                           npool=snapshot.npool,
                           unit_length_in_cm=snapshot.unit_length,
                           unit_mass_in_g=snapshot.unit_mass,
                           unit_velocity_in_cm_per_s=snapshot.unit_velocity,
                           to_physical=False,
                           buffer=snapshot.buffer,
                           verbose=True
                           )

    if verbose:
        print('Particle data loaded (took {} s)'.format(time.time() - t0))

    return snapshot_


def load_halo_particle_ids_velociraptor(snapshot, catalogue, s):

    catalogue_ = VelociraptorCat(path=catalogue.catalogue_path,
                                 catalogue_filename=
                                 catalogue.catalogue_filename,
                                 snapshot_number=s,
                                 particle_type=catalogue.particle_type,
                                 verbose=True
                                 )
    if not catalogue_.has_cat:
        return None, None
    else:
        return catalogue_.halo['ParticleIDs'], catalogue_


def load_snapshot_obj_velociraptor(snapshot, catalogue, s, hids, n_radii,
                                   verbose):

    if verbose:
        print('Loading particle data...')
        t0 = time.time()

    snapshot_ = GadgetSnap(path=snapshot.snapshot_path,
                           snapshot_filename=snapshot.snapshot_filename,
                           snapshot_number=s,
                           particle_type=snapshot.particle_type,
                           cutout_positions=catalogue.halo['CM'][hids],
                           cutout_radii=n_radii * catalogue.group['R_200'][
                               hids],
                           read_mode=snapshot.read_mode,
                           npool=snapshot.npool,
                           unit_length_in_cm=snapshot.unit_length,
                           unit_mass_in_g=snapshot.unit_mass,
                           unit_velocity_in_cm_per_s=snapshot.unit_velocity,
                           to_physical=False,
                           buffer=snapshot.buffer,
                           verbose=True
                           )

    if verbose:
        print('Particle data loaded (took {} s)'.format(time.time() - t0))

    return snapshot_


def load_halo_particle_ids_ahf(snapshot, catalogue, s):

    catalogue_ = AHFCat(path=catalogue.catalogue_path,
                        catalogue_filename=catalogue.catalogue_filename,
                        snapshot_number=s,
                        particle_type=catalogue.particle_type,
                        verbose=True
                        )

    if not catalogue_.has_cat:
        return None, None
    else:
        return catalogue_.halo['ParticleIDs'], catalogue_


def load_snapshot_obj_ahf(snapshot, catalogue, s, hids, n_radii, verbose):

    if verbose:
        print('Loading particle data...')
        t0 = time.time()

    snapshot_ = GadgetSnap(path=snapshot.snapshot_path,
                           snapshot_filename=snapshot.snapshot_filename,
                           snapshot_number=s,
                           particle_type=snapshot.particle_type,
                           cutout_positions=catalogue.halo['Pos'][hids],
                           cutout_radii=n_radii * catalogue.halo['Radius'][
                               hids],
                           read_mode=snapshot.read_mode,
                           npool=snapshot.npool,
                           unit_length_in_cm=snapshot.unit_length,
                           unit_mass_in_g=snapshot.unit_mass,
                           unit_velocity_in_cm_per_s=snapshot.unit_velocity,
                           to_physical=False,
                           buffer=snapshot.buffer,
                           verbose=True
                           )

    if verbose:
        print('Particle data loaded (took {} s)'.format(time.time() - t0))

    return snapshot_
