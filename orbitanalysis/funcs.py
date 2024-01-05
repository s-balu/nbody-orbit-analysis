import numpy as np
import time
from simtools.sim_readers import GadgetSnapshot, GadgetCatalogue, \
    AHFCatalogue, VelociraptorCatalogue


def load_halo_particle_ids_gadget(snapshot_obj, catalogue_obj,
                                  snapshot_number):

    snapshot_ = GadgetSnapshot(
        path=snapshot_obj.snapshot_path,
        snapshot_filename=snapshot_obj.snapshot_filename,
        snapshot_number=snapshot_number,
        particle_type=snapshot_obj.particle_type,
        load_coords=False,
        load_vels=False,
        load_masses=False,
        read_mode=snapshot_obj.read_mode,
        npool=snapshot_obj.npool,
        unit_length_in_cm=snapshot_obj.unit_length,
        unit_mass_in_g=snapshot_obj.unit_mass,
        unit_velocity_in_cm_per_s=snapshot_obj.unit_velocity,
        buffer=snapshot_obj.buffer,
        verbose=True
        )

    catalogue_ = GadgetCatalogue(
        path=catalogue_obj.catalogue_path,
        catalogue_filename=catalogue_obj.catalogue_filename,
        snapshot_number=snapshot_number,
        particle_type=catalogue_obj.particle_type,
        verbose=True
        )

    if not snapshot_.has_snap or not catalogue_.has_cat:
        return None, None
    else:
        inds_list = [
            np.arange(offset, offset + npart, dtype=np.int32) for
            offset, npart in zip(catalogue_.halo['offset'], catalogue_.halo[
                'number_of_particles'])]
        return [snapshot_.ids[inds] for inds in inds_list], catalogue_


def load_snapshot_obj_gadget(snapshot_obj, snapshot_number, region_positions,
                             region_radii, verbose):

    if verbose:
        print('Loading particle data...')
        t0 = time.time()

    snapshot_ = GadgetSnapshot(
        path=snapshot_obj.snapshot_path,
        snapshot_filename=snapshot_obj.snapshot_filename,
        snapshot_number=snapshot_number,
        particle_type=snapshot_obj.particle_type,
        region_positions=region_positions,
        region_radii=region_radii,
        use_kdtree=snapshot_obj.use_kdtree,
        read_mode=snapshot_obj.read_mode,
        npool=snapshot_obj.npool,
        unit_length_in_cm=snapshot_obj.unit_length,
        unit_mass_in_g=snapshot_obj.unit_mass,
        unit_velocity_in_cm_per_s=snapshot_obj.unit_velocity,
        buffer=snapshot_obj.buffer,
        verbose=True
        )

    if verbose:
        print('Particle data loaded (took {} s)'.format(time.time() - t0))

    return snapshot_


def load_halo_particle_ids_velociraptor(snapshot_obj, catalogue_obj,
                                        snapshot_number):

    catalogue_ = VelociraptorCatalogue(
        path=catalogue_obj.catalogue_path,
        catalogue_filename=catalogue_obj.catalogue_filename,
        snapshot_number=snapshot_number,
        particle_type=catalogue_obj.particle_type,
        verbose=True
        )
    if not catalogue_.has_cat:
        return None, None
    else:
        return catalogue_.halo['particle_IDs'], catalogue_


def load_halo_particle_ids_ahf(snapshot, catalogue_obj, snapshot_number):

    catalogue_ = AHFCatalogue(
        path=catalogue_obj.catalogue_path,
        catalogue_filename=catalogue_obj.catalogue_filename,
        snapshot_number=snapshot_number,
        particle_type=catalogue_obj.particle_type,
        verbose=True
        )

    if not catalogue_.has_cat:
        return None, None
    else:
        return catalogue_.halo['particle_IDs'], catalogue_
