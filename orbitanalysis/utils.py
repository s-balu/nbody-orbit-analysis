import numpy as np


def myin1d(a, b, kind=None):
    """
    Returns the indices of a with values that are also in b, in the order that
    those elements appear in b.
    """
    loc = np.in1d(a, b, kind=kind)
    order = a[loc].argsort()[b.argsort().argsort()]
    return np.where(loc)[0][order]


def vector_norm(vectors, return_norm=True, return_unit_vectors=False):
    vmags = np.sqrt(np.einsum('...i,...i', vectors, vectors))
    if return_norm and return_unit_vectors:
        return vmags, vectors / vmags[:, np.newaxis]
    elif return_norm:
        return vmags
    elif return_unit_vectors:
        return vectors / vmags[:, np.newaxis]


def recenter_coordinates(position, boxsize):
    """
    Center a periodic simulation box on a specified position.
    """
    if isinstance(boxsize, (float, np.floating, int, np.integer)):
        boxsize = boxsize * np.ones(3)
    for dim, bs in enumerate(boxsize):
        position[np.argwhere((position[:, dim] > bs/2)), dim] -= bs
        position[np.argwhere((position[:, dim] < -bs/2)), dim] += bs
    return position


def hubble_parameter(z, H0, Omega_m, Omega_L, Omega_k=0):
    return H0 * np.sqrt(Omega_m * (1 + z)**3 +
                        Omega_k * (1 + z)**2 +
                        Omega_L)
