import numpy as np


def myin1d(a, b, kind=None):
    """
    Returns the indices of a with values that are also in b, in the order that
    those elements appear in b.

    """
    loc = np.in1d(a, b, kind=kind)
    order = a[loc].argsort()[b.argsort().argsort()]
    return np.where(loc)[0][order]


def magnitude(vectors, return_magnitude=True, return_unit_vectors=False):
    vmags = np.sqrt(np.einsum('...i,...i', vectors, vectors))
    if return_magnitude and return_unit_vectors:
        return vmags, vectors / vmags[:, np.newaxis]
    elif return_magnitude:
        return vmags
    elif return_unit_vectors:
        return vectors / vmags[:, np.newaxis]
