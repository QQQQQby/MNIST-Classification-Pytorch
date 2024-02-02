"""
Utils
"""

import numpy as np


def shuffle_arrays_in_unison(*arrays):
    """Shuffle multiple arrays in unison"""
    if not arrays:
        return
    rng_state = np.random.get_state()
    np.random.shuffle(arrays[0])
    for i in range(1, len(arrays)):
        np.random.set_state(rng_state)
        np.random.shuffle(arrays[i])
