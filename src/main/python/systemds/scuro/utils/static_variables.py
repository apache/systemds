import numpy as np

global_rng = np.random.default_rng(42)


def get_seed():
    return global_rng.integers(0, 1024)
