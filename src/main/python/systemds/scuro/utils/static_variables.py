import numpy as np
import torch

global_rng = np.random.default_rng(42)


def get_seed():
    return global_rng.integers(0, 1024)


def get_device():
    return torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "mps" if torch.mps.is_available() else "cpu"
    )
