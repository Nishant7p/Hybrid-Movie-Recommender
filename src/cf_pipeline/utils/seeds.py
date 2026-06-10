import os
import random
import numpy as np
import torch

def set_global_seed(seed: int = 42) -> np.random.Generator:
    """Set seeds across random, numpy, torch (cpu+cuda) and return a numpy Generator."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return np.random.default_rng(seed)
