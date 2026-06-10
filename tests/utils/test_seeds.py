import numpy as np
import torch
from cf_pipeline.utils.seeds import set_global_seed

def test_seed_makes_numpy_reproducible():
    set_global_seed(42)
    a = np.random.rand(5)
    set_global_seed(42)
    b = np.random.rand(5)
    assert np.array_equal(a, b)

def test_seed_makes_torch_reproducible():
    set_global_seed(42)
    a = torch.rand(5)
    set_global_seed(42)
    b = torch.rand(5)
    assert torch.equal(a, b)

def test_seed_returns_rng():
    rng = set_global_seed(42)
    assert hasattr(rng, "integers")
