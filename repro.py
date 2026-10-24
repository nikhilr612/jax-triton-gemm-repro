"""``jax.jit`` miscompiles transpose -> sum-pool -> matmul on GPU (JAX 0.12.0, 0.12.0.1).

The same function run eagerly, or jitted with ``--xla_gpu_enable_triton_gemm=false``,
matches NumPy; jitted, it is wrong by more than the result's own magnitude.
Exits with status 1 when the jitted result is wrong.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np


def f(a, b):
    """(16, 16, 3) -> (3, 16, 16) -> 4x4 sum-pool -> (3, 16) @ (16, 32)."""
    pooled = a.transpose(2, 0, 1).reshape(3, 4, 4, 4, 4).sum((2, 4))
    return pooled.reshape(3, 16) @ b


rng = np.random.default_rng(0)
a = rng.normal(size=(16, 16, 3)).astype(np.float32)
b = rng.normal(size=(16, 32)).astype(np.float32)
ref = f(a.astype(np.float64), b.astype(np.float64))  # NumPy, float64


def rel_err(x):
    return float(np.abs(np.asarray(x) - ref).max() / np.abs(ref).max())


eager, jitted = rel_err(f(jnp.asarray(a), jnp.asarray(b))), rel_err(jax.jit(f)(a, b))
print(f"jax {jax.__version__}, jaxlib {jax.lib.__version__}, {jax.devices()[0].device_kind}")
print(f"relative error vs NumPy: eager {eager:.1e}, jit {jitted:.1e}")
sys.exit(int(jitted > 1e-2))
