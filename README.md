# `jax.jit` drops a transpose before a Triton GEMM fusion (JAX 0.12, GPU)

On GPU, JAX 0.12.0 and 0.12.0.1 compile `transpose -> sum-pool -> matmul` to a
wrong result. The same function run eagerly, on CPU, on JAX ≤ 0.11.2, or with
`XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` matches NumPy.

```python
def f(a, b):  # a: (16, 16, 3), b: (16, 32)
    pooled = a.transpose(2, 0, 1).reshape(3, 4, 4, 4, 4).sum((2, 4))  # (3, 4, 4)
    return pooled.reshape(3, 16) @ b
```

```sh
uv run repro.py      # exits 1 when the jitted result is wrong
```

## Results

Relative error of the result against NumPy in float64. Eager GPU error is TF32 rounding.

| setting | eager | jit | `repro.py` exit |
|---|---|---|---|
| jax 0.12.0.1, GPU | 3.0e-04 | **1.1e+00** | 1 |
| jax 0.12.0, GPU | 3.0e-04 | **1.1e+00** | 1 |
| jax 0.12.0.1, GPU, `--xla_gpu_enable_triton_gemm=false` | 3.0e-04 | 3.0e-04 | 0 |
| jax 0.12.0.1, CPU (`JAX_PLATFORMS=cpu`) | 8.0e-08 | 8.0e-08 | 0 |
| jax 0.11.2, GPU | 3.0e-04 | 3.0e-04 | 0 |
| jax 0.10.2, GPU | 3.0e-04 | 3.0e-04 | 0 |

To run another version: `uv run --with "jax[cuda13]==0.11.2" repro.py`.

The bug appears across different pool sizes and number of channels.
Without the pooling
(`a.transpose(2, 0, 1)[:, ::4, ::4]`), or with the pooling done before the transpose,
the result is correct.

## Compiled HLO

On 0.11.2, the fusion feeding the GEMM produces the
transposed `(k, h, w)` array, so the bitcast to `[3,16]` is a valid reshape:

```
%loop_transpose_fusion = f32[3,4,4]{2,1,0} fusion(%a.1), calls=%fused_transpose
%bitcast.9.0 = f32[3,16]{1,0} bitcast(%loop_transpose_fusion)
ROOT %gemm_fusion_dot_general.1 = f32[3,32]{1,0} fusion(%bitcast.9.0, %b.1), ...
```

On 0.12.0.1, the reduce fusion produces `(h, w, k)`, and that buffer is *bitcast*
to `[3,16]{1,0}`:

```
%loop_reduce_fusion = f32[4,4,3]{2,1,0} fusion(%a.1), ..."kind":"__triton"
%bitcast.8.0 = f32[3,16]{1,0} bitcast(%loop_reduce_fusion)
ROOT %gemm_fusion_dot_general.1 = f32[3,32]{1,0} fusion(%b.1, %bitcast.8.0), ..."kind":"__triton_nested_gemm_fusion"

%fused_computation.gemm_fusion_dot_general.1_computation (param_0.1: f32[16,32], param_1.1: f32[3,16]) -> f32[3,32] {
  %param_1.1 = f32[3,16]{1,0} parameter(1)
  %copy.2 = f32[3,16]{1,0} copy(%param_1.1)
  %param_0.1 = f32[16,32]{1,0} parameter(0)
  ROOT %dot_general.0 = f32[3,32]{1,0} dot(%copy.2, %param_0.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
```

Printed by `jax.jit(f).lower(a, b).compile().as_text()`.

## Environment

- NVIDIA GeForce RTX 4090 (compute capability 8.9), driver 615.71.09, CUDA UMD 13.4
- Linux 7.0.0 (Ubuntu 24.04), x86_64; Python 3.13.9
- `jax[cuda13]==0.12.0.1` (`jaxlib`, `jax-cuda13-plugin`, `jax-cuda13-pjrt` 0.12.0.1), numpy 2.4.4.
  The versions are pinned in `pyproject.toml` and `uv.lock`.

## Context

Found in a U-Net that mean-pools `(H, W, K)` region masks (transposed to
`(K, H, W)`) and contracts them with features (`einsum("khw,chw->kc")`).
Training losses jumped to ~1e12 whenever one of the regions was empty, because
the pooled sum was then divided by ~1e-6.
