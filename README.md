# mnist-cuda-speedrun

From pure C to hand-written CUDA with BF16 Tensor Cores — a clean progression showing every optimization step toward maximum GPU efficiency on MNIST.

## Setup

```bash
bash downloader_data.sh
```

## Implementations

### 3-Layer MLP — 784 → 512 → 512 → 10

| File | Language | Epochs | Train Time | Final Loss | Notes |
|------|----------|--------|------------|------------|-------|
| `00_pure_c_3x512.c` | C (CPU) | 10 | — | — | No BLAS, true baseline |
| `01_numpy_3x512.py` | NumPy | 10 | 28.7 s | 0.156 | Readable, vectorized |
| `02_pytorch_3x512.py` | PyTorch | 40 | ~11s s | 0.041 | Convenient, unoptimized |
| `03_pytorch_3x512_compiled.py` | PyTorch + `torch.compile` | 10 | ~40 s | 0.041 | Compile overhead > gain on small model |
| `04_cuda_3x512_naive.cu` | CUDA (naive matmul) | 40 | ~20.6 s | 0.0116 | Hand-written kernels, no tiling |
| `05_cuda_3x512_basictiles.cu` | CUDA (tiled matmul) | 40 | ~11 s | ~0.009 | Shared memory tiles in gemms, big jump |
| `06_cuda_3x512_cublas.cu` | CUDA (cuBLAS) | 40 | **2.26 s** | ~0.214 | First cuBLAS integration, naive host-device flow |
| `07_cuda_3x512_fused.cu` | CUDA (cuBLAS + fused) | 40 | **1.7 s** | ~0.214 | GPU-resident data, fused bias+ReLU and weight-update kernels, warp-shuffle softmax, GPU-side loss |

`07_cuda_3x512_fused.cu` builds on 06: data fully GPU-resident at startup (no per-batch memcpy), fused bias+ReLU and weight-update kernels, warp-shuffle softmax, loss accumulated directly on device.

---

### 2-Layer MLP — 784 → 512 → 10

Although my 3-layer "MLP" was fun, it was time for me to switch to 2-layer architecture to mirror the standard FFN blocks in Transformers. See the next project here [CUDA Transformer](https://github.com/legio201516/cuda-transformer.git) But first i needed to to explore BF16 mixed precision + tensor cores.

| File | Language | Epochs | Train Time | Final Loss | Notes |
|------|----------|--------|------------|------------|-------|
| `02_pytorch_2x512.py` | PyTorch | 40 | ~3.4 s* | ~0.237 | Includes evaluate; warm regime |
| `07_cuda_2x512_bf16_fwd.cu` | CUDA BF16 (fwd only) | 40 | ~0.767 s | ~0.237 | Forward pass in BF16, bwd in FP32 |
| `07_cuda_2x512_bf16.cu` | CUDA full BF16 pipeline | 40 | **0.567 s** | ~0.214 | Full precision strategy below |

**~6× faster than PyTorch on the same architecture** (3.4 s → 0.567 s).
**~4× faster than torch.compile on the same architecture** (2.2s → 0.567 s).


---

## Precision Strategy — `07_cuda_2x512_bf16.cu`

Every precision decision, one line each:

- **All GEMMs**: BF16 inputs × BF16 weights → FP32 accumulation (`CUBLAS_COMPUTE_32F`). Tensor Cores engaged, no precision loss in accumulation.
- **`X_bf16` and `H_bf16`**: cast once in forward, reused directly in backward — zero redundant casts.
- **Bias gradients `db1`, `db2`**: FP32 `cublasSgemv` — vectors are tiny (512, 10), no native BF16 GEMV in cuBLAS.
- **ReLU mask**: applied on FP32 `H` — I don't want that the edge cases around zero with BF16.
- **`dW1`, `dW2`**: accumulated in FP32 — BF16 gradients would underflow small updates and break convergence.
- **Master weights**: kept in FP32 throughout.

---

## Training Time Remarks

**cuBLAS warm-up**: the first PyTorch run in a fresh process is ~1.5–2× slower than subsequent runs. cuBLAS benchmarks multiple GEMM algorithms on first call for a given shape, then caches the winner in memory. Changing batch size (e.g. 1024 → 512 → 1024) forces a re-tune for new shapes. This is not throttling — it's expected behavior. Stabilized regime: ~3.4 s.

**GPU boost/thermal**: laptop GPU boost clocks depend on thermal state. First cold run often faster (full boost), then a dip, then stabilization. ~10% difference plugged vs battery (power cap throttling).

**`torch.compile` **: Even on a model this small, compiling with torch.compile shows improvements as it make us go from 3.4 to 2.4s in training time when done with proper warm up runs. 

**Hidden size scaling**: switching to `HIDDEN_SIZE = 1024` on the 3-layer modeldoubled training time usually — expected, the hidden×hidden matmul is `O(H²)`.

---

## Profiling

Most of the profiling was simply done with NVIDIA Nsight Systems — aka `nsys`.

To save hours I wrote a **cool innovative Python script** (`nsysrun.py`) that automates the loop: runs `nsys profile`, extracts key kernels (`kernel_tiled_matmul_*`, cuBLAS calls, memcpy), and saves clean `.txt` reports to `/profiles/`.


```bash
python3 nsysrun.py run <output_name> ./<binary>
```

## Next !

:\)

-I'm very happy of all this, I learned a lot. 

-I will try to run benchmarks on different gpus than my current Rtx 4050.

-And now let's use all this to do a some more interresting stuff [CUDA Transformer](https://github.com/legio201516/cuda-transformer.git)...
