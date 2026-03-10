# mnist-cuda-speedrun

From pure NumPy to hand-written CUDA tiled matmuls — a clean progression showing every step toward maximum efficiency on MNIST.
## Setup
To download the dataset, run:
`bash download_data.sh`

## Implementations

| File                                   | Language | Layers | Hidden Size | Epochs | Train Time     | Final Loss | Comment |
|----------------------------------------|----------|--------|-------------|--------|----------------|------------|-------|
| `01_numpy_3x512.py`                    | NumPy    | 3      | 512         | 10     | 28.7 s         | 0.156      | Readable baseline |
| `02_pytorch_3x512.py`                  | PyTorch  | 3      | 512         | 40     | 30 s        | 0.041      | Easy but slower |
| `03_pytorch_3x512_compiled.py`         | PyTorch  | 3      | 512         | 10     | 40.0 s         | 0.041      | `torch.compile` added overhead |
| `04_cuda_3layer_512_naive.cu`     | CUDA     | 3      | 512         | 40     | ~20.6 s        | 0.0116     | **Best speed/accuracy sweet spot** |
| `05_cuda_3layer_512_tiled_powerful.cu` | CUDA     | 3      | 512         | 40     | ~11 s       | ~0.009     | Fully tiled — strongest model |

### Training Time Remarks
- All files in this repo use **hidden size = 512** (3 layers).  
- When I briefly switched to **HIDDEN_SIZE = 1024**, training time went **through the roof** (33 s instead of ~13–20 s). This is expected: the dominant H×H matmul scales quadratically — 4× more work in the middle layer.
- Exact times vary **a lot** depending on your GPU architecture (RTX 30xx vs 40xx vs A100, etc.).
- I also noticed a consistent **~10 % difference** in train time between plugged-in vs battery mode on my laptop (power limit throttling).
- torch.compile shows worse performance on such a lightweight model

### Profiling (smart & automated)
I profiled everything mainly with **NVIDIA Nsight Systems (`nsys`)**.  
To save hours I wrote a **new innovative Python script** (`nsysrun.py`) that:
- Automatically runs `nsys profile` on any file
- Extracts only the key kernels (`kernel_tiled_matmul_*`, memcpy, etc.)
- Saves clean synthetic `.txt` reports directly into the `/profiles/` folder

```bash
 python3 nsysrun.py run wanted_output_txt_file_name ./05_cuda_3layer_512_tiled_powerful.o
