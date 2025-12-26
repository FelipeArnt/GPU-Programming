## 1. TO-DO 

Próximos passos do projeto GPU-Programming em CUDA.

### CUDA
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- https://docs.nvidia.com/cuda/cuda-runtime-api/

# KERNEL
- https://sysprog21.github.io/lkmpg/#introduction

---

| Alteração | Aprendizado |
|-----------|-------------|
| `N = 1 024` | Escalabilidade |
| `threads = 32` | Exato tamanho de **warp** |
| `__shared__ float buf[256]` | Introduz **memória local** |
| `atomicAdd(&z[0], 1)` | Redução e **concorrência** |
| `cudaMallocManaged` | **Unified Memory** – zero cópias |

---

| Tarefa | O que aprenderá |
|--------|-----------------|
| Altere `N` para 1 048 576 | Grande escalabilidade |
| Troque `threads` 256 → 512 / 1024 | Escolha ideal de bloco |
| Use `cudaMallocManaged` | Unified Memory (menos cópias) |
| Adicione `__shared__ float buf[256]` | Memória compartilhada |
| Troque `add` por `axpy` (y = a*x + y) | BLAS nível 1 |

---

## 📚 Material de referência

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)  
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)

