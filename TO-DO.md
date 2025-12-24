## 1. TO-DO 

Próximos passos do projeto GPU-Programming em CUDA.

---

| Alteração | Aprendizado |
|-----------|-------------|
| `N = 1 024` | Escalabilidade |
| `threads = 32` | Exato tamanho de **warp** |
| `__shared__ float buf[256]` | Introduz **memória local** |
| `atomicAdd(&z[0], 1)` | Redução e **concorrência** |
| `cudaMallocManaged` | **Unified Memory** – zero cópias |

---


