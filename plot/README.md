# Plotando Funções matemáticas e formas geométricas



```text
gpu-benchmark/
├── common/
│   ├── geometry.h          # Estruturas de dados
│   └── plot_functions.h    # Funções matemáticas
├── cpu/
│   └── renderer_cpu.c      # Implementação sequencial
├── cuda/
│   ├── renderer_cuda.cu    # Kernel e host code CUDA
│   └── Makefile
├── opencl/
│   ├── renderer_cl.c       # Host code OpenCL
│   ├── renderer_cl_kernel.cl  # Kernel OpenCL
│   └── Makefile
└── benchmark/
    └── main.c              # Orquestração e medição
```
<!---->
