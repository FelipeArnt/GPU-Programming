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


## Tópicos a serem estudados no arquivo geometria.h

 Definir estruturas de dados que serão utilizadas por todos backends.

O que estudar:
- **`struct Point`**: {float x,y,z;}
- **`struct Triangle`**: {Point v[3];}
- **`struct Mesh`**: {Triangle*faces; int num_faces;}
- **`Alinhamento de memória`**: Usar __attribute__((aligned(16))) para vetorização.

> Para entender melhor, desenhar cubo manualmente -> 12 Triângulos e 8 vértices.


---- 


Recursos de Estudo

- CUDA: "CUDA C++ Programming Guide" (docs.nvidia.com)
- OpenCL: "OpenCL 3.0 Reference Card" (khronos.org)
- Benchmarking: perf (Linux) para cache misses
- Livro: "Structured Parallel Programming" - Patterns para CUDA/OpenCL
- Dica: Comece com vetores pequenos (1024 elementos) e vá aumentando até 2^28.

