# 📊 CUDA – Notação, Cálculo & Anatomia do Algoritmo  

> ####  Primeira experiência com programação em GPUs.


```bash

GPU
├── Streaming Multiprocessors (SMs) - dezenas
│   ├── CUDA Cores/Stream Processors - centenas por SM
│   ├── Shared Memory - memória rápida compartilhada
│   └── Registers - registradores locais
├── Global Memory - memória principal (lenta)
└── Cache Hierarchy - sistema de cache


Grid (Grade) - Todo o trabalho
└── Blocks (Blocos) - grupos de threads
    └── Threads - unidade básica de execução


Grid: Processar uma imagem 1920x1080
├── Block 0: Processar pixels 0-255
│   ├── Thread 0: Pixel 0
│   ├── Thread 1: Pixel 1
│   └── ... (256 threads)
├── Block 1: Processar pixels 256-511
└── ... (8640 blocos no total)
```


---

## 1. Objetivo
Somar **1 Mi de floats** na GPU e, **no fim**, mostrar apenas as **128 primeiras posições** em formato 8 × 16 para conferir a paralelização.
> 1 048 576 elementos e o resultado é impresso 8 × 16 para validação visual.

---

## 2. Convenções & Notação

| Símbolo | Significado | Valor aqui |
|---------|-------------|------------|
| **N** | total de elementos | 1 048 576 (1 « 20) |
| **threads por bloco** | `blockDim.x` | 128 |
| **nº de blocos** | `gridDim.x` = `(N+127)/128` | 8 192 |
| **índice global** | `i = blockIdx.x·blockDim.x + threadIdx.x` | 0 … 1 048 575 |
| **memória** | `cudaMallocManaged` (unificado) | 3 × 4 MiB |

---

## 3. Kernel – Algebricamente

```
∀ i ∈ [0, N-1] :
    z[i] ← x[i] + y[i]          // x = 35, y = 34 → z = 69
```

Código:
```cpp
__global__ void add(int n, const float *x, const float *y, float *z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)  z[i] = x[i] + y[i];
}
```

---

## 4. Fluxo de Dados (Unified Memory)

```
CPU (host)                        GPU (device)
x,y,z ← cudaMallocManaged  ────►  idem (paginado on-demand)
std::generate(x,y valor)   ────►  residente já visível
add<<<BLOCKS,128>>>(…)     ────►  kernel executa
cudaDeviceSynchronize()    ◄────  barreira global
imprime 8×16               ◄────  mesma memória
cudaFree                   ────►  liberação única
```

---

## 5. Compilação & Execução

```bash

nvcc -arch=sm_75 cuda.cu -o cuda

./cuda
```

---

## 6. Saída Esperada (GTX 1650)

```zsh
~/Projetos/GPU-Programming/cuda (main*) » make run

./cuda

[GPU]: NVIDIA GeForce GTX 1650
[Computação]: 7.5
[Multiprocessadores]: 14
[Total CUDA Cores]: 896
[Kernel]: 3.8847ms

[Threads por bloco]: 1024

69  69  69  69  69  69  69  69  69  69  69  69  69  69  69  69
69  69  69  69  69  69  69  69  69  69  69  69  69  69  69  69
69  69  69  69  69  69  69  69  69  69  69  69  69  69  69  69
69  69  69  69  69  69  69  69  69  69  69  69  69  69  69  69
69  69  69  69  69  69  69  69  69  69  69  69  69  69  69  69
69  69  69  69  69  69  69  69  69  69  69  69  69  69  69  69
69  69  69  69  69  69  69  69  69  69  69  69  69  69  69  69
69  69  69  69  69  69  69  69  69  69  69  69  69  69  69  69

```

---

## 7. Complexidade & Métricas

| Grandeza | Valor | Notação |
|----------|-------|---------|
| Work-items | 1 048 576 | Θ(N) |
| Instruções | 1 FMA / thread | Θ(1) |
| Memória lida | 2·N·4 B ≈ 8 MiB | Θ(N) |
| Memória escrita | N·4 B ≈ 4 MiB | Θ(N) |
| Tempo medido | ≈ 6.8 ms (GTX 1650) | T(N) ≈ Θ(N) |

---
