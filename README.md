## 📊 CUDA – Notação, Cálculo & Anatomia do Algoritmo

> #### Primeira experiência com programação em GPUs (GTX 1650, Unified Memory + grid-stride loop).

---

## 1. Arquitetura & Organização

```bash
GPU
├── Streaming Multiprocessors (SMs) - dezenas
│   ├── CUDA Cores/Stream Processors - centenas por SM
│   ├── Shared Memory - memória rápida compartilhada
│   └── Registers - registradores locais
├── Global Memory   - memória principal (mais lenta)
└── Cache Hierarchy - L1/L2, etc.

Grid (Grade) - Todo o trabalho
└── Blocks (Blocos) - grupos de threads
    └── Threads     - unidade básica de execução
```

Com **grid-stride loop**, cada thread pode processar **vários elementos** espaçados por um `stride`:

```c
__global__ void add(int n, const float *x, const float *y, float *z) {
    
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        z[i] = x[i] + y[i];
    }
}
```  

Isso torna o kernel flexível para qualquer combinação de `blocks/grid` e `threads/block`.

---

## 2. Objetivo

Somar **1 Mi de floats** na GPU usando Unified Memory e verificar a paralelização:

- `x[i] = 35`, `y[i] = 34` → `z[i] = 69` para `0 ≤ i < N`.
- Ao final, imprimir as **128 primeiras posições** em formato **8 × 16** para validação visual.

Parâmetros usados:

| Símbolo              | Significado            | Valor aqui                 |
|----------------------|------------------------|----------------------------|
| **N**                | total de elementos     | 1 048 576 (`1 << 20`)      |
| **threads/block**    | `blockDim.x`           | 256 (ajustado ao device)   |
| **blocks/grid**      | `gridDim.x`            | `(N + 255) / 256`          |
| **alocação**         | `cudaMallocManaged`    | 3 vetores de 4 MiB         |
| **tempo do kernel**  | medido com eventos     | ~3–7 ms na GTX 1650        |  

Na inicialização, o código ainda consulta as propriedades do device (`cudaGetDeviceProperties`) para garantir que `threadsPerBlock` não excede `maxThreadsPerBlock`.

---

## 3. Kernel – Forma Algébrica

Formalmente:

```text
∀ i ∈ [0, N-1] :
    z[i] ← x[i] + y[i]
```

Implementação com grid-stride loop:

```c

__global__ void add(int n, const float *x, const float *y, float *z) {
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        z[i] = x[i] + y[i];
    }
}
```  

- `idx` é o índice global inicial da thread.  
- `stride = blockDim.x * gridDim.x` é o “passo” entre elementos processados pela mesma thread.


## 4. Fluxo de Dados (Unified Memory)

O código usa **Unified Memory** com `cudaMallocManaged`, de forma que host e device enxergam os mesmos ponteiros.

```text

CPU (host)                          GPU (device)
x,y,z ← cudaMallocManaged  ──────►  memória unificada, páginas migradas sob demanda
init_vetores(x,y)          ──────►  dados já prontos para o kernel
run_kernel(...)             ──────►  add<<<grid,block>>> soma x e y em z
cudaDeviceSynchronize()     ◄──────  garante término do kernel
tabela(z)                   ◄──────  leitura direta do mesmo ponteiro
cudaFree(x,y,z)             ──────►  desalocação única
```

Não há `cudaMemcpy` explícito; o driver migra páginas conforme uso.

---

## 5. Medição de Desempenho

A versão final mede **somente o tempo do kernel** usando **eventos CUDA**:

```c
double run_kernel(int n, float *x, float *y, float *z, int threadsPerBlock, int blocksPerGrid)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    add<<<blocksPerGrid, threadsPerBlock>>>(n, x, y, z);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return (double)ms;
}
```

O `main` calcula `threadsPerBlock` (256, limitado por `prop.maxThreadsPerBlock`) e `blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock`, passa isso para `run_kernel`, valida o resultado e imprime uma amostra 8×16.

***

## 6. Saída Esperada (GTX 1650)

Exemplo típico de saída:

```bash
Validação: OK
  69  69  69  69  69  69  69  69  69  69  69  69  69  69  69  69
  69  69  69  69  69  69  69  69  69  69  69  69  69  69  69  69
  69  69  69  69  69  69  69  69  69  69  69  69  69  69  69  69
  69  69  69  69  69  69  69  69  69  69  69  69  69  69  69  69
  69  69  69  69  69  69  69  69  69  69  69  69  69  69  69  69
  69  69  69  69  69  69  69  69  69  69  69  69  69  69  69  69
  69  69  69  69  69  69  69  69  69  69  69  69  69  69  69  69
  69  69  69  69  69  69  69  69  69  69  69  69  69  69  69  69

[GPU]: NVIDIA GeForce GTX 1650
[Computação]: 7.5
[Multiprocessadores]: 14
[Total CUDA Cores]: 896
[Max threads por bloco]: 1024
[Max threads por SM]: 1024
[Kernel time (CUDA events)]: ~4.049 ms

```

Os valores de tempo dependem da máquina, mas na GTX 1650 ficam tipicamente na faixa de poucos milissegundos para 1 Mi de elementos.

---

## 7. Complexidade & Métricas

| Grandeza          | Valor aproximado           | Notação      |
|-------------------|---------------------------|---------------|
| Work-items        | N = 1 048 576             | Θ(N)          |
| Operações         | 1 soma por elemento       | Θ(N)          |
| Memória lida      | 2 · N · 4 B ≈ 8 MiB       | Θ(N)          |
| Memória escrita   | N · 4 B ≈ 4 MiB           | Θ(N)          |
| Tempo de kernel   | medido com eventos CUDA   | T(N) ≈ Θ(N)   |