# Repositório com Projetos relacionados a programação em GPUs.

- A pasta **`cuda`** contém o primeiro projeto relacionado a GPUs, primeiro contato com o assunto, onde foi implementado uma soma de vetores `x[35]` e `y[34]` 1 Mi de vezes.
- A pasta **`plot`** contém o projeto de comparação entre openCL e CUDA para plotar gráficos do tempo de execução de ambos.

Todos os projetos tem o objetivo de aprimorar o meu conhecimento no assunto, assim como noções básicas da programação paralelizada. Os projetos estão em fase de desenvolvimento, cada um deles é separado em aprendizados por semana. 

## Cuda
- <!--TODO-->
## Plot
Para o projeto de plot, serão 4 semanas de estudo com implementações graduais de cada parte do código.
- Semana 1: Implementações de CPU e geração de malhas.
- Semana 2: Adição CUDA (kernel simples, mas sem otimizações)
- Semana 3: Adição OpenCL (reaproveitando lógica do CUDA)
- Semana 4: Otimizações avançadas:
  - CUDA: Streams, pinned memory, UVA;
  - OpenCL: Work-group sizes, local memory;

### Recursos de Estudo
- CUDA: "CUDA C++ Programming Guide" (docs.nvidia.com)
- OpenCL: "OpenCL 3.0 Reference Card" (khronos.org)
- Benchmarking: perf (Linux) para cache misses
- Livro: "Structured Parallel Programming" - Patterns para CUDA/OpenCL

> Dica: Começar com vetores pequenos (1024 elementos) e ir aumentando até 2^28.

### Anotações

- **main.c**

```c
/* [ Orquestração e Medição ]
 * ---------------------------------------------------------------------------
 * Repetições: Executar cada teste 10x, tirar a média e desvio padrão.
 * Warming up: Descartar primeira iteração (cache/kernel compilation)
 * Validação: Comparar resultados CPU vs GPU (assert() ou diferença tolerância;)
 * -----------------------------------------------------------------------------
 * [ Métricas a coletar ] 
 * Tempo médio de execução.
 * SpeedUp: tempo_cpu / tempo_gpu
 * GB/s (bytes lidos/bytes escritos)
 * Para gerar gráficos de barras de CUDA vs OpenCL, usar gnuplot.
 * -----------------------------------------------------------------------------
*/

```

- **cpu_renderizador.c**

```c
/*  [ Baseline sequencial para comparação ]
 *
 * malloc() para alocar malha.;
 * Loop (for int i=0; i<num_faces; i++);
 * Cache locality: Acessar dados sequencialmente.
 * Otimizações GCC: flags -O3, -ffast-math
 * 
 * Exercício: clock_gettime(CLOCK_MONOTONIC....)
 * */
```

- **cuda_renderizador.cu**

```c
/* [ Implementação CUDA ]  
 *
 * Memory hierarchy: __global__, __shared__, __constant__;
 *
 *  Grid / Block configuration: kernel <<< blocks, threads>>>(....)
 *  CUDA Math API: sinf(), cosf() (Versão GPU)
 *  Atomics: atomicAdd() para contador de inclusão
 *
 *  Passo a passo:
 *  Aloca memória na GPU: cudaMalloc();
 *  Copia dados: cudaMemcpy(host -> device)
 *  Executa kernel: computeFaces<<<num_faces / 256 + 1, 256>>>(....)
 *  Sincroniza: cudaDeviceSyncronize()
 *  Libera memória: cudaFree()
 *  Debugger: Usar cuda-gdb e printf dentro do kernel para debug.
 * */
```

- **renderizador_cl.c**

```c
/* [ Host openCL ]
 *
 * Plataforma/Device: clGetPlatformIDs(), clGetDeviceIDs();
 *
 * Contexto e fila: clCreateContext(), clCreateCommandQueue();
 *
 * Compilação JIT: clCreateProgramWithSource(), clBuildProgram();
 *
 * Buffers: clCreateBuffer(), clEnqueueWriteBuffer();
 *
 * Flows Obrigatórios:
 *
 * 1. Descobrir a GPU: clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, ....);
 * 2. Criar kernel a partir do arquivo .cl: ler arquivo -> string -> compilar
 * 3. Set argumentos: clSetKernelArg();
 * 4. Ler resultado: clEnqueueReadBuffer();
 *
 * clinfo para ver se é possível reconhecer a GPU
 * */
```

- **renderizador_cl_kernel.cl**

```c
/*
* Built-In Functions ( get_global_id(0) ), ( get_local_id(0) )
* Tipos de memória ( __global, __local, __constant)
* Vetorização ( float4 no lugar de struct Point)
* Kernel em arquivo separado faz com que o host leia ele em tempo real como uma string.
*/

```
