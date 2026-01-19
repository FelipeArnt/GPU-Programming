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
