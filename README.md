# 📊 CUDA – Notação, Cálculo & Anatomia do Algoritmo

Pequeno projeto de programação em GPUs: Soma de 128 números na GPU e geração do resultado no formato de **tabela 8 × 16** para exibir a paralelização.

---

## 📦 O que está aqui

| Arquivo | Descrição |
|---------|-----------|
| `cuda-hello-table.cu` | Código-fonte completo (C++17 + CUDA) |
| `Makefile` | Compila com um comando só |
| `README.md` | Este arquivo |

---

## 🎯 Objetivo do código

1. Aloja **128 floats** na CPU e na GPU.  
2. Preencche `x = 1.0`, `y = 2.0`.  
3. Executa kernel `add<<<blocks, 256>>>` – **1 thread por elemento**.  
4. Devolve o vetor `z = x + y` (valor 3 em todas as posições).  
5. Imprime **8 linhas × 16 colunas** alinhadas.  
6. Mostra **tempo de execução total** (alocação → cópia → kernel → cópia → print).

---

## 🔧 Requisitos

- GPU NVIDIA com Compute Capability ≥ 3.5  
- CUDA Toolkit instalado (provê `nvcc`)  
- Compilador C++ (g++ ou clang)  


## 🚀 Compilação & execução

Compilar:
```bash
nvcc -arch=sm_75 -std=c++17 -O3 cuda-hello-table.cu -o cuda-hello-table
./cuda-hello-table
```


## 🖥️ Saída esperada

```
Resultado: 
   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3
   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3
   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3
   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3
   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3
   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3
   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3
   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3
[INFO]: 4 ms
```

---


| Tarefa | O que aprenderá |
|--------|-----------------|
| Altere `N` para 1 048 576 | Grande escalabilidade |
| Troque `threads` 256 → 512 / 1024 | Escolha ideal de bloco |
| Use `cudaMallocManaged` | Unified Memory (menos cópias) |
| Adicione `__shared__ float buf[256]` | Memória compartilhada |
| Troque `add` por `axpy` (y = a*x + y) | BLAS nível 1 |



---

- **Cálculo**: soma elemento-a-elemento.  
- **Notação**: índices 1-D mapeados em 2-D por `lin = i / cols`, `col = i % cols`.  
- **Complexidade**: Θ(N) trabalho, Θ(N) tráfego de memória.  
- **Visual**: tabela 8 × 16 = confirmação instantânea de correção.


## 1. Cálculo

> **z = x + y**, onde **x = 1.0**, **y = 2.0** → **z = 3.0** em **todas as 128 posições**.

---

## 2. Notação & Convenções

| Símbolo | Significado | Valor aqui |
|---------|-------------|------------|
| **N** | total de elementos (threads) | 128 |
| **threads por bloco** | `blockDim.x` | 256 |
| **nº de blocos** | `gridDim.x` = `(N + 255) / 256` | 1 |
| **índice global** | `int i = blockIdx.x * blockDim.x + threadIdx.x` | 0 … 127 |
| **índice local** | `threadIdx.x` | 0 … 255 (mas só 0-127 válido) |

---

## 3. Fluxo de Dados 

```
CPU (HOST)                     GPU (DEVICE)
┌-------------┐               ┌-------------┐
│ h_x = 1.0   │──cudaMemcpy──►│ d_x         │
│ h_y = 2.0   │──cudaMemcpy──►│ d_y         │
│             │               │ d_z         │
│ h_z (vazio) │◀-cudaMemcpy-►│ d_z ← add() │
└-------------┘               └-------------┘
```

---

## 4. Kernel – Algebricamente

Kernel `add`:

```
∀ i ∈ [0, N − 1] :
    z[i] ← x[i] + y[i]
```

Implementação SIMT:

```
i ← blockIdx·blockDim + threadIdx
if i < N :
    z[i] ← x[i] + y[i]
```

A condição `if` evita **out-of-bounds** quando `N` não é múltiplo de `blockDim`.

---

## 5. Complexidade & Métricas

| Grandeza | Valor | Notação |
|----------|-------|---------|
| **Work-items** | 128 | O(N) |
| **Instruções** | 1 add / thread | O(1) por thread |
| **Memória lida** | 2·N·4 B = 1 024 B | Θ(N) |
| **Memória escrita** | N·4 B = 512 B | Θ(N) |
| **Tempo medido** | ≈ 4 ms (GTX 1650) | T(N) = Θ(N) |

---

## 6. Warm-up & Sincronização

- `cudaDeviceSynchronize()` após o kernel = **barreira global** – CPU só prossegue quando **todas as threads** terminaram.  
- Sem ela o cronômetro mediria **só o lançamento**, não a execução.

---

## 7. Visual

- 128 = 2⁷ → fatoração 2⁴ × 2³ = 16 × 8 gera **tabela quadrada visualmente agradável**.  
- Facilita verificar de relance se **todos os elementos** estão corretos.
---


