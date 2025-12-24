# 📊 CUDA--Table – Notação, Cálculo & Anatomia do Algoritmo

Pequeno projeto de programaap em GPUs: Foram somados 128 números na GPU e o resultado foi impresso no formato de **tabela 8 × 16** para exibir a paralelização.

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

## 3. Mapeamento Índice ⇨ Tabela 2-D

Linha-major (row-major):

```
índice 1-D:   0  1  2 … 15 | 16 … 31 | … | 112 … 127
tabela 2-D:  linha 0     | linha 1  | … | linha 7
```

Fórmula de conversão:

```
lin = i / 16        (divisão inteira)
col = i % 16        (resto)
```

Por isso o laço de impressão é:

```cpp
for (lin = 0 … 7)
    for (col = 0 … 15)
        print h_z[lin*16 + col]
```

---

## 4. Fluxo de Dados (esquema textual)

```
CPU (HOST)                     GPU (DEVICE)
┌-------------┐               ┌-------------┐
│ h_x = 1.0   │──cudaMemcpy──►│ d_x         │
│ h_y = 2.0   │──cudaMemcpy──►│ d_y         │
│             │               │ d_z         │
│ h_z (vazio) │◀--cudaMemcpy--│ d_z ← add() │
└-------------┘               └-------------┘
```

---

## 5. Kernel – Álgebricamente

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

## 6. Complexidade & Métricas

| Grandeza | Valor | Notação |
|----------|-------|---------|
| **Work-items** | 128 | O(N) |
| **Instruções** | 1 add / thread | O(1) por thread |
| **Memória lida** | 2·N·4 B = 1 024 B | Θ(N) |
| **Memória escrita** | N·4 B = 512 B | Θ(N) |
| **Tempo medido** | ≈ 4 ms (GTX 1650) | T(N) = Θ(N) |

---

## 7. Warm-up & Sincronização

- `cudaDeviceSynchronize()` após o kernel = **barreira global** – CPU só prossegue quando **todas as threads** terminaram.  
- Sem ela o cronômetro mediria **só o lançamento**, não a execução.

---

## 8. Visual

- 128 = 2⁷ → fatoração 2⁴ × 2³ = 16 × 8 gera **tabela quadrada visualmente agradável**.  
- Facilita verificar de relance se **todos os elementos** estão corretos (tudo 3).

---

## 9. Possíveis Variações Didáticas

| Alteração | Aprendizado |
|-----------|-------------|
| `N = 1 024` | Escalabilidade |
| `threads = 32` | Exato tamanho de **warp** |
| `__shared__ float buf[256]` | Introduz **memória local** |
| `atomicAdd(&z[0], 1)` | Redução e **concorrência** |
| `cudaMallocManaged` | **Unified Memory** – zero cópias |

