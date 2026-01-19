/* Implementação CUDA 
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

/* Soma de vetores em CUDA*/
__global__ void somaKernel(float *a, float *b, float *c, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) c[idx] = a[idx] + b[idx];
}



