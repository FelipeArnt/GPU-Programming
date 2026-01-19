/* Implementação CUDA */

/* Soma de vetores em CUDA*/
__global__ void somaKernel(float *a, float *b, float *c, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) c[idx] = a[idx] + b[idx];
}



