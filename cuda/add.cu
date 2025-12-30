#include <stdio.h>
#include <cuda_runtime.h>


__global__ void vectorAdd(int* a, int* b, int* c)
{
  int i = threadIdx.x;
  
  c[i] = a[i] + b[i];

}

int main()
{

  int a [] = {1,2,3};
  int b [] = {4,5,6};
  int c [sizeof(a) / sizeof(int)] = {0};
  
  // Criando ponteiros na GPU
  int* cudaA = 0;
  int* cudaB = 0;
  int* cudaC = 0;

  //Alocando memória na GPU
  cudaMallocManaged(&cudaA,sizeof(a));
  cudaMallocManaged(&cudaB,sizeof(b));
  cudaMallocManaged(&cudaC,sizeof(c));

  // Copiando os vetores
  cudaMemcpy(cudaA, a, sizeof(a), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaB, b, sizeof(b), cudaMemcpyHostToDevice);


  vectorAdd <<< 1, sizeof(a) / sizeof(int) >>> (cudaA, cudaB, cudaC);

  for (int lin = 0; lin < 8; lin++ ){
    for (int col = 0; col < 16; col++)
    {
      printf("%4d", (int)c[lin * 16 + col]);
    }
  }
}
