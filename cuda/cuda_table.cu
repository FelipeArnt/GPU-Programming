#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>


#define N 2048     // 8 × 16

__global__
void add(int n, const float *x, const float *y, float *z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = x[i] + y[i];
    printf("[GPU]:[block %d]-[thread %d]\n",blockIdx.x, threadIdx.x);
}

int main()
{
  // Tempo de exucução
  auto start = std::chrono::high_resolution_clock::now();

  size_t bytes = N * sizeof(float);

    // 1) aloca e preenche HOST
    float *h_x = new float[N];
    float *h_y = new float[N];
    float *h_z = new float[N];

    for (int i = 0; i < N; ++i) {
        h_x[i] = 35.0f;
        h_y[i] = 34.0f;
    }

    // 2) aloca DEVICE
    float *d_x, *d_y, *d_z;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMalloc(&d_z, bytes);

    // 3) copiando Host -> Device
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);

    // 4) executa kernel
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    add<<<blocks, threads>>>(N, d_x, d_y, d_z);

    cudaDeviceSynchronize();

    // 5) copiando Device -> Host
    cudaMemcpy(h_z, d_z, bytes, cudaMemcpyDeviceToHost);

    // 6) imprime tabela 8 × 16
  std::cout << "GPU Programming : \n";
    for (int lin = 0; lin < 8; ++lin) {
        for (int col = 0; col < 16; ++col)
            std::cout << std::setw(4) << static_cast<int>(h_z[lin * 16 + col]);
        std::cout << '\n';
    }

    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "[INFO]:  " << duration.count() << " ms" << std::endl;


    // 7) cleanup p finalizar
    delete [] h_x;
    delete [] h_y;
    delete [] h_z;
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
}
