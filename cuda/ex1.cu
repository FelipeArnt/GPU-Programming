#include <stdio.h>
#include <cuda_runtime.h>

__global__ void reduce_sum(const float *g_in, float *g_out, int N) {
    extern __shared__ float sdata[];

    unsigned int tid  = threadIdx.x;
    unsigned int idx  = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;
    if (idx < N)              sum += g_in[idx];
    if (idx + blockDim.x < N) sum += g_in[idx + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    // redução em árvore
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_out[blockIdx.x] = sdata[0];
    }
}

int main() {
    const int N = 1 << 20;
    size_t size = N * sizeof(float);

    float *h_in = (float*)malloc(size);
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);

    int threads = 256;
    int blocks  = (N + threads * 2 - 1) / (threads * 2);
    cudaMalloc(&d_out, blocks * sizeof(float));

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    reduce_sum<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    float *h_out = (float*)malloc(blocks * sizeof(float));
    cudaMemcpy(h_out, d_out, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float total = 0.0f;
    for (int i = 0; i < blocks; ++i) total += h_out[i];
    printf("Soma = %f (esperado %d)\n", total, N);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}

