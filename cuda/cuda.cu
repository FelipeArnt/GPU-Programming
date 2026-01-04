#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

// macros de erro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error em %s:%d – %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Kernel com grid-stride loop
__global__ void add(int n, const float *x, const float *y, float *z)
{
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        z[i] = x[i] + y[i];
    }
}

// Converte SM -> núcleos por SM
static int convertSMVer2Cores(int major, int minor)
{
    switch (major) {
    case 2:  return (minor == 1) ? 48 : 32;                // Fermi
    case 3:  return 192;                                   // Kepler
    case 5:  return 128;                                   // Maxwell
    case 6:  return (minor == 0) ? 64  : 128;              // Pascal
    case 7:  return (minor == 0 || minor == 5) ? 64 : 128; // Volta/Turing
    case 8:  return 128;                                   // Ampere / Ada
    case 9:  return 128;                                   // Hopper
    case 10: return 128;                                   // Blackwell
    default: return 0;
    }
}

// imprime propriedades do dispositivo + tempo de kernel + config usada
void dispositivo(double kernel_ms, int threadsPerBlock, int blocksPerGrid)
{
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        int coresPerSM = convertSMVer2Cores(prop.major, prop.minor);
        int totalCores = coresPerSM * prop.multiProcessorCount;

        printf("\n[GPU]: %s\n", prop.name);
        printf("[Computação]: %d.%d\n", prop.major, prop.minor);
        printf("[Multiprocessadores]: %d\n", prop.multiProcessorCount);
        printf("[Total CUDA Cores]: %d\n", totalCores);
        printf("[Max threads por bloco]: %d\n", prop.maxThreadsPerBlock);
        printf("[Max threads por SM]: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("[Kernel time (CUDA events)]: %.3f ms\n", kernel_ms);
    }
}

// inicializa vetores
void init_vetores(float *x, float *y, int n)
{
    for (int i = 0; i < n; ++i) {
        x[i] = 35.0f;
        y[i] = 34.0f;
    }
}

// valida resultado
int validacao(const float *z, int n, float esperado)
{
    for (int j = 0; j < n; ++j) {
        if (z[j] != esperado) return 0;
    }
    return 1;
}

// imprime uma tabela 8x16 da saída
void tabela(const float *z)
{
    for (int lin = 0; lin < 8; ++lin) {
        for (int col = 0; col < 16; ++col) {
            printf("%4d", (int)z[lin * 16 + col]);
        }
        putchar('\n');
    }
}

// executa o kernel e retorna tempo em ms (CUDA events)
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

// Main
int main(void)
{
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    float *x, *y, *z;

    // alocação unificada
    CUDA_CHECK(cudaMallocManaged((void **)&x, bytes));
    CUDA_CHECK(cudaMallocManaged((void **)&y, bytes));
    CUDA_CHECK(cudaMallocManaged((void **)&z, bytes));

    init_vetores(x, y, N);

    // pegar propriedades do device (GTX 1650)
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // threads por bloco: múltiplo de warp, sem exceder maxThreadsPerBlock
    int threadsPerBlock = 256;
    if (threadsPerBlock > prop.maxThreadsPerBlock)
        threadsPerBlock = prop.maxThreadsPerBlock;

    // blocks por grid
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    double kernel_ms = run_kernel(N, x, y, z, threadsPerBlock, blocksPerGrid);

    int ok = validacao(z, N, 69.0f);
    printf("\nValidação: %s\n", ok ? "OK" : "ERROR");
    tabela(z);
    dispositivo(kernel_ms, threadsPerBlock, blocksPerGrid);

    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
    CUDA_CHECK(cudaFree(z));
    return 0;
}
