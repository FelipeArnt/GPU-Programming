#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include <cuda_runtime.h>

// macros de erro, existe a lib cudacheck_help, mas não consegui importar.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error em %s:%d – %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Kernel
__global__ void add(int n, const float *x, const float *y, float *z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)  z[i] = x[i] + y[i];    
}

// Utilitario de tempo
static double wtime(void)   /* segundos desde a época – precisão ~ms */
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

//Função para converter a capacidade de computação.

static int convertSMVer2Cores(int major, int minor)
{
    switch (major) {
    case 2: return (minor == 1) ? 48 : 32;              /* Fermi   */
    case 3: return 192;                               /* Kepler  */
    case 5: return 128;                               /* Maxwell */
    case 6: return (minor == 0) ? 64  : 128;          /* Pascal  */
    case 7: return (minor == 0 || minor == 5) ? 64 : 128; /* Volta/Turing */
    case 8: return 128;                               /* Ampere / Ada */
    case 9: return 128;                               /* Hopper  */
    case 10:return 128;                               /* Blackwell */
    default:return 0;
    }
}

// Constantes
enum { N = 1 << 20, THREADS = 256 };
enum { BLOCKS = (N + THREADS - 1) / THREADS };

// Main 
int main(void)
{
    const size_t bytes = N * sizeof(float);

    float *x, *y, *z;

// alocação unificada 
    CUDA_CHECK(cudaMallocManaged((void **)&x, bytes));
    CUDA_CHECK(cudaMallocManaged((void **)&y, bytes));
    CUDA_CHECK(cudaMallocManaged((void **)&z, bytes));


// inicialização 
    for (int i = 0; i < N; ++i) { x[i] = 35.0f;  y[i] = 34.0f; }

    CUDA_CHECK(cudaDeviceSynchronize());

// execução
    const double t0 = wtime();
    add<<<BLOCKS, THREADS>>>(N, x, y, z);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const double t1 = wtime();


// informações do dispositivo
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        const int coresPerSM = convertSMVer2Cores(prop.major, prop.minor);
        const int totalCores = coresPerSM * prop.multiProcessorCount;

        printf("\n[GPU]: %s\n", prop.name);
        printf("[Computação]: %d.%d\n", prop.major, prop.minor);
        printf("[Multiprocessadores]: %d\n", prop.multiProcessorCount);
        printf("[Total CUDA Cores]: %d\n", totalCores);
        printf("[Kernel time]: %.3fms\n", (t1 - t0) * 1e3);
        printf("[Threads por bloco]: %d\n", prop.maxThreadsPerBlock);

        /* validação */
        int ok = 1;
        for (int i = 0; i < N && ok; ++i) ok &= (z[i] == 69.0f);
        printf("\nValidação: %s\n", ok ? "OK\n" : "ERROR");

        /* amostra 8×16 */
        for (int lin = 0; lin < 8; ++lin) {
            for (int col = 0; col < 16; ++col)
                printf("%4d", (int)z[lin * 16 + col]);
            putchar('\n');
        }
    }

// cleanup
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
    CUDA_CHECK(cudaFree(z));
    return 0;
}
