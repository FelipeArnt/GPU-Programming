    #include <iostream>
    #include <cmath>
    #include <iomanip>
    #include <chrono>
    #include <cuda_runtime.h>
    #include <algorithm>

    // IOSTREAM e IOMANIP ==> formatação da saída, 
    // CHRONO ==> medição de tempo,
    // CUDA_RUNTIME ==> API CUDA.
    // ALGORITHM ==>  ConvertSMVer2Cores

    // Macro CUDA_CHECK encapsula qualquer chamada/call CUDA. Se falhar, imprime (__FILE__) (__LINE__), a mensagem de erro e dps aborta.
    #define CUDA_CHECK(call) \
        do { \
            cudaError_t err = call; \
            if (err != cudaSuccess) { \
                std::cerr << "CUDA Error em " << __FILE__ << ":" << __LINE__ \
                        << " - " << cudaGetErrorString(err) << std::endl; \
                exit(EXIT_FAILURE); \
            } \
        } while (0)

    // Interessante a comparação do tempo de execução do kernel com e sem o printf(GPU, block, thread, blockDim)
    __global__
    void add(int n, const float *x, const float *y, float *z)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) z[i] = x[i] + y[i];
    }

    // Função para converter a capacidade de computação para o numero de COREs por SM.
    int ConvertSMVer2Cores(int major, int minor) {
        switch (major) {
            case 2: return (minor == 1) ? 48 : 32; // Fermi
            case 3: return 192; // Kepler
            case 5: return 128; // Maxwell
            case 6: return (minor == 0) ? 64 : 128; // Pascal
            case 7: return (minor == 0 || minor == 5) ? 64 : 128; // Volta and Turing
            case 8: return (minor == 0) ? 64 : ((minor == 6 || minor == 9) ? 128 : 128); // Ampere and Ada Lovelace
            case 9: return 128; // Hopper
            case 10: return 128; // Blackwell
            default: return 0; // Unknown device type
        }
    }

    constexpr int N = 1 << 20;          // 1 << 20 = 1Mi elementos
    constexpr int THREADS = 128;        //  128/256/512/1024
    constexpr int BLOCKS  = (N + THREADS - 1) / THREADS;

    int main(int argc, char** argv)
    {
        const size_t bytes = N * sizeof(float);

        float *x, *y, *z;

        CUDA_CHECK(cudaMallocManaged(&x, bytes));
        CUDA_CHECK(cudaMallocManaged(&y, bytes));
        CUDA_CHECK(cudaMallocManaged(&z, bytes));

        std::generate(x, x + N, [](){ return 35.0f; });
        std::generate(y, y + N, [](){ return 34.0f; });
               
        CUDA_CHECK(cudaDeviceSynchronize());
        
        auto t0 = std::chrono::steady_clock::now();

        add<<<BLOCKS, THREADS>>>(N, x, y, z);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        
    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
            
        int coresPerSM = ConvertSMVer2Cores(prop.major, prop.minor);
        int totalCores = coresPerSM * prop.multiProcessorCount;
        
        std::cout << "[GPU]: "<< prop.name << std::endl;
        std::cout << "[Computação]: " << prop.major << "." << prop.minor << "" << std::endl;
        std::cout << "[Multiprocessadores]: " << prop.multiProcessorCount << std::endl;
        std::cout << "[Total CUDA Cores]: " << totalCores << std::endl;
        std::cout << "[Kernel]: " << ms << "ms" << std::endl;
        std::cout << "\n[Threads por bloco]: " << prop.maxThreadsPerBlock << std::endl;


        // validação ==> Caso todas posições sejam preenchidas com 69 (35 + 34), 
        bool ok = true;
        for (int idx = 0; idx < N; ++idx) ok &= (z[idx] == 69.0f);
        for (int lin = 0; lin < 8; ++lin) {
            for (int col = 0; col < 16; ++col)
                std::cout << std::setw(4) << static_cast<int>(z[lin * 16 + col]);
            std::cout << '\n';
            }    
        }
        
        cudaFree(x); cudaFree(y); cudaFree(z);
    }