#include <iostream>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>

// libs utilizadas para formatação da saída, medição de tempo e a API CUDA.
//
// Macro CUDA_CHECK encapsula qualquer chamada/call CUDA. Se falhar, imprime (__FILE__) (__LINE__), a mensagem de erro e dps aborta.

#define CUDA_CHECK(call)  
