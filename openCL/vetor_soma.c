#include <stdio.h>
#include <stdlib.h>
// Usando versões antigas da QUEUE apenas para teste 
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <time.h>


// src do kernel
const char *kernel_source = 
"__kernel void add(__global const float *A, \n"
"                         __global const float *B, \n"
"                         __global float *C, \n"
"                         const unsigned int size) { \n"
"    int id = get_global_id(0); \n"
"    if (id < size) C[id] = A[id] + B[id]; \n"
"}";

//entry point
int main()
{
    const int SIZE = 1000000;
//Allocando    
    float *A = malloc(SIZE * sizeof(float));
    float *B = malloc(SIZE * sizeof(float));
    float *C = malloc(SIZE * sizeof(float));


//Iniciando os dados

    for (int i = 0; i < SIZE; i++){
        A[i] = 35.0f;
        B[i] = 34.0f;   
    }

//openCL setup
    cl_platform_id plataform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem bufferA, bufferB, bufferC;


// Obter dispositivos: GTX 1650
    clGetPlatformIDs(1, &plataform, NULL);
    clGetDeviceIDs(plataform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);

// Criando buffers
    bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE * sizeof(float), NULL, NULL);
    bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE * sizeof(float), NULL, NULL);
    bufferC = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE * sizeof(float), NULL, NULL);

    clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, SIZE * sizeof(float), A, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, SIZE * sizeof(float), B, 0, NULL, NULL);

// Criando e compilando o kernel

    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "add", NULL);

// Configurando argumentos

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    clSetKernelArg(kernel, 3, sizeof(int), &SIZE);

// Executando
    
    size_t local_size = 256;
    size_t global_size = ((SIZE + local_size - 1) / local_size) * local_size;


    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    clFinish(queue);

// Lendo resultado

    clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, SIZE * sizeof(float), C, 0, NULL, NULL);

    printf("\n");
    for (int lin = 0; lin < 8; lin++)
      for(int col = 0; col < 16; col++)
          printf("%4d", (int)C[lin * 16 + col]);
      putchar('\n');

// Limpando buffers, kernel, program, ....

    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(A); free(B); free(C);
    return 0;
}
