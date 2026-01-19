/* Host openCL 
 *
 * Plataforma/Device: clGetPlatformIDs(), clGetDeviceIDs();
 *
 * Contexto e fila: clCreateContext(), clCreateCommandQueue();
 *
 * Compilação JIT: clCreateProgramWithSource(), clBuildProgram();
 *
 * Buffers: clCreateBuffer(), clEnqueueWriteBuffer();
 *
 * Flows Obrigatórios:
 *
 * 1. Descobrir a GPU: clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, ....);
 * 2. Criar kernel a partir do arquivo .cl: ler arquivo -> string -> compilar
 * 3. Set argumentos: clSetKernelArg();
 * 4. Ler resultado: clEnqueueReadBuffer();
 *
 * clinfo para ver se é possível reconhecer a GPU
 * */

