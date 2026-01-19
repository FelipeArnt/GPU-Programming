

typedef struct {
  double tempo_cuda;
  double tempo_opencl;

  float speedup; /*Ratio openCL e CUDA*/
  bool resultados /*Verificação de correção*/

} BenchmarkResultado;

void benchmarkAmbos(int data_size, int iteracoes);


