; Built-In Functions ( get_global_id(0) ), ( get_local_id(0) )
; Tipos de memória ( __global, __local, __constant)
; Vetorização ( float4 no lugar de struct Point)
; Kernel em arquivo separado faz com que o host leia ele em tempo real como uma string.


"""__kernel void somaKernel(__global const float *a,
                         __global const float *b,
                         __global float *c){
    int idx = get_global_id(0);
    c[idx] = a[idx] + b[idx];
}
"""

