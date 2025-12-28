#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>


int main()
{
  
  cl_platform_id *platforms;
  cl_uint num_platforms;

  clGetPlatformIDs(0, NULL, &num_platforms);
  platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
  clGetPlatformIDs(num_platforms, platforms, NULL);

  printf("Plataformas OpenCL  %d\n", num_platforms);

  //iterando pelo numero de plataformas encontradas
  for(int i = 0; i < num_platforms; i++)
  {
    char buffer[1024];
    clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 1024, buffer, NULL);
    printf("Plataforma %d: %s\n", i,buffer);

    cl_device_id *devices;
    cl_uint num_devices;
    
    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

    for (int j = 0; j < num_devices; j++)
      {
      clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 1024, buffer, NULL);
      printf("Dispositivo %d: %s\n", j, buffer);
      }
    free(devices);
  }
  free(platforms);
  return 0;
}
