#include <sys/time.h>
#include <stdio.h>
#include <math.h> //Include standard math library containing sqrt.

#define TPB 128    // Specify threads per block

// Timer
double get_cpu_millisecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec*1.e6 + (double)tp.tv_usec);
}

// A scaling function to convert integers 0,1,...,N-1 to evenly spaced floats
__device__ float scale(int i, long n)
{
  return ((float)i*1.0)/(n - 1);
}

// Compute the distance between 2 points on a line.
__device__ float distance(float x1, float x2)
{
  return sqrt((x2 - x1)*(x2 - x1));
}

__global__ void distanceKernel(float *d_out, float ref, long len)
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const float x = scale(i, len);
  d_out[i] = distance(x, ref);
}

int main(int argc, char** argv)
{
  // Specify a constant value for array length.
  long N = atol(argv[1]);
  // Choose a reference value from which distances are measured.
  const float ref = 0.5; 

  // Allocate the output array in device memory
  float *d_out = 0;
  cudaMalloc(&d_out, N*sizeof(float));

  // Allocate the output array in host memory
  float *h_out = 0;
  h_out = (float *) calloc(sizeof(float), N);

  double iStart = get_cpu_millisecond();
  // Launch kernel to compute and store distance values
  distanceKernel<<<N/TPB, TPB>>>(d_out, ref, N);
  cudaDeviceSynchronize();
  double iElaps = get_cpu_millisecond() - iStart;
  

  // copy output from device memory to host memory
  cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
  
  //printf("N %ld, Elapsed %.2f ms, h_out[0] = %f\n", N, iElaps, h_out[0]);
  printf("%.2f, ", iElaps);

  free(h_out);
  cudaFree(d_out);
  return 0;
}
