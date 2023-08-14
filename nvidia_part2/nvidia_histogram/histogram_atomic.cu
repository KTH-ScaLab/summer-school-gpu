#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Privatized bins
  extern __shared__ unsigned int bins_s[];
  for (unsigned int binIdx = threadIdx.x; binIdx < num_bins;
       binIdx += blockDim.x) {
    bins_s[binIdx] = 0;
  }
  __syncthreads();

  // Histogram
  for (unsigned int i = tid; i < num_elements; i += blockDim.x * gridDim.x) {
    atomicAdd(&(bins_s[input[i]]), 1);
  }
  __syncthreads();

  // Commit to global memory
  for (unsigned int binIdx = threadIdx.x; binIdx < num_bins;
       binIdx += blockDim.x) {
    atomicAdd(&(bins[binIdx]), bins_s[binIdx]);
  }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_bins) {
    bins[tid] = min(bins[tid], 127);
  }
  
}


int main(int argc, char **argv) {
  
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  //@@ Insert code below to read in inputLength from args
  inputLength = atoi(argv[1]);
  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput = (unsigned int *)malloc(inputLength * sizeof(unsigned int));
  hostBins  = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  resultRef = (unsigned int *)calloc(NUM_BINS, sizeof(unsigned int));
  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  std::random_device rd; 
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distr(0, (NUM_BINS - 1)); // define the range
  for(int i=0; i<inputLength; i++){
    hostInput[i] = distr(gen);
  }
  //@@ Insert code below to create reference result in CPU
  for(int i=0; i<inputLength; i++){
    resultRef[hostInput[i]] ++;
  }
  for(int i=0; i<NUM_BINS; i++){
    if(resultRef[i]>127) resultRef[i]=127;
  }   

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc((void **)&deviceInput, inputLength * sizeof(unsigned int));
  cudaMalloc((void **)&deviceBins, NUM_BINS * sizeof(unsigned int));

  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);

  //@@ Insert code to initialize GPU results
  cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

  //@@ Initialize the grid and block dimensions here
  dim3 blockDim1(32);
  dim3 gridDim1(ceil(((float)inputLength) / ((float)blockDim1.x)));

  //@@ Launch the GPU Kernel here
  histogram_kernel<<<gridDim1, blockDim1, NUM_BINS * sizeof(unsigned int)>>>(deviceInput, deviceBins, inputLength, NUM_BINS);

  dim3 blockDim2(32);
  dim3 gridDim2(ceil(((float)NUM_BINS) / ((float)blockDim2.x)));
  convert_kernel<<<gridDim2, blockDim2>>>(deviceBins, NUM_BINS);
  
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference
  bool valid = true;
  for(int i=0; i<NUM_BINS; i++){
    printf("%d, ", hostBins[i]);
    if( hostBins[i] != resultRef[i] ){
      printf("hostBins[%d] = %d != %d\n", i, hostBins[i], resultRef[i]);
      valid = false;
      break;
    }
  }
  if(valid) printf("\nvalid\n");

  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);

  //@@ Free the CPU memory here
  free(hostInput);
  free(hostBins);
  free(resultRef);

  return 0;
}
