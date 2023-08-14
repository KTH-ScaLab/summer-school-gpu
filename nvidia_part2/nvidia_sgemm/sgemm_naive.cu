
#include <stdio.h>
#include <sys/time.h>

#define DataType double

// Compute C = A * B
// Sgemm stands for single precision general matrix-matrix multiply
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < numARows && col < numBColumns) {
    DataType sum = 0;
    for (int ii = 0; ii < numAColumns; ii++) {
      sum += A[row * numAColumns + ii] * B[ii * numBColumns + col];
    }
    C[row * numBColumns + col] = sum;
  }
}

//@@ Insert code to implement timer
struct timeval t_start, t_end;
void myCPUTimer_start(){
  gettimeofday(&t_start, 0);
}
//@@ Insert code to implement timer
void myCPUTimer_stop(){
  cudaDeviceSynchronize();
  gettimeofday(&t_end, 0);
  double time = (1000000.0*(t_end.tv_sec-t_start.tv_sec) + t_end.tv_usec-t_start.tv_usec);
  printf("Elasped %6.1f microseconds \n", time);
}

int main(int argc, char **argv) {
  
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
  numARows    = atoi(argv[1]);
  numAColumns = atoi(argv[2]);
  numBRows    = numAColumns;
  numBColumns = atoi(argv[3]);
  numCRows    = numARows;
  numCColumns = numBColumns;
  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostA = (DataType *)malloc(numARows * numAColumns * sizeof(DataType));
  hostB = (DataType *)malloc(numBRows * numBColumns * sizeof(DataType));
  hostC = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));
  resultRef  = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));
  
  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  for(int i=0; i<numARows; i++){
    for(int j=0; j<numAColumns; j++){
      hostA[i*numAColumns+j] = 1.0;
    }
  }
  for(int i=0; i<numBRows; i++){
    for(int j=0; j<numBColumns; j++){
      hostB[i*numBColumns+j] = 3.0;
    }
  }  
  for(int i=0; i<numARows; i++){
    for(int j=0; j<numCColumns; j++){
      DataType tmp = 0;
      for(int ii=0; ii<numAColumns; ii++)
          tmp += hostA[i*numAColumns+ii] * hostB[ii*numBColumns+j];
      resultRef[i*numCColumns+j] = tmp;
    }    
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(DataType));
  cudaMalloc((void **)&deviceB, numBRows * numBColumns * sizeof(DataType));
  cudaMalloc((void **)&deviceC, numCRows * numCColumns * sizeof(DataType));

  //@@ Insert code to below to Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(DataType), cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 blockDim(16, 16);
  dim3 gridDim(ceil(((float)numCColumns) / ((float)blockDim.x)), ceil(((float)numCRows) / ((float)blockDim.y)));

  //@@ Launch the GPU Kernel here
  gemm<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, numARows,
                              numAColumns, numBRows, numBColumns);


  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost);


  //@@ Insert code below to compare the output with the reference
  bool valid = true;
  for(int i=0; i<numCRows; i++){
    for(int j=0; j<numCColumns; j++){
      if( hostC[i*numCColumns+j] != resultRef[i*numCColumns+j] ){
        printf("hostC[%d][%d] = %f != %f\n", i,j,hostC[i*numCColumns+j],resultRef[i*numCColumns+j]);
        valid = false;
        break;
      }
    }
  }
  if(valid) printf("valid\n");

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);
  free(resultRef);

  return 0;
}
