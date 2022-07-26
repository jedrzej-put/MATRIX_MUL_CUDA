#include <stdio.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <helper_functions.h>
#include <helper_cuda.h>

template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(
    float *A_system_memory, float *A_global, bool *A_buffered_list, int wA,
    float *B_system_memory, float *B_global, , bool *B_buffered_list, int wB,
    float *C) {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;


  int row_to_thread = by * blockDim.y + ty;
	int col_to_thread = bx * blockDim.x + tx;
	float C_local = 0;

	for (int m = 0; m < wA / BLOCK_SIZE; ++m) {
    
    __shared__ float Ads[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bds[BLOCK_SIZE][BLOCK_SIZE];
    
    if (!A_buffered_list[by * gridDim.x + m]){
      A_global[row_to_thread * wA + m*BLOCK_SIZE + tx] = A_system_memory[row_to_thread * wA + m*BLOCK_SIZE + tx];
    }
    
    if (!B_buffered_list[m * gridDim.x + bx]){
      B_global[(m*BLOCK_SIZE + ty)*wA + col_to_thread] = B_system_memory[(m*BLOCK_SIZE + ty)*wA + col_to_thread];
    }
		
    Ads[ty][tx] = A_global[row_to_thread * wA + m*BLOCK_SIZE + tx];
		Bds[ty][tx] = B_global[(m*BLOCK_SIZE + ty)*wA + col_to_thread];
		__syncthreads();
    
    A_buffered_list[by * gridDim.x + m] = true;
    B_buffered_list[m * gridDim.x + bx] = true;
		
    #pragma unroll
		for (int k = 0; k < BLOCK_SIZE; ++k)
			C_local += Ads[ty][k] * Bds[k][tx];
		__syncthreads();
	}

	C[row_to_thread * wA + col_to_thread] = C_local;
}

void ConstantInit(float *data, int size, float val) {
  for (int i = 0; i < size; ++i) {
    data[i] = val;
  }
}


int MatrixMultiply(int argc, char **argv,
                   int block_size, const dim3 &dimsA,
                   const dim3 &dimsB) {
  
  // Allocate device memory
  float *d_A, *d_B, *d_C;
  // Allocate device ptrs to host
  float *d_A_system_memory, *d_B_system_memory;
  // Allocate device cache info
  bool *d_A_buffered_list, *d_B_buffered_list;

  // Allocate host memory for matrices A and B
  unsigned int size_A = dimsA.x * dimsA.y;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float *h_A;
  checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
  checkCudaErrors(cudaHostGetDevicePointer((void **)&d_A_system_memory, (void *)h_A, 0));
  unsigned int size_B = dimsB.x * dimsB.y;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float *h_B;
  checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
  checkCudaErrors(cudaHostGetDevicePointer((void **)&d_B_system_memory, (void *)h_B, 0));
  cudaStream_t stream;

  // Initialize host memory
  const float valB = 0.01f;
  ConstantInit(h_A, size_A, 1.0f);
  ConstantInit(h_B, size_B, valB);


  // Allocate host matrix C
  dim3 dimsC(dimsB.x, dimsA.y, 1);
  unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
  float *h_C;
  checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));

  if (h_C == NULL) {
    fprintf(stderr, "Failed to allocate host matrix C!\n");
    exit(EXIT_FAILURE);
  }

  // Allocate CUDA events that we'll use for timing
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));

  // Setup execution parameters
  dim3 threads(block_size, block_size);
  dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

  unsigned int mem_size_A_buffered_list = sizeof(bool) * grid.x * grid.y;
  unsigned int mem_size_B_buffered_list = sizeof(bool) * grid.x * grid.y;
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A_buffered_list), mem_size_A_buffered_list));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B_buffered_list), mem_size_B_buffered_list));
  // Create and start timer
  printf("Computing result using CUDA Kernel...\n");

  // Performs warmup operation using matrixMul CUDA kernel
  if (block_size == 8) {
    MatrixMulCUDA<8>
        <<<grid, threads, 0, stream>>>( d_A_system_memory, d_A, d_A_buffered_list, dimsA.x, d_B_system_memory, d_B, d_B_buffered_list,  dimsB.x, d_C);
  }
  else if (block_size == 16) {
    MatrixMulCUDA<16>
        <<<grid, threads, 0, stream>>>( d_A_system_memory, d_A, d_A_buffered_list, dimsA.x, d_B_system_memory, d_B, d_B_buffered_list,  dimsB.x, d_C);
  } else {
    MatrixMulCUDA<32>
        <<<grid, threads, 0, stream>>>( d_A_system_memory, d_A, d_A_buffered_list, dimsA.x, d_B_system_memory, d_B, d_B_buffered_list,  dimsB.x, d_C);
  }

  printf("done\n");
  checkCudaErrors(cudaStreamSynchronize(stream));

  // Record the start event
  checkCudaErrors(cudaEventRecord(start, stream));

  // Execute the kernel
  int nIter = 1;

  for (int j = 0; j < nIter; j++) {
    if (block_size == 8) {
      MatrixMulCUDA<8>
          <<<grid, threads, 0, stream>>>( d_A_system_memory, d_A, d_A_buffered_list, dimsA.x, d_B_system_memory, d_B, d_B_buffered_list,  dimsB.x, d_C);
    }
    else if (block_size == 16) {
      MatrixMulCUDA<16>
          <<<grid, threads, 0, stream>>>( d_A_system_memory, d_A, d_A_buffered_list, dimsA.x, d_B_system_memory, d_B, d_B_buffered_list,  dimsB.x, d_C);
    } else if (block_size == 32){
      MatrixMulCUDA<32>
          <<<grid, threads, 0, stream>>>( d_A_system_memory, d_A, d_A_buffered_list, dimsA.x, d_B_system_memory, d_B, d_B_buffered_list,  dimsB.x, d_C);
    }
  }

  // Record the stop event
  checkCudaErrors(cudaEventRecord(stop, stream));

  // Wait for the stop event to complete
  checkCudaErrors(cudaEventSynchronize(stop));
  
  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  // Compute and print the performance
  float msecPerMatrixMul = msecTotal / nIter;
  double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
                             static_cast<double>(dimsA.y) *
                             static_cast<double>(dimsB.x);
  double gigaFlops =
      (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
  printf(
      "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
      " WorkgroupSize= %u threads/block\n",
      gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads.x * threads.y);

  // Copy result from device to host
  checkCudaErrors(
      cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));

  printf("Checking computed result for correctness: ");
  bool correct = true;

  // test relative error by the formula
  //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
  double eps = 1.e-6;  // machine zero

  for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
    double abs_err = fabs(h_C[i] - (dimsA.x * valB));
    double dot_length = dimsA.x;
    double abs_val = fabs(h_C[i]);
    double rel_err = abs_err / abs_val / dot_length;

    if (rel_err > eps) {
      printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
             i, h_C[i], dimsA.x * valB, eps);
      correct = false;
    }
  }

  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

  // Clean up memory
  checkCudaErrors(cudaFreeHost(h_A));
  checkCudaErrors(cudaFreeHost(h_B));
  checkCudaErrors(cudaFreeHost(h_C));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  printf(
      "\nNOTE: The CUDA Samples are not meant for performance "
      "measurements. Results may vary when GPU Boost is enabled.\n");

  if (correct) {
    return EXIT_SUCCESS;
  } else {
    return EXIT_FAILURE;
  }
}


/**
 * Program main
 */
int main(int argc, char **argv) {
  printf("[Matrix Multiply Using CUDA] - Starting...\n");

  if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
      checkCmdLineFlag(argc, (const char **)argv, "?")) {
    printf("Usage -device=n (n >= 0 for deviceID)\n");
    printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
    printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
    printf("  Note: Outer matrix dimensions of A & B matrices" \
           " must be equal.\n");

    exit(EXIT_SUCCESS);
  }

  // This will pick the best possible CUDA capable device, otherwise
  // override the device ID based on input provided at the command line
  int dev = findCudaDevice(argc, (const char **)argv);
  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaSetDevice(dev));

  /* Verify the selected device supports mapped memory and set the device
     flags for mapping host memory. */

  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
  checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));

  int block_size = 32;

  dim3 dimsA(block_size * 32, block_size * 32, 1);
  dim3 dimsB(block_size * 32, block_size * 32, 1);

  // width of Matrix A
  if (checkCmdLineFlag(argc, (const char **)argv, "wA")) {
    dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
  }

  // height of Matrix A
  if (checkCmdLineFlag(argc, (const char **)argv, "hA")) {
    dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
  }

  // width of Matrix B
  if (checkCmdLineFlag(argc, (const char **)argv, "wB")) {
    dimsB.x = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
  }

  // height of Matrix B
  if (checkCmdLineFlag(argc, (const char **)argv, "hB")) {
    dimsB.y = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
  }

  if (dimsA.x != dimsB.y) {
    printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
           dimsA.x, dimsB.y);
    exit(EXIT_FAILURE);
  }

  printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y,
         dimsB.x, dimsB.y);

  checkCudaErrors(cudaProfilerStart());
  int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);
  checkCudaErrors(cudaProfilerStop());

  exit(matrix_result);
}
