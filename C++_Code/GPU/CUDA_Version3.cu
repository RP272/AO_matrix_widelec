#include <cstdint>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

#define N 2048
#define blocksize 16
#define timeNumber 10




uint64_t nanos() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()
    ).count();
}


// !! See full comments in the CUDA_Version1 !!
float *A, *B, *C;


__global__
// ONLY for square matrices for now!!!
void threading(const float *A, const float *B, float *C)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N || col >= N)
        return;

    float acc = 0.f;
    for (int k = 0; k < N; ++k)
        acc += A[row * N + k] * B[k * N + col];

    C[row * N + col] = acc;

}


int main()
{

    // Memory allocation
    cudaMallocManaged(&A, N*N*sizeof(float));
    cudaMallocManaged(&B, N*N*sizeof(float));
    cudaMallocManaged(&C, N*N*sizeof(float));

    // Matrix initialization
    for (int y=0; y<N; ++y)
        for (int x=0; x<N; ++x) {
            A[y * N + x] = (y + x) * 0.001f;
            B[y * N + x] = (y - x) * 0.002f;
        }

    for(int times = 0; times < timeNumber; times++) {
        // Start to count the time needed
        uint64_t start = nanos();

        dim3 block(32, 32); // We are choosing a block size inside a C matrix
        // Here we are calculating how many blocks are needed to cover the C matrix horizontally and vertically
        dim3 grid((N + block.x - 1) / block.x,
                  (N + block.y - 1) / block.y);

        // At the end we have (considering the 32 x 32 block size) 64 x 64 blocks (which gives us 4096 such blocks)
        // Additionally each block has 1024 threads (32x32) (also a maximum number of threads).
        // This gives us a big number of threads (4 194 304) - one for each cell inside the C matrix


        threading<<<grid, block>>>(A, B, C);


        cudaDeviceSynchronize();





        // Finalize time counting
        uint64_t end = nanos();
        double gflop = (N * N * 2.0 * N) * 1e-9;
        double s = (end - start) * 1e-9;
        std::cout << "GFLOPS " << gflop / s<<std::endl;
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);


    return 0;
}
