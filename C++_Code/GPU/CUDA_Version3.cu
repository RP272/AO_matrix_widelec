#include <cstdint>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 2048
#define blocksize 32
#define timeNumber 10.0

uint64_t nanos() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()
    ).count();
}

float *A, *B, *C, *C_ref;

__global__
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


int main() {
    uint64_t startone = nanos();
    double gflop = (N * N * 2.0 * N) * 1e-9;
    double sumTimeCPU = 0.0;
    double sumTimeGPU = 0.0;

    size_t bytes = size_t(N) * size_t(N) * sizeof(float);

    // Allocate unified memory (CPU+GPU accessible)
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);
    cudaMallocManaged(&C_ref, bytes);

    // Initialize matrices
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            // A[y * N + x] = (y + x) * 0.001f;
            A[y * N + x] = 2*(y + x);
            // B[y * N + x] = (y - x) * 0.002f;
            B[y * N + x] = 3*(y - x);
        }

    // Set device
    int dev = 0;
    cudaGetDevice(&dev);

    cudaMemAdvise(A, bytes, cudaMemAdviseSetPreferredLocation, dev);
    cudaMemAdvise(B, bytes, cudaMemAdviseSetPreferredLocation, dev);
    cudaMemAdvise(C, bytes, cudaMemAdviseSetPreferredLocation, dev);
    cudaMemAdvise(C_ref, bytes, cudaMemAdviseSetPreferredLocation, dev);

    cudaStream_t s = 0;

    // Prefetch
    cudaMemPrefetchAsync(A, bytes, dev, s);
    cudaMemPrefetchAsync(B, bytes, dev, s);
    cudaMemPrefetchAsync(C, bytes, dev, s);
    cudaMemPrefetchAsync(C_ref, bytes, dev, s);
    cudaStreamSynchronize(s);

    // Define grid & block
    dim3 block(blocksize, blocksize);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    // Warmup kernel
    for (int t = 0; t < 2; t++) {
        threading<<<grid, block>>>(A, B, C);
    }
    cudaDeviceSynchronize();

    // CUDA event setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int times = 0; times < timeNumber; times++) {
        uint64_t startCPU = nanos();

        cudaEventRecord(start);
        threading<<<grid, block>>>(A, B, C);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaDeviceSynchronize();

        uint64_t endCPU = nanos();

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        double sGPU = ms / 1000.0;
        double sCPU = (endCPU - startCPU) * 1e-9;
        sumTimeCPU += sCPU;
        sumTimeGPU += sGPU;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS uses column-major order by default
    // So we compute C_ref = Bᵗ * Aᵗ, which is equivalent to A * B in row-major
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                B, N,   // B
                A, N,   // A
                &beta,
                C_ref, N); // C_ref = A*B

    cudaDeviceSynchronize();
    cublasDestroy(handle);

    double maxDiff = 0.0, avgDiff = 0.0;
    for (int i = 0; i < N * N; i++) {
        double diff = std::fabs(C[i] - C_ref[i]);
        maxDiff = std::max(maxDiff, diff);
        avgDiff += diff;
    }
    avgDiff /= (N * N);

    std::cout << "Average latency (CPU chrono): " << (sumTimeCPU / timeNumber) << " s\n";
    std::cout << "Average latency (GPU event):  " << (sumTimeGPU / timeNumber) << " s\n";
    std::cout << "GFLOPS (CPU chrono): " << gflop / (sumTimeCPU / timeNumber) << "\n";
    std::cout << "GFLOPS (GPU event):  " << gflop / (sumTimeGPU / timeNumber) << "\n";

    std::cout << "\nVerification (cuBLAS):\n";
    std::cout << "  Max abs diff: " << maxDiff << "\n";
    std::cout << "  Avg abs diff: " << avgDiff << "\n";

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(C_ref);

    return 0;
}
