#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void count_greater_than_five(const float* d_array, int* d_count, int n) {
    __shared__ int local_count[256]; // Lokális számláló minden blokkhoz
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Inicializáljuk a lokális számlálót
    local_count[tid] = 0;

    // Ellenőrzés: 5-nél nagyobb számok
    if (idx < n && d_array[idx] > 5.0f) {
        local_count[tid] = 1;
    }
    __syncthreads();

    // Redukció a blokkon belül
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            local_count[tid] += local_count[tid + stride];
        }
        __syncthreads();
    }

    // A blokk eredményének mentése a globális memóriába
    if (tid == 0) {
        atomicAdd(d_count, local_count[0]);
    }
}

int main() {
    const int N = 1024; // Elem szám a vektorban
    std::vector<float> h_array(N);

    // Töltjük a vektort véletlenszerű értékekkel [0, 10] tartományban
    for (int i = 0; i < N; ++i) {
        h_array[i] = static_cast<float>(rand() % 11); // [0, 10] közötti egész számok
    }

    // GPU memória allokálása
    float* d_array;
    int* d_count;
    cudaMalloc(&d_array, N * sizeof(float));
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    // Másolás a GPU-ra
    cudaMemcpy(d_array, h_array.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel indítása
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    count_greater_than_five<<<blocks, threads_per_block>>>(d_array, d_count, N);

    // Eredmény visszamásolása
    int h_count;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Eredmény kiíratása
    std::cout << "5-nél nagyobb számok száma: " << h_count << std::endl;

    // GPU memória felszabadítása
    cudaFree(d_array);
    cudaFree(d_count);

    int input;
    std::cin>>input;
    std::cout<<input*2;

    return 0;
}
