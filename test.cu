// kernel.cu
#include <iostream>

__global__ void add(int *a, int *b, int *c) {
    int index = threadIdx.x;
    c[index] = a[index] + b[index];
}

int main() {
    const int arraySize = 5;
    int a[arraySize] = {1, 2, 3, 4, 5};
    int b[arraySize] = {10, 20, 30, 40, 50};
    int c[arraySize] = {0};

    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, arraySize * sizeof(int));
    cudaMalloc((void **)&d_b, arraySize * sizeof(int));
    cudaMalloc((void **)&d_c, arraySize * sizeof(int));

    cudaMemcpy(d_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    add<<<1, arraySize>>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Result: ";
    for (int i = 0; i < arraySize; i++) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}