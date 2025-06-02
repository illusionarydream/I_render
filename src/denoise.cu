#include "denoise.cuh"

__device__ float median(float *arr, int len) {
    for (int i = 0; i <= len / 2; ++i) {
        for (int j = i + 1; j < len; ++j) {
            if (arr[j] < arr[i]) {
                float tmp = arr[i];
                arr[i] = arr[j];
                arr[j] = tmp;
            }
        }
    }
    return arr[len / 2];
}

__global__ void denoiseKernel(V3f *image,
                              const V3f *input_image,
                              const int width,
                              const int height,
                              const int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    // * set shared memory
    int half = kernel_size / 2;
    int local_x = threadIdx.x + half;
    int local_y = threadIdx.y + half;
    __shared__ V3f shared_block_image[BLOCK_DKERNEL_SIZE][BLOCK_DKERNEL_SIZE];

    // * move the data to shared memory
    if (x < width && y < height) {
        if (!(x > 0 && x < width - 1 && y > 0 && y < height - 1))
            // if the pixel is not on the border, copy the pixel value to shared memory
            shared_block_image[local_y][local_x] = input_image[idx];
        else {
            // if the pixel is on the border, fill the shared memory with the pixel value
            for (int i = -half; i <= half; ++i) {
                for (int j = -half; j <= half; ++j) {
                    int neighbor_x = min(max(x + i, 0), width - 1);
                    int neighbor_y = min(max(y + j, 0), height - 1);
                    shared_block_image[local_y + j][local_x + i] = input_image[neighbor_y * width + neighbor_x];
                }
            }
        }
    }

    // * synchronize the threads
    __syncthreads();

    // * median filter
    if (x < width && y < height) {
        // median filter 的排序数组
        float r[MAX_DKERNEL_SIZE * MAX_DKERNEL_SIZE];
        float g[MAX_DKERNEL_SIZE * MAX_DKERNEL_SIZE];
        float b[MAX_DKERNEL_SIZE * MAX_DKERNEL_SIZE];
        int count = 0;

        for (int dy = -half; dy <= half; ++dy) {
            for (int dx = -half; dx <= half; ++dx) {
                V3f col = shared_block_image[local_y + dy][local_x + dx];
                r[count] = col[0];
                g[count] = col[1];
                b[count] = col[2];
                count++;
            }
        }

        image[idx][0] = median(r, count);
        image[idx][1] = median(g, count);
        image[idx][2] = median(b, count);
    }

    // * Bilateral Filter
}
