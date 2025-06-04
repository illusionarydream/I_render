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

__device__ float gaussian(float x, float sigma) {
    return expf(-(x * x) / (2.0f * sigma * sigma));
}

__global__ void denoiseKernel(V3f *image,
                              const V3f *input_image,
                              const int width,
                              const int height,
                              const int kernel_size,
                              const int denoise_type,
                              const float sigma_spatial) {
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
    if (denoise_type == 0 && x < width && y < height) {
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

    // * gaussian Filter
    if (denoise_type == 1 && x < width && y < height) {
        float sum_weight = 0.0f;
        float r = 0.0f, g = 0.0f, b = 0.0f;

        for (int dy = -half; dy <= half; ++dy) {
            for (int dx = -half; dx <= half; ++dx) {
                float distance = sqrtf(float(dx * dx + dy * dy));
                float weight = gaussian(distance, sigma_spatial);

                V3f neighbor = shared_block_image[local_y + dy][local_x + dx];
                r += neighbor[0] * weight;
                g += neighbor[1] * weight;
                b += neighbor[2] * weight;
                sum_weight += weight;
            }
        }

        image[idx][0] = r / sum_weight;
        image[idx][1] = g / sum_weight;
        image[idx][2] = b / sum_weight;
    }

    // * Bilateral Filter
    if (denoise_type == 2 && x < width && y < height) {
        V3f center = shared_block_image[local_y][local_x];
        float sigma_color = 0.5f;
        float sigma_space = kernel_size / 2.0f;

        float w_sum = 0.0f;
        V3f filtered_pixel = V3f(0.0f, 0.0f, 0.0f);

        for (int dy = -half; dy <= half; ++dy) {
            for (int dx = -half; dx <= half; ++dx) {
                int nx = local_x + dx;
                int ny = local_y + dy;
                V3f neighbor = shared_block_image[ny][nx];

                float spatial_dist2 = float(dx * dx + dy * dy);
                float color_dist2 = dot(neighbor - center, neighbor - center);

                float w = gaussian(sqrtf(spatial_dist2), sigma_space) *
                          gaussian(sqrtf(color_dist2), sigma_color);

                filtered_pixel[0] += neighbor[0] * w;
                filtered_pixel[1] += neighbor[1] * w;
                filtered_pixel[2] += neighbor[2] * w;
                w_sum += w;
            }
        }

        if (w_sum > 0.0f) {
            image[idx][0] = filtered_pixel[0] / w_sum;
            image[idx][1] = filtered_pixel[1] / w_sum;
            image[idx][2] = filtered_pixel[2] / w_sum;
        } else {
            image[idx] = center;
        }
    }
}
