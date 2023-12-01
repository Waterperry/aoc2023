__global__ void vec_add(int size, const int *a, const int *b, int *out){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) out[i] = a[i] + b[i];
}

__global__ void vec_sub(int size, const int *a, const int *b, int *out){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) out[i] = a[i] - b[i];
}

__global__ void vec_mul(int size, const int *a, const int *b, int *out){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) out[i] = a[i] * b[i];
}

__global__ void vec_floor_div(int size, const int *a, const int *b, int *out){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) out[i] = a[i] / b[i];
}

__global__ void vec_exp(int size, const int *a, const int *b, int *out){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        int power = 1;
        out[i] = a[i];
        while (power < b[i]) {
            out[i] *= a[i];
            power += 1;
        }
    }
}

__global__ void vec_reduce_sum(int size, int *arr, int *out){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    auto l_idx = i * 2;
    int offset = 1;

    while (offset < size) {
        if (l_idx < size) {
            auto r_idx = l_idx + offset;
            if (r_idx < size) {
                arr[l_idx] += arr[r_idx];
            }
        }
        __syncthreads();
        offset *= 2;
        l_idx *= 2;
    }

    __syncthreads();
    if (i == 0) *out = arr[0];
}
