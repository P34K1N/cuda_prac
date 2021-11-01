#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <sstream>
#include <stdlib.h>

#include "Image.h"
#include "clock.h"
#include "memman.cuh"
#include "safecalls.cuh"

__inline__ __device__ uint8_t sh_c(uint8_t c, uint8_t u, uint8_t l, uint8_t d, uint8_t r) {
    return uint8_t(round(
        double(c) * 5 - double(u) - double(l) - double(d) - double(r)
    ));
}

__global__ void sharpen(uint8_t* src, uint8_t* tgt, int width, int height) {
    extern __shared__ uint8_t buf[];

    int x = (blockDim.x - 2) * blockIdx.x + threadIdx.x;
    int y = (blockDim.y - 2) * blockIdx.y + threadIdx.y;
    if (x >= width + 2 || y >= height + 2) { return; }

    buf[blockDim.x * threadIdx.y + threadIdx.x] = src[(width + 2) * y + x];
    __syncthreads();

    if (threadIdx.x == 0 || threadIdx.y == 0 || threadIdx.x == blockDim.x - 1 || threadIdx.y == blockDim.y - 1 || x > width || y > height) return;

    uint8_t centre = buf[blockDim.x * threadIdx.y + threadIdx.x];
    uint8_t up = buf[blockDim.x * (threadIdx.y - 1) + threadIdx.x];
    uint8_t left = buf[blockDim.x * threadIdx.y + threadIdx.x - 1];
    uint8_t down = buf[blockDim.x * (threadIdx.y + 1) + threadIdx.x];
    uint8_t right = buf[blockDim.x * threadIdx.y + threadIdx.x + 1];

    tgt[width * (y - 1) + (x - 1)] = sh_c(centre, up, left, down, right);
}

__inline__ __device__ uint8_t ed_c(uint8_t c, uint8_t u, uint8_t l, uint8_t d, uint8_t r,
    uint8_t ul, uint8_t ur, uint8_t dr, uint8_t dl) {
    return uint8_t(round(
        double(c) * 8 - double(u) - double(l) - double(d) - double(r)
        - double(ul) - double(ur) - double(dr) - double(dl)
    ));
}

__global__ void edge(uint8_t* src, uint8_t* tgt, int width, int height, bool is_alpha) {
    extern __shared__ uint8_t buf[];

    int x = (blockDim.x - 2) * blockIdx.x + threadIdx.x;
    int y = (blockDim.y - 2) * blockIdx.y + threadIdx.y;
    if (x >= width + 2 || y >= height + 2) { return; }

    if (is_alpha) {
        tgt[width * (y - 1) + x - 1] = 255;
        return;
    }

    buf[blockDim.x * threadIdx.y + threadIdx.x] = src[(width + 2) * y + x];
    __syncthreads();

    if (threadIdx.x == 0 || threadIdx.y == 0 || threadIdx.x == blockDim.x - 1 || threadIdx.y == blockDim.y - 1 || x > width || y > height) return;

    uint8_t centre = buf[blockDim.x * threadIdx.y + threadIdx.x];
    uint8_t up = buf[blockDim.x * (threadIdx.y - 1) + threadIdx.x];
    uint8_t left = buf[blockDim.x * threadIdx.y + threadIdx.x - 1];
    uint8_t down = buf[blockDim.x * (threadIdx.y + 1) + threadIdx.x];
    uint8_t right = buf[blockDim.x * threadIdx.y + threadIdx.x + 1];
    uint8_t up_left = buf[blockDim.x * (threadIdx.y - 1) + threadIdx.x - 1];
    uint8_t up_right = buf[blockDim.x * (threadIdx.y - 1) + threadIdx.x + 1];
    uint8_t down_right = buf[blockDim.x * (threadIdx.y + 1) + threadIdx.x + 1];
    uint8_t down_left = buf[blockDim.x * (threadIdx.y + 1) + threadIdx.x - 1];

    tgt[width * (y - 1) + x - 1] = ed_c(centre, up, left, down, right,
        up_left, up_right, down_right, down_left);
}

__inline__ __device__ uint8_t g5_c(uint8_t c, uint8_t u, uint8_t l, uint8_t d, uint8_t r,
    uint8_t ul, uint8_t ur, uint8_t dr, uint8_t dl,
    uint8_t uu, uint8_t uur, uint8_t uurr, uint8_t urr,
    uint8_t rr, uint8_t drr, uint8_t ddrr, uint8_t ddr,
    uint8_t dd, uint8_t ddl, uint8_t ddll, uint8_t dll,
    uint8_t ll, uint8_t ull, uint8_t uull, uint8_t uul) {
    return uint8_t(round(
        36. / 256. * c +
        24. / 256. * (u + r + d + l) +
        16. / 256. * (ul + ur + dr + dl) +
        6. / 256. * (uu + rr + dd + ll) +
        4. / 256. * (uur + urr + drr + ddr + ddl + dll + ull + uul) +
        1. / 256. * (uurr + ddrr + ddll + uull)
    ));
}

__global__ void gauss5(uint8_t* src, uint8_t* tgt, int width, int height) {
    extern __shared__ uint8_t buf[];

    int x = (blockDim.x - 4) * blockIdx.x + threadIdx.x;
    int y = (blockDim.y - 4) * blockIdx.y + threadIdx.y;
    if (x >= width + 4 || y >= height + 4) { return; }

    buf[blockDim.x * threadIdx.y + threadIdx.x] = src[(width + 4) * y + x];
    __syncthreads();

    if (threadIdx.x <= 1 || threadIdx.y <= 1 || threadIdx.x >= blockDim.x - 2 || threadIdx.y >= blockDim.y - 2 || x > width + 1 || y > height + 1) return;

    uint8_t centre = buf[blockDim.x * threadIdx.y + threadIdx.x];
    uint8_t up = buf[blockDim.x * (threadIdx.y - 1) + threadIdx.x];
    uint8_t left = buf[blockDim.x * threadIdx.y + threadIdx.x - 1];
    uint8_t down = buf[blockDim.x * (threadIdx.y + 1) + threadIdx.x];
    uint8_t right = buf[blockDim.x * threadIdx.y + threadIdx.x + 1];
    uint8_t up_left = buf[blockDim.x * (threadIdx.y - 1) + threadIdx.x - 1];
    uint8_t up_right = buf[blockDim.x * (threadIdx.y - 1) + threadIdx.x + 1];
    uint8_t down_right = buf[blockDim.x * (threadIdx.y + 1) + threadIdx.x + 1];
    uint8_t down_left = buf[blockDim.x * (threadIdx.y + 1) + threadIdx.x - 1];
    uint8_t up_up = buf[blockDim.x * (threadIdx.y - 2) + threadIdx.x];
    uint8_t up_up_right = buf[blockDim.x * (threadIdx.y - 2) + threadIdx.x + 1];
    uint8_t up_up_right_right = buf[blockDim.x * (threadIdx.y - 2) + threadIdx.x + 2];
    uint8_t up_right_right = buf[blockDim.x * (threadIdx.y - 1) + threadIdx.x + 2];
    uint8_t right_right = buf[blockDim.x * threadIdx.y + threadIdx.x + 2];
    uint8_t down_right_right = buf[blockDim.x * (threadIdx.y + 1) + threadIdx.x + 2];
    uint8_t down_down_right_right = buf[blockDim.x * (threadIdx.y + 2) + threadIdx.x + 2];
    uint8_t down_down_right = buf[blockDim.x * (threadIdx.y + 2) + threadIdx.x + 1];
    uint8_t down_down = buf[blockDim.x * (threadIdx.y + 2) + threadIdx.x];
    uint8_t down_down_left = buf[blockDim.x * (threadIdx.y + 2) + threadIdx.x - 1];
    uint8_t down_down_left_left = buf[blockDim.x * (threadIdx.y + 2) + threadIdx.x - 2];
    uint8_t down_left_left = buf[blockDim.x * (threadIdx.y + 1) + threadIdx.x - 2];
    uint8_t left_left = buf[blockDim.x * threadIdx.y + threadIdx.x - 2];
    uint8_t up_left_left = buf[blockDim.x * (threadIdx.y - 1) + threadIdx.x - 2];
    uint8_t up_up_left_left = buf[blockDim.x * (threadIdx.y - 2) + threadIdx.x - 2];
    uint8_t up_up_left = buf[blockDim.x * (threadIdx.y - 2) + threadIdx.x - 1];

    tgt[width * (y - 2) + (x - 2)] = g5_c(centre, up, left, down, right,
        up_left, up_right, down_right, down_left,
        up_up, up_up_right, up_up_right_right, up_right_right,
        right_right, down_right_right, down_down_right_right, down_down_right,
        down_down, down_down_left, down_down_left_left, down_left_left,
        left_left, up_left_left, up_up_left_left, up_up_left);
}


class CudaProcessor {
    int in_width, in_height, in_size;
    int out_width, out_height, out_size;
    int block_dim, blocks_x, blocks_y;
    char kernel_type;

    uint8_t* chh[4]; //channels on host
    uint8_t* chid[4]; //channels on device - input
    uint8_t* chod[4]; //channels on device - output

public:
    CudaProcessor(int width, int height, char kernel_type, int block_dim_base) {
        this->in_width = width;
        this->in_height = height;
        this->in_size = width * height;

        this->kernel_type = kernel_type;
        int ker_dim = 3;
        if (kernel_type == 'g') ker_dim = 5;

        this->out_width = width - ker_dim + 1;
        this->out_height = height - ker_dim + 1;
        this->out_size = out_width * out_height;

        this->block_dim = block_dim_base - ker_dim + 1;
        this->blocks_x = int(ceil(double(out_width) / block_dim));
        this->blocks_y = int(ceil(double(out_height) / block_dim));

        for (int i = 0; i < 4; i++) {
            SAFE_CALL(cudaMallocHost(&chh[i], this->in_size));
            SAFE_CALL(cudaMalloc(&chid[i], this->in_size));
            SAFE_CALL(cudaMalloc(&chod[i], this->out_size));
        }
    }

    void ProcessImage(Pixel* input, Pixel* output) {
        for (int i = 0; i < this->in_size; i++) {
            chh[0][i] = input[i].r;
            chh[1][i] = input[i].g;
            chh[2][i] = input[i].b;
            chh[3][i] = input[i].a;
        }
        for (int ch = 0; ch < 4; ch++) {
            cudaStream_t a_stream;
            SAFE_CALL(cudaStreamCreate(&a_stream));
            SAFE_CALL(cudaMemcpyAsync(chid[ch], chh[ch], in_size, cudaMemcpyHostToDevice, a_stream));
            switch (this->kernel_type) { //no function pointer kernel calls sadge
            case 's':
                SAFE_KERNEL_CALL((sharpen <<<dim3(blocks_x, blocks_y), dim3(block_dim + 2, block_dim + 2), (block_dim + 2) * (block_dim + 2), a_stream >>> (chid[ch], chod[ch], out_width, out_height)));
                break;
            case 'e':
                SAFE_KERNEL_CALL((edge <<<dim3(blocks_x, blocks_y), dim3(block_dim + 2, block_dim + 2), (block_dim + 2) * (block_dim + 2), a_stream >>> (chid[ch], chod[ch], out_width, out_height, ch == 3)));
                break;
            case 'g':
                SAFE_KERNEL_CALL((gauss5 <<<dim3(blocks_x, blocks_y), dim3(block_dim + 4, block_dim + 4), (block_dim + 4) * (block_dim + 4), a_stream >>> (chid[ch], chod[ch], out_width, out_height)));
                break;
            default:
                break;
            }
            SAFE_CALL(cudaMemcpyAsync(chh[ch], chod[ch], out_size, cudaMemcpyDeviceToHost, a_stream));
            SAFE_CALL(cudaStreamDestroy(a_stream));
        }
        for (int i = 0; i < this->out_size; i++) {
            output[i].r = chh[0][i];
            output[i].g = chh[1][i];
            output[i].b = chh[2][i];
            output[i].a = chh[3][i];
        }
    }

    ~CudaProcessor() {
        for (int i = 0; i < 4; i++) {
            SAFE_CALL(cudaFreeHost(chh[i]));
            SAFE_CALL(cudaFree(chid[i]));
            SAFE_CALL(cudaFree(chod[i]));
        }
    }
};

void cudaHostRegisterSmart(void* p, size_t size, unsigned int flags = 0) {
    cudaPointerAttributes my_attr;
    if (cudaPointerGetAttributes(&my_attr, p) == cudaErrorInvalidValue) {
        cudaGetLastError(); // clear out the previous API error
        SAFE_CALL(cudaHostRegister(p, size, flags));
    }
}

std::string to_string(const int n)
{
    std::ostringstream stm;
    stm << n;
    return stm.str();
}

int from_string(const std::string & str)
{
    std::istringstream stm(str);
    int n;
    stm >> n;
    return n;
}

std::string get_name(int i) {
    std::string s = to_string(i);
    while (s.length() < 4) {
        s = std::string("0") + s;
    }
    return s + ".png";
}


int main(int argc, char *argv[]) {
    try {
        if (argc < 3) { return 0; }

        int ker_dim = 3;
        if (argv[1][0] == 'g') ker_dim = 5;

        double start = get_wall_time();
        switch (argv[2][0]) {
        case 'l':
        {
            Image input("../resources/large.png");
            MemoryManager::MMpin(input.Data(), input.Size());
            Image output(input.Width() - ker_dim + 1, input.Height() - ker_dim + 1, 4);
            MemoryManager::MMpin(output.Data(), output.Size());
            CudaProcessor proc(input.Width(), input.Height(), argv[1][0], 32);
            proc.ProcessImage(input.Data(), output.Data());
            output.Save("../resources/large_proc.png");
            break;
        }
        case 's':
        {
            Image input("../resources/small/0000.png");
            Image output(input.Width() - ker_dim + 1, input.Height() - ker_dim + 1, 4, false);
            CudaProcessor proc(input.Width(), input.Height(), argv[1][0], 32);
            int N;
            const int TOT = 1000;
            if (argc >= 4)  {
                N = from_string(std::string(argv[3]));
            }
            else {
                N = 50;
            }
            Image* inputs = new Image[N];
            Pixel** outputs = (Pixel**)malloc(N * sizeof(Pixel*));
            for (int i = 0; i < N; i++) {
                outputs[i] = (Pixel *)malloc(output.Size());
                //SAFE_CALL(cudaMallocHost(&outputs[i], output.Size()));
            }
            int cur = 0;
            while (cur < TOT) {
                for (int i = 0; i < N && cur + i < TOT; i++) {
                    std::string name = get_name(i + cur);
                    inputs[i].Renew(std::string("../resources/small/") + name);
                    //MemoryManager::MMpin(inputs[i].Data(), inputs[i].Size());
                }
                for (int i = 0; i < N && cur + i < TOT; i++) {\
                    proc.ProcessImage(inputs[i].Data(), outputs[i]);
                }
                for (int i = 0; i < N && cur + i < TOT; i++) {
                    std::string name = get_name(i + cur);
                    output.Data() = outputs[i];
                    output.Save(std::string("../resources/small_proc/") + name);
                }
                cur += N;
            }
            for (int i = 0; i < N; i++) {
                //SAFE_CALL(cudaFreeHost(outputs[i]));
                free(outputs[i]);
            }
            delete [] inputs;
            free(outputs);
            break;
        }
        }
        double end = get_wall_time();
        printf("%lf\n", end - start);

        return 0;
    }
    catch (std::string s) {
        printf(s.c_str());
        throw s;
    }
}