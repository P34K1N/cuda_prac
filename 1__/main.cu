#include <stdio.h>
#include <math.h>

#include "Image.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////

#define SAFE_CALL( CallInstruction ) { \
    cudaError_t cuerr = CallInstruction; \
    if(cuerr != cudaSuccess) { \
         printf("CUDA error: %s at call \"" #CallInstruction "\"\n", cudaGetErrorString(cuerr)); \
		 throw "error in CUDA API function, aborting..."; \
    } \
}

#define SAFE_KERNEL_CALL( KernelCallInstruction ){ \
    KernelCallInstruction; \
    cudaError_t cuerr = cudaGetLastError(); \
    if(cuerr != cudaSuccess) { \
        printf("CUDA error in kernel launch: %s at kernel \"" #KernelCallInstruction "\"\n", cudaGetErrorString(cuerr)); \
		throw "error in CUDA kernel launch, aborting..."; \
    } \
    cuerr = cudaDeviceSynchronize(); \
    if(cuerr != cudaSuccess) { \
        printf("CUDA error in kernel execution: %s at kernel \"" #KernelCallInstruction "\"\n", cudaGetErrorString(cuerr)); \
		throw "error in CUDA kernel execution, aborting..."; \
    } \
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void conv(Pixel *src, Pixel *tgt, double *ker, int ker_pad, int width, int height) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height) { return; }

    double r = 0, g = 0, b = 0, a = 0;

    for (int i = -ker_pad; i <= ker_pad; i++) {
        for (int j = -ker_pad; j <= ker_pad; j++) {
            Pixel px = src[(width + 2 * ker_pad) * (y + ker_pad + j) + x + ker_pad + i];
            double kr = ker[(2 * ker_pad + 1) * (j + ker_pad) + i + ker_pad];
            r += (double(px.r) / 255) * kr;
            g += (double(px.g) / 255) * kr;
            b += (double(px.b) / 255) * kr;
            a += (double(px.a) / 255) * kr;
        }
    }
    tgt[width * y + x].a = round(a * 255);
    tgt[width * y + x].r = round(r * 255);
    tgt[width * y + x].g = round(g * 255);
    tgt[width * y + x].b = round(b * 255);
}

double SHARPEN[] = {
    0, -1, 0,
    -1, 5, -1,
    0, -1, 0
};

double EDGE[] = {
    -1, -1, -1,
    -1, 8, -1,
    -1, -1, -1
};

double GAUSS5[] = {
    1. / 256,  4. / 256,  6. / 256,  4. / 256, 1. / 256,
    4. / 256, 16. / 256, 24. / 256, 16. / 256, 4. / 256,
    6. / 256, 24. / 256, 36. / 256, 24. / 256, 6. / 256,
    4. / 256, 16. / 256, 24. / 256, 16. / 256, 4. / 256,
    1. / 256,  4. / 256,  6. / 256,  4. / 256, 1. / 256,
};

class CudaProcessor {
    Pixel *cudaImgIn, *cudaImgOut;
    double *cudaKernel;
    int ker_dim, ker_padding;
    int in_width, in_height, in_size;
    int out_width, out_height, out_size;
    int block_dim, blocks_x, blocks_y;
    float memcpy_time, kernel_time;

public:
    CudaProcessor(int width, int height, double* convker, int _ker_dim) {
        this->in_width = width;
        this->in_height = height;
        this->in_size = width * height * sizeof(*cudaImgIn);
        SAFE_CALL(cudaMalloc(&cudaImgIn, in_size));

        this->out_width = width - _ker_dim + 1;
        this->out_height = height - _ker_dim + 1;
        this->out_size = out_width * out_height * sizeof(*cudaImgOut);
        SAFE_CALL(cudaMalloc(&cudaImgOut, out_size));

        this->ker_dim = _ker_dim;
        this->ker_padding = (ker_dim - 1) / 2;
        SAFE_CALL(cudaMalloc(&cudaKernel, ker_dim * ker_dim * sizeof(*cudaKernel)));
        SAFE_CALL(cudaMemcpy(cudaKernel, convker, ker_dim * ker_dim * sizeof(*cudaKernel), cudaMemcpyHostToDevice));

        this->block_dim = 32;
        this->blocks_x = int(ceil(double(out_width) / block_dim));
        this->blocks_y = int(ceil(double(out_height) / block_dim));

        this->memcpy_time = 0;
        this->kernel_time = 0;
    }

    void ProcessImage(Image& input, Image& output) {
        cudaEvent_t start, stop;
        float temp;
        SAFE_CALL(cudaEventCreate(&start));
        SAFE_CALL(cudaEventCreate(&stop));

        SAFE_CALL(cudaEventRecord(start));
        SAFE_CALL(cudaMemcpy(cudaImgIn, input.Data(), in_size, cudaMemcpyHostToDevice));
        SAFE_CALL(cudaDeviceSynchronize());
        SAFE_CALL(cudaEventRecord(stop));
        SAFE_CALL(cudaEventSynchronize(stop));
        SAFE_CALL(cudaEventElapsedTime(&temp, start, stop));
        memcpy_time += temp / 1000.;

        SAFE_CALL(cudaEventRecord(start));
        SAFE_KERNEL_CALL((conv <<<dim3(blocks_x, blocks_y), dim3(block_dim, block_dim) >>> (cudaImgIn, cudaImgOut, cudaKernel, ker_padding, out_width, out_height)));
        SAFE_CALL(cudaDeviceSynchronize());
        SAFE_CALL(cudaEventRecord(stop));
        SAFE_CALL(cudaEventSynchronize(stop));
        SAFE_CALL(cudaEventElapsedTime(&temp, start, stop));
        kernel_time += temp / 1000.;

        SAFE_CALL(cudaEventRecord(start));
        SAFE_CALL(cudaMemcpy(output.Data(), cudaImgOut, out_size, cudaMemcpyDeviceToHost));
        SAFE_CALL(cudaDeviceSynchronize());
        SAFE_CALL(cudaEventRecord(stop));
        SAFE_CALL(cudaEventSynchronize(stop));
        SAFE_CALL(cudaEventElapsedTime(&temp, start, stop));
        memcpy_time += temp / 1000.;
    }

    float KernelTime() {
        return kernel_time;
    }

    float TotalTime() {
        return kernel_time + memcpy_time;
    }

    ~CudaProcessor() {
        SAFE_CALL(cudaFree(cudaImgIn));
        SAFE_CALL(cudaFree(cudaImgOut));
        SAFE_CALL(cudaFree(cudaKernel));
    }
};

std::string get_name(int i) {
    std::string s = std::to_string(i);
    while (s.length() < 4) {
        s = std::string("0") + s;
    }
    return s + ".png";
}

int main(int argc, char *argv[]) {
    double* convker;
    int ker_dim = 0;

    if (argc < 3) { return 0; }

    switch (argv[1][0]) {
    case 's':
        convker = SHARPEN;
        ker_dim = 3;
        break;
    case 'e':
        convker = EDGE;
        ker_dim = 3;
        break;
    case 'g':
        convker = GAUSS5;
        ker_dim = 5;
        break;
    }

    switch (argv[2][0]) {
    case 'l':
    {
        Image input("../resources/large.png");
        Image output(input.Width() - ker_dim + 1, input.Height() - ker_dim + 1, 4);
        CudaProcessor proc(input.Width(), input.Height(), convker, ker_dim);
        proc.ProcessImage(input, output);
        output.Save("../resources/large_proc.png");
        printf("%lf\t%lf\r\n", proc.KernelTime(), proc.TotalTime());
        break;
    }
    case 's':
    {
        Image input("../resources/small/0000.png");
        Image output(input.Width() - ker_dim + 1, input.Height() - ker_dim + 1, 4);
        CudaProcessor proc(input.Width(), input.Height(), convker, ker_dim);
        for (int i = 0; i < 1000; i++) {
            auto name = get_name(i);
            Image input(std::string("../resources/small/") + name);
            proc.ProcessImage(input, output);
            output.Save(std::string("../resources/small_proc/") + name);
        }
        printf("%lf\t%lf\r\n", proc.KernelTime(), proc.TotalTime());
        break;
    }
    }
    
    return 0;
}