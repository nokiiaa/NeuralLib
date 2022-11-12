#include <stdio.h>
#include <assert.h>
#include <math.h>

#include <vector>
#include <algorithm>
#include <utility>
#include <cstdint>
#include <cstddef>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#define Align(x, n) (((x) + (n) - 1) & -(n))
#define AlignBlock(x, n) (((x) + (n) - 1) / (n))

#define DevAllocate(alloc, offset, name, size) \
    float *name  = (float*)(alloc + offset); \
    offset = Align(offset + size, 256);
 
#define LinearBlockCount(length) AlignBlock((length), MaxThreadsPerBlock)
#define LinearBlockSize(length) std::min(length, MaxThreadsPerBlock)
#define LinearBlockCountEx(length, block_sz) AlignBlock((length), (block_sz))
#define LinearBlockSizeEx(length, block_sz) std::min((length), (block_sz))

int ConcurrentKernelCount = -1;
bool Initialized = false;

struct CudaStream
{
    cudaStream_t Stream;
    uint8_t *DeviceAllocation;
    int DeviceAllocationSize;
} *CudaStreams;

extern "C" __declspec(dllexport)
bool CudaCoreInitialize()
{
    if (!Initialized) {
        int dev = findCudaDevice(0, nullptr);
        cudaDeviceProp prop;
        
        if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
            return false;
        
        ConcurrentKernelCount = prop.concurrentKernels ? 32 : 1;
        
        checkCudaErrors(cudaMallocHost((void **)&CudaStreams, sizeof(CudaStream) * ConcurrentKernelCount));
        
        for (int i = 0; i < ConcurrentKernelCount; i++)
            checkCudaErrors(cudaStreamCreateWithFlags(&CudaStreams[i].Stream, cudaStreamNonBlocking));
        
        Initialized = true;
    }
    
    return true;
}

int CudaCoreConcurrentKernelCount()
{
    return ConcurrentKernelCount;
}

constexpr int MaxThreadsPerBlock = 512;
constexpr int MmTileDim = 16;

// https://gist.github.com/wh5a/4313739
__global__ void KernelMatrixMultiply(
    float *dest, float *a, float *b,
    int wc, int hc,
    int wa, int ha,
    int wb, int hb)
{
    __shared__ float sub_a[MmTileDim][MmTileDim + 1];
    __shared__ float sub_b[MmTileDim][MmTileDim + 1];
    
    int bx = blockIdx.x, by = blockIdx.y,
       tx = threadIdx.x, ty = threadIdx.y,
       row = by * MmTileDim + ty,
       col = bx * MmTileDim + tx;
       
    float elem = 0;

    for (int m = 0; m < (wa + MmTileDim - 1) / MmTileDim; m++) {
        sub_a[ty][tx] =
            row < ha && m * MmTileDim + tx < wa ?
            a[row * wa + m * MmTileDim + tx] :
            0;
          
        sub_b[ty][tx] =
            col < wb && m * MmTileDim + ty < hb ?
            b[(m * MmTileDim + ty) * wb + col] :
            0;

        __syncthreads();
           
        for (int k = 0; k < MmTileDim; k++)
            elem += sub_a[ty][k] * sub_b[k][tx];
          
        __syncthreads();
    }
        
    if (row < hc && col < wc)
        dest[row * wc + col] = elem;
}

constexpr int MtTileDim = 32, MtBlockRows = 8;

// https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc
__global__ void KernelMatrixTranspose(float *output, const float *input, int w, int h)
{
    __shared__ float sub[MtTileDim][MtTileDim + 1];

    int x = blockIdx.x * MtTileDim + threadIdx.x;
    int y = blockIdx.y * MtTileDim + threadIdx.y;
    
    for (int j = 0; j < MtTileDim; j += MtBlockRows)
        sub[threadIdx.y + j][threadIdx.x] = (x < w && y + j < h) * input[(y + j) * w + x];
        
    __syncthreads();
        
    x = blockIdx.y * MtTileDim + threadIdx.x;
    y = blockIdx.x * MtTileDim + threadIdx.y;
    
    for (int j = 0; j < MtTileDim && x < h && y + j < w; j += MtBlockRows)
        output[(y + j) * h + x] = sub[threadIdx.x][threadIdx.y + j];
    
    /*if (y < w && x < h)
        output[y * h + x] = input[x * w + y];
    
    /*for (int j = 0; j < MtTileDim; j += MtBlockRows)
        sub[threadIdx.y + j][threadIdx.x] = y + j < h && x < w ? input[(y + j) * w + x] : 0;

    __syncthreads();

    x = blockIdx.y * MtTileDim + threadIdx.x;
    y = blockIdx.x * MtTileDim + threadIdx.y;

    for (int j = 0; j < MtTileDim && y + j < w && x < h; j += MtBlockRows)
        output[(y + j) * h + x] = sub[threadIdx.x][threadIdx.y + j];*/
}

__global__ void KernelHaramardMul(float *c, const float *a, const float *b, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size)
        c[idx] = a[idx] * b[idx];
}

__global__ void KernelAddTensors(float *c, const float *a, const float *b, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size)
        c[idx] = a[idx] + b[idx];
}


__global__ void KernelActivateReLU(
    float *dst_array, float *src_array, int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) dst_array[idx] = src_array[idx] < 0 ? 0 : src_array[idx];
}

__global__ void KernelActivateLeakyReLU(
    float *dst_array, float *src_array, int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) dst_array[idx] = src_array[idx] < 0 ? .01f * src_array[idx] : src_array[idx];
}

__global__ void KernelActivateReLUPrime(
    float *dst_array, float *src_array, int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) dst_array[idx] = src_array[idx] < 0 ? 0 : 1;
}

__global__ void KernelActivateLeakyReLUPrime(
    float *dst_array, float *src_array, int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) dst_array[idx] = src_array[idx] < 0 ? .01f : 1;
}

__global__ void KernelConv(
    float *output, float *input, float *kernel, float *bias,
    int iw, int ih, int ic, int padding,
    int kw, int kh, int kd)
{
    const int ow = iw + padding * 2 - kw + 1;
    const int oh = ih + padding * 2 - kh + 1;

    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < ow && y < oh && d < kd) {
        int oidx = (d * oh + y) * ow + x;
        
        float r = bias[oidx];
        
        for (int kc = 0; kc < ic; kc++) {
            for (int ky = 0; ky < kh && y - padding + ky < ih; ky++) {
                int Y = y - padding + ky;
            
                for (int kx = 0; kx < kw && x - padding + kx < iw; kx++) {
                    float I = 0.f;
                    
                    int X = x - padding + kx;
                    
                    if (X >= 0 && Y >= 0) {
                        I = input[(kc * ih + Y) * iw + X];
                        r += kernel[((d * ic + kc) * kh + ky) * kw + kx] * I;
                    }
                }
            }
        }
        
        output[oidx] = r;
    }
}

__global__ void KernelConvBackprop1(
    float *kernel, float *delta, float *prev_activations,
    int iw, int ih, int ic, int padding,
    int kw, int kh, int kd)
{
    const int ow = iw + padding * 2 - kw + 1;
    const int oh = ih + padding * 2 - kh + 1;

    int a = blockIdx.x * blockDim.x + threadIdx.x;
    
    int c = a / kd;
    int d = a % kd;
    int kx = blockIdx.y * blockDim.y + threadIdx.y;
    int ky = blockIdx.z * blockDim.z + threadIdx.z;
    
    float r = 0.f;
    
    if (kx < kw && ky < kh && c < ic) {
        for (int y = 0; y < oh && y - padding + ky < ih; y++) {
            int Y = y - padding + ky;

            for (int x = 0; x < ow && x - padding + kx < iw; x++) {
                float I = 0.f;
                        
                int X = x - padding + kx;
                        
                if (X >= 0 && Y >= 0) {
                    I = prev_activations[(c * ih + Y) * iw + X];
                    r += delta[(d * oh + y) * ow + x] * I;
                }
            }
        }
        
        kernel[((d * ic + c) * kh + ky) * kw + kx] = r;
    }
}

__global__ void KernelConvBackprop2(
    float *new_delta, float *kernel, float *delta,
    int iw, int ih, int ic, int padding,
    int kw, int kh, int kd)
{
    const int ow = iw + padding * 2 - kw + 1;
    const int oh = ih + padding * 2 - kh + 1;

    int x = blockIdx.x * blockDim.x + threadIdx.x + padding;
    int y = blockIdx.y * blockDim.y + threadIdx.y + padding;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < iw + padding && y < ih + padding && c < ic) {
        float r = 0.f;
        
        for (int n = 0; n < kh && n <= y && y - n < oh; n++)
            for (int m = 0; m < kw && m <= x && x - m < ow; m++)
                for (int d = 0; d < kd; d++)
                    r += delta[(d * oh + y - n) * ow + x - m] *
                         kernel[((d * ic + c) * kh + n) * kw + m];
        
        new_delta[(c * ih + y - padding) * iw + x - padding] = r;
    }
}

__global__ void KernelPool(
    float *output, float *input,
    int iw, int ih, int ic, int ks)
{
    const int ow = iw / ks;
    const int oh = ih / ks;
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    
    float avg = 0.f;
    
    if (x < ow && y < oh && c < ic) {
        for (int ky = 0; ky < ks && y * ks + ky < ih; ky++)
            for (int kx = 0; kx < ks && x * ks + kx < iw; kx++)
                avg += input[(c * ih + y * ks + ky) * iw + x * ks + kx];

        output[(c * oh + y) * ow + x] = avg / (ks * ks);
    }
}

__global__ void KernelPoolBackprop(
    float *new_delta, float *delta,
    int w, int h, int ic, int ks)
{
    const int ow = w / ks;
    const int oh = h / ks;
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < w && y < h && c < ic)
        new_delta[(c * h + y) * w + x] =
            delta[(c * oh + y / ks) * ow + x / ks] / (ks * ks);
}

__global__ void KernelUpdateGradients(
    float *layer_weights_gradient,
    float *layer_bias_gradient,
    float *weights_gradient,
    float *bias_gradient,
    int batch_size,
    int bias_length, int weights_length)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < bias_length)
        atomicAdd(layer_bias_gradient + i, bias_gradient[i] / batch_size);
    
    if (i < weights_length)
        atomicAdd(layer_weights_gradient + i, weights_gradient[i] / batch_size);
}

__global__ void KernelSoftmaxSum(
    float *out_sum,
    float *in_activations,
    int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < length)
        atomicAdd(out_sum, expf(in_activations[idx]));
}

__global__ void KernelSoftmaxDeltaAndCrossEntropy(
    float *out_delta,
    float *out_loss,
    float *in_activations,
    float *in_expected,
    float *softmax_sum,
    int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < length) {
        float expected = in_expected[idx];
        float act = in_activations[idx];
        float normalized = expf(act) / (*softmax_sum);
        out_delta[idx] = normalized - expected;
        atomicAdd(out_loss, -logf(normalized) * expected);
    }
}

__global__ void KernelAdamStep(
    float *weights, float *bias,
    float *weight_gradients,
    float *weight_gradients_m,
    float *weight_gradients_v,
    float *bias_gradients,
    float *bias_gradients_m,
    float *bias_gradients_v,
    float b1, float b2,
    float if1, float if2,
    float rate, int weights_len, int bias_len)
{
    constexpr float epsilon = 10e-4f;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float gb, gw, a, b;
    
    if (idx < bias_len) {
        gb = bias_gradients[idx];
        a = bias_gradients_m[idx] = bias_gradients_m[idx] * b1 + gb * (1 - b1);
        b = bias_gradients_v[idx] = bias_gradients_v[idx] * b2 + (gb * gb) * (1 - b2);
        bias[idx] -= (a / if1) / (sqrtf(b / if2) + epsilon) * rate;
    }
    
    if (idx < weights_len) {
        gw = weight_gradients[idx];
        a = weight_gradients_m[idx] = weight_gradients_m[idx] * b1 + gw * (1 - b1);
        b = weight_gradients_v[idx] = weight_gradients_v[idx] * b2 + (gw * gw) * (1 - b2);
        weights[idx] -= (a / if1) / (sqrtf(b / if2) + epsilon) * rate;
    }
}

enum
{
    LayerTypeConv = 1,
    LayerTypeDense,
    LayerTypePool
};

enum
{
    ActivationTypeNone = -1,
    ActivationTypeReLU = 0,
    ActivationTypeLeakyReLU = 1
};

struct Layer
{
    int Type, Activation;
    float *DeviceWeights, *DeviceBias;
    
    virtual int InLayerLen() { return -1; }
    virtual int OutLayerLen() { return -1; }
    virtual int WeightsLen() { return -1; }
    
    Layer(int type, int activation) : Type(type), Activation(activation) {}
};

struct DenseLayer : public Layer
{
    int InputLen, OutputLen;

    int WeightsLen() { return InputLen * OutputLen; }

    int InLayerLen() { return InputLen; }
    int OutLayerLen() { return OutputLen; }

    DenseLayer(int input_length, int output_length,
        float *device_weights, float *device_bias,
        int activation)
        : Layer(LayerTypeDense, activation)
    {
        InputLen = input_length;
        OutputLen = output_length;
        DeviceWeights = device_weights;
        DeviceBias = device_bias;
    }
};

struct ConvLayer : public Layer
{
    int InputW, InputH, InputChannels, KernelW, KernelH, Padding;
    int FilterCount;
        
    int OutputW() { return InputW + Padding * 2 - KernelW + 1; }
    int OutputH() { return InputH + Padding * 2 - KernelH + 1; }
    int InLayerLen() { return InputW * InputH * InputChannels; }
    int OutLayerLen() { return OutputW() * OutputH() * FilterCount; }
    int WeightsLen() { return KernelW * KernelH * InputChannels * FilterCount; }

    ConvLayer(
        int iw, int ih, int ic, int padding,
        int kw, int kh, int kd, float *d_kernel, float *d_bias,
        int activation)
        : Layer(LayerTypeConv, activation)
    {
        InputW = iw;
        InputH = ih;
        InputChannels = ic;
        KernelW = kw;
        KernelH = kh;
        Padding = padding;
        FilterCount = kd;
        DeviceWeights = d_kernel;
        DeviceBias = d_bias;
    }
};

struct PoolLayer : public Layer
{
    int InputW, InputH, Channels;
    int KernelSize;
    
    int OutputW() { return InputW / KernelSize; }
    int OutputH() { return InputH / KernelSize; }
    int InLayerLen() { return InputW * InputH * Channels; }
    int OutLayerLen() { return OutputW() * OutputH() * Channels; }
    
    PoolLayer(int iw, int ih, int chan, int kernel_size)
        : Layer(LayerTypePool, ActivationTypeNone)
    {
        InputW = iw;
        InputH = ih;
        Channels = chan;
        KernelSize = kernel_size;
    }
};

int BatchSize;
std::vector<Layer*> Layers;
// Gradients averaged after the whole batch has been processed
std::vector<float*> DevWeightGradients, DevBiasGradients;
std::vector<float*> DevWeightGradientsM, DevBiasGradientsM;
std::vector<float*> DevWeightGradientsV, DevBiasGradientsV;

// Per-example temp gradients and weights
std::vector<float*> BatchCurWeightGradients, BatchCurBiasGradients;
std::vector<float*> BatchWeightsT;
// Per-example temp vectors for forward and backpropagation
std::vector<float*> BatchPartialDeltaOut, BatchPartialDeltaIn;
std::vector<float*> BatchForwardOut, BatchForwardIn;
std::vector<std::vector<float*>> BatchActPrimes, BatchActivations;


extern "C" __declspec(dllexport)
void CudaCoreAddDenseLayer(
    int input_length, int output_length,
    float *init_weights, float *init_bias,
    int activation)
{
    float *d_weights, *d_bias;
    
    const int w_size = input_length * output_length * sizeof(float);
    const int b_size = output_length * sizeof(float);
    
    cudaMalloc((void**)&d_weights, w_size);
    cudaMalloc((void**)&d_bias, b_size);
    
    checkCudaErrors(cudaMemcpy(d_weights, init_weights, w_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_bias, init_bias, b_size, cudaMemcpyHostToDevice));
    
    Layers.push_back(new DenseLayer(input_length, output_length, d_weights, d_bias, activation));
}

extern "C" __declspec(dllexport)
void CudaCoreAddConvLayer(
    int iw, int ih, int ic,
    int kw, int kh, int padding,
    int kd, float *init_kernel, float *init_bias,
    int activation)
{
    float *d_kernel, *d_bias;
    
    const int ow = iw + padding * 2 - kw + 1;
    const int oh = ih + padding * 2 - kh + 1;
    
    const int k_size = kw * kh * ic * kd * sizeof(float);
    const int b_size = ow * oh * kd * sizeof(float);
    
    checkCudaErrors(cudaMalloc((void**)&d_kernel, k_size));
    checkCudaErrors(cudaMalloc((void**)&d_bias, b_size));
    
    checkCudaErrors(cudaMemcpy(d_kernel, init_kernel, k_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_bias, init_bias, b_size, cudaMemcpyHostToDevice));
    
    Layers.push_back(new ConvLayer(iw, ih, ic, padding, kw, kh, kd, d_kernel, d_bias, activation));
}

extern "C" __declspec(dllexport)
void CudaCoreAddPoolLayer(int iw, int ih, int chan, int kernel_size)
{
    Layers.push_back(new PoolLayer(iw, ih, chan, kernel_size));
}

extern "C" __declspec(dllexport)
void CudaCorePrepareForTraining(int batch_size)
{
    if (DevWeightGradients.size() || DevBiasGradients.size())
        return;
    
    BatchSize = batch_size;
    
    int max_weights_size = 0, max_dense_weights_size, max_bias_size = 0, max_layer_size = 0;
    
    for (int j = 0; j < BatchSize; j++) {
        BatchActPrimes.push_back(std::vector<float*>());
        BatchActivations.push_back(std::vector<float*> { nullptr });
        checkCudaErrors(cudaMalloc((void**)&BatchActivations[j][0], Layers[0]->InLayerLen() * sizeof(float)));
    }
    
    for (int i = 0; i < Layers.size(); i++) {
        Layer *layer = Layers[i];
        
        float *g_weights_ptr = nullptr, *g_weights_m_ptr = nullptr, *g_weights_v_ptr = nullptr;
        float *g_bias_ptr = nullptr, *g_bias_m_ptr = nullptr, *g_bias_v_ptr = nullptr;
        
        DenseLayer *dense;
        ConvLayer *conv;
        PoolLayer *pool;
        
        int wsize = 0, bsize = 0;
                
        switch (layer->Type) {
        case LayerTypeDense:
            dense = (DenseLayer*)layer;
            
            max_layer_size = std::max(max_layer_size, std::max(dense->InputLen, dense->OutputLen) * (int)sizeof(float));
            
            wsize = dense->WeightsLen() * (int)sizeof(float);
            bsize = dense->OutputLen * (int)sizeof(float);
            
            max_weights_size = std::max(max_weights_size, wsize);
            max_dense_weights_size = std::max(max_dense_weights_size, wsize);
            max_bias_size = std::max(max_bias_size, bsize);

            break;
            
        case LayerTypeConv:
            conv = (ConvLayer*)layer;
            
            max_layer_size = std::max(max_layer_size,
                std::max(
                conv->InLayerLen(),
                conv->OutLayerLen()) * (int)sizeof(float));
            
            wsize = conv->WeightsLen() * (int)sizeof(float);
            bsize = conv->OutLayerLen() * (int)sizeof(float);

            max_weights_size = std::max(max_weights_size, wsize);
            max_bias_size = std::max(max_bias_size, bsize);
            
            break;
            
        case LayerTypePool:
            pool = (PoolLayer*)layer;

            bsize = pool->OutLayerLen() * (int)sizeof(float);

            max_layer_size = std::max(max_layer_size,
                std::max(
                pool->InLayerLen() * (int)sizeof(float),
                bsize));
            break;
        }
        
        if (layer->Type != LayerTypePool) {
            checkCudaErrors(cudaMalloc((void**)&g_weights_ptr, wsize));
            checkCudaErrors(cudaMalloc((void**)&g_weights_m_ptr, wsize));
            checkCudaErrors(cudaMalloc((void**)&g_weights_v_ptr, wsize));
            checkCudaErrors(cudaMemset(g_weights_m_ptr, 0, wsize));
            checkCudaErrors(cudaMemset(g_weights_v_ptr, 0, wsize));
            
            checkCudaErrors(cudaMalloc((void**)&g_bias_ptr, bsize));
            checkCudaErrors(cudaMalloc((void**)&g_bias_m_ptr, bsize));
            checkCudaErrors(cudaMalloc((void**)&g_bias_v_ptr, bsize));
            checkCudaErrors(cudaMemset(g_bias_m_ptr, 0, bsize));
            checkCudaErrors(cudaMemset(g_bias_v_ptr, 0, bsize));
        }
        
        for (int j = 0; j < BatchSize; j++) {
            BatchActPrimes[j].push_back(nullptr);
            BatchActivations[j].push_back(nullptr);
            checkCudaErrors(cudaMalloc((void**)&BatchActPrimes[j][i], bsize));
            checkCudaErrors(cudaMalloc((void**)&BatchActivations[j][i + 1], bsize));
        }

        DevWeightGradients.push_back(g_weights_ptr);
        DevWeightGradientsM.push_back(g_weights_m_ptr);
        DevWeightGradientsV.push_back(g_weights_v_ptr);
        DevBiasGradients.push_back(g_bias_ptr);
        DevBiasGradientsM.push_back(g_bias_m_ptr);
        DevBiasGradientsV.push_back(g_bias_v_ptr);
    }
        
    for (int i = 0; i < BatchSize; i++) {
        BatchPartialDeltaOut.push_back(nullptr);
        BatchPartialDeltaIn.push_back(nullptr);
        BatchForwardOut.push_back(nullptr);
        BatchForwardIn.push_back(nullptr);
        BatchCurWeightGradients.push_back(nullptr);
        BatchCurBiasGradients.push_back(nullptr);
        BatchWeightsT.push_back(nullptr);
        
        checkCudaErrors(cudaMalloc((void**)&BatchPartialDeltaOut[i], max_layer_size));
        checkCudaErrors(cudaMalloc((void**)&BatchPartialDeltaIn[i], max_layer_size));
        checkCudaErrors(cudaMalloc((void**)&BatchForwardOut[i], max_layer_size));
        checkCudaErrors(cudaMalloc((void**)&BatchForwardIn[i], max_layer_size));
        checkCudaErrors(cudaMalloc((void**)&BatchCurWeightGradients[i], max_weights_size));
        checkCudaErrors(cudaMalloc((void**)&BatchCurBiasGradients[i], max_bias_size));
        checkCudaErrors(cudaMalloc((void**)&BatchWeightsT[i], max_dense_weights_size));
    }
}

extern "C" __declspec(dllexport)
void CudaCoreCleanUpAfterTraining()
{
    while (DevWeightGradients.size()) {
        checkCudaErrors(cudaFree(DevWeightGradients.back()));
        checkCudaErrors(cudaFree(DevWeightGradientsM.back()));
        checkCudaErrors(cudaFree(DevWeightGradientsV.back()));
        checkCudaErrors(cudaFree(DevBiasGradients.back()));
        checkCudaErrors(cudaFree(DevBiasGradientsM.back()));
        checkCudaErrors(cudaFree(DevBiasGradientsV.back()));

        DevWeightGradients.pop_back();
        DevWeightGradientsM.pop_back();
        DevWeightGradientsV.pop_back();
        DevBiasGradients.pop_back();
        DevBiasGradientsM.pop_back();
        DevBiasGradientsV.pop_back();
    }

    while (BatchForwardIn.size()) {
        checkCudaErrors(cudaFree(BatchForwardOut.back()));
        checkCudaErrors(cudaFree(BatchForwardIn.back()));
        checkCudaErrors(cudaFree(BatchPartialDeltaOut.back()));
        checkCudaErrors(cudaFree(BatchPartialDeltaIn.back()));

        BatchForwardOut.pop_back();
        BatchForwardIn.pop_back();
        BatchPartialDeltaOut.pop_back();
        BatchPartialDeltaIn.pop_back();
    }
}

void CudaCoreActivationFunc(
    cudaStream_t stream, float *dst_values, float *src_values, int length, int activation, bool prime)
{
    int n_blocks = LinearBlockCount(length);
    int block_size = LinearBlockSize(length);
    
    if (prime) {
        switch (activation) {
        case ActivationTypeReLU:
            KernelActivateReLUPrime<<<n_blocks, block_size, 0, stream>>>(dst_values, src_values, length);
            break;
        case ActivationTypeLeakyReLU:
            KernelActivateLeakyReLUPrime<<<n_blocks, block_size, 0, stream>>>(dst_values, src_values, length);
            break;
        default:
            checkCudaErrors(cudaMemcpyAsync(dst_values, src_values, length * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            break;
        }
    }
    else {
        switch (activation) {
        case ActivationTypeReLU:
            KernelActivateReLU<<<n_blocks, block_size, 0, stream>>>(dst_values, src_values, length);
            break;
        case ActivationTypeLeakyReLU:
            KernelActivateLeakyReLU<<<n_blocks, block_size, 0, stream>>>(dst_values, src_values, length);
            break;
        default:
            checkCudaErrors(cudaMemcpyAsync(dst_values, src_values, length * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            break;
        }
    }
}

extern "C" __declspec(dllexport)
void CudaCoreUploadParamsToCpu(
    float **weightsArray, float **biasArray)
{
    for (int i = 0; i < Layers.size(); i++) {
        if (Layers[i]->Type != LayerTypePool) {
            checkCudaErrors(cudaMemcpy(
                weightsArray[i], Layers[i]->DeviceWeights,
                Layers[i]->WeightsLen() * sizeof(float), cudaMemcpyDeviceToHost));
            
            checkCudaErrors(cudaMemcpy(
                biasArray[i], Layers[i]->DeviceBias,
                Layers[i]->OutLayerLen() * sizeof(float), cudaMemcpyDeviceToHost));
        }
    }
    
    cudaDeviceSynchronize();
}

extern "C" __declspec(dllexport)
void CudaCoreTrainOnBatch(
    float *example_in_data, int one_example_in_length,
    float *example_out_data, int one_example_out_length,
    float *out_avg_loss, int *out_correct_count,
    float learning_rate, float b1, float b2,
    int t)
{
    float *d_out_losses, *out_losses, *outputs, *d_softmax_sums;
    float *d_example_in_data, *d_example_out_data;
    
    int examples_in_size  = one_example_in_length  * BatchSize * sizeof(float);
    int examples_out_size = one_example_out_length * BatchSize * sizeof(float);

    checkCudaErrors(cudaMallocHost((void**)&outputs, BatchSize * one_example_out_length * sizeof(float)));

    checkCudaErrors(cudaMallocHost((void**)&out_losses, BatchSize * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_out_losses,   BatchSize * sizeof(float)));
    checkCudaErrors(cudaMemset(d_out_losses, 0, BatchSize * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_softmax_sums, BatchSize * sizeof(float)));
    checkCudaErrors(cudaMemset(d_softmax_sums, 0, BatchSize * sizeof(float)));

    checkCudaErrors(cudaMalloc((void**)&d_example_in_data, examples_in_size));
    checkCudaErrors(cudaMemcpy(d_example_in_data, example_in_data, examples_in_size, cudaMemcpyHostToDevice));
    
    checkCudaErrors(cudaMalloc((void**)&d_example_out_data, examples_out_size));
    checkCudaErrors(cudaMemcpy(d_example_out_data, example_out_data, examples_out_size, cudaMemcpyHostToDevice));
    
    for (int i = 0; i < Layers.size(); i++) {
        if (Layers[i]->Type != LayerTypePool) {
            checkCudaErrors(cudaMemset(DevWeightGradients[i], 0, Layers[i]->WeightsLen() * sizeof(float)));
            checkCudaErrors(cudaMemset(DevBiasGradients[i],   0, Layers[i]->OutLayerLen() * sizeof(float)));
        }
    }
    
    int n_blocks, block_size;
    
    for (int i = 0; i < BatchSize; i++) {
        CudaStream& cStream = CudaStreams[i % ConcurrentKernelCount];
        cudaStream_t stream = cStream.Stream;
        
        float *d_ex_in = &d_example_in_data[i * one_example_in_length];
        
        checkCudaErrors(cudaMemcpyAsync(
            BatchForwardIn[i], d_ex_in,
            one_example_in_length * sizeof(float), cudaMemcpyDeviceToDevice, stream));

        checkCudaErrors(cudaMemcpyAsync(
            BatchActivations[i][0], d_ex_in,
            one_example_in_length * sizeof(float), cudaMemcpyDeviceToDevice, stream));

        // Forward propagation

        int sz, out_len = 0;

        for (int j = 0; j < Layers.size(); j++) {
            Layer *layer = Layers[j];
            
            DenseLayer *dense;
            ConvLayer *conv;
            PoolLayer *pool;
            
            int activation = layer->Activation;
                        
            switch (layer->Type) {
                case LayerTypeDense: {
                    dense = (DenseLayer*)layer;

                    out_len = dense->OutputLen;
                    n_blocks = LinearBlockCountEx(dense->OutputLen, MmTileDim);
                    block_size = LinearBlockSizeEx(dense->OutputLen, MmTileDim);
                    
                    constexpr int block = MmTileDim;
    
                    dim3 block_dim(block, block);
                    dim3 blocks(1, AlignBlock(out_len, block));
                        
                    KernelMatrixMultiply<<<blocks, block_dim, 0, stream>>>(
                        BatchForwardOut[i], dense->DeviceWeights, BatchForwardIn[i],
                        1, out_len,
                        dense->InputLen, out_len,
                        1, dense->InputLen);
                    
                    KernelAddTensors<<<n_blocks, block_size, 0, stream>>>
                        (BatchForwardOut[i], BatchForwardOut[i], dense->DeviceBias, dense->OutputLen);
                    break;
                }
                    
                case LayerTypeConv: {
                    conv = (ConvLayer*)layer;
                    
                    int kd = conv->FilterCount,   ow = conv->OutputW(), oh = conv->OutputH();
                    int ic = conv->InputChannels, iw = conv->InputW,    ih = conv->InputH;
                    int kw = conv->KernelW,       kh = conv->KernelH;
                    int padding = conv->Padding;
                    
                    dim3 block_dim(2, 16, 16);
                    dim3 blocks(AlignBlock(kd, block_dim.x), AlignBlock(ow, block_dim.y), AlignBlock(oh, block_dim.z));
                    
                    KernelConv<<<blocks, block_dim, 0, stream>>>(
                        BatchForwardOut[i], BatchForwardIn[i], conv->DeviceWeights, conv->DeviceBias,
                        iw, ih, ic, padding,
                        kw, kh, kd);
                        
                    out_len = conv->OutLayerLen();

                    break;
                }
                
                case LayerTypePool: {
                    pool = (PoolLayer*)layer;
                    
                    int ow = pool->OutputW(), oh = pool->OutputH();
                    int ic = pool->Channels, iw = pool->InputW, ih = pool->InputH;
                    int ks = pool->KernelSize;
                    
                    out_len = pool->OutLayerLen();
                    
                    dim3 block_dim(16, 16, 2);
                    dim3 blocks(AlignBlock(ow, block_dim.x), AlignBlock(oh, block_dim.y), AlignBlock(ic, block_dim.z));
                    
                    KernelPool<<<blocks, block_dim, 0, stream>>>(BatchForwardOut[i], BatchForwardIn[i], iw, ih, ic, ks);
                    
                    break;
                }
            }
            
            sz = out_len * sizeof(float);
            
            CudaCoreActivationFunc(stream, BatchActPrimes[i][j], BatchForwardOut[i], out_len, activation, true);

            CudaCoreActivationFunc(stream, BatchActivations[i][j + 1], BatchForwardOut[i], out_len, activation, false);

            checkCudaErrors(cudaMemcpyAsync(BatchForwardOut[i], BatchActivations[i][j + 1], sz, cudaMemcpyDeviceToDevice, stream));
            
            std::swap(BatchForwardOut[i], BatchForwardIn[i]);
        }
        
        int n_blocks = LinearBlockCountEx(out_len, MmTileDim);
        int block_size = LinearBlockSizeEx(out_len, MmTileDim);
    
        const int last = Layers.size() - 1;
        
        checkCudaErrors(cudaMemcpyAsync(
            outputs + one_example_out_length * i,
            BatchForwardIn[i], sz, cudaMemcpyDeviceToHost, stream));
        
        KernelSoftmaxSum<<<n_blocks, block_size, 0, stream>>>(d_softmax_sums + i, BatchForwardIn[i], out_len);
        KernelSoftmaxDeltaAndCrossEntropy<<<n_blocks, block_size, 0, stream>>>(
            BatchPartialDeltaIn[i], d_out_losses + i, BatchForwardIn[i],
            d_example_out_data + one_example_out_length * i, d_softmax_sums + i, out_len);
        
        // Backpropagation
        
        for (int j = last; j >= 0; j--) {
            Layer *layer = Layers[j];
            
            DenseLayer *dense;
            ConvLayer *conv;
            PoolLayer *pool;
            
            int n_blocks, block_size;
            int bias_length, weights_length;

            switch (layer->Type) {
                case LayerTypeDense: {
                    dense = (DenseLayer*)layer;
                    
                    int in_len = dense->InputLen;
                    out_len = dense->OutputLen;
                    
                    weights_length = dense->WeightsLen();
                    bias_length = out_len;
                    
                    n_blocks = LinearBlockCountEx(out_len, MmTileDim);
                    block_size = LinearBlockSizeEx(out_len, MmTileDim);
                    
                    KernelHaramardMul<<<n_blocks, block_size, 0, stream>>>
                        (BatchCurBiasGradients[i], BatchPartialDeltaIn[i], BatchActPrimes[i][j], out_len);
                                 
                    constexpr int block1 = MmTileDim;

                    dim3 block_dim(block1, block1);
                    dim3 blocks(AlignBlock(in_len, block1), AlignBlock(out_len, block1));
                        
                    KernelMatrixMultiply<<<blocks, block_dim, 0, stream>>>(
                        BatchCurWeightGradients[i], BatchCurBiasGradients[i], BatchActivations[i][j],
                        in_len, out_len,
                        1, out_len,
                        in_len, 1);
                        
                    blocks = dim3(AlignBlock(out_len, MtTileDim), AlignBlock(in_len, MtTileDim));
                    block_dim = dim3(MtTileDim, MtBlockRows);
                    
                    KernelMatrixTranspose<<<blocks, block_dim, 0, stream>>>
                        (BatchWeightsT[i], dense->DeviceWeights, in_len, out_len);

                    blocks = dim3(1, AlignBlock(in_len, block1));
                    block_dim = dim3(block1, block1);
                    
                    KernelMatrixMultiply<<<blocks, block_dim, 0, stream>>>(
                        BatchPartialDeltaOut[i], BatchWeightsT[i], BatchCurBiasGradients[i],
                        1, in_len,
                        out_len, in_len,
                        1, out_len);
                    break;
                }
                
                case LayerTypeConv: {
                    conv = (ConvLayer*)layer;

                    out_len = conv->OutLayerLen();

                    weights_length = layer->WeightsLen();
                    bias_length = out_len;
                    
                    int kd = conv->FilterCount, kw = conv->KernelW, kh = conv->KernelH;
                    int ic = conv->InputChannels, iw = conv->InputW, ih = conv->InputH;
                    int padding = conv->Padding;
    
                    n_blocks = LinearBlockCount(out_len);
                    block_size = LinearBlockSize(out_len);
                    
                    KernelHaramardMul<<<n_blocks, block_size, 0, stream>>>
                        (BatchCurBiasGradients[i], BatchPartialDeltaIn[i], BatchActPrimes[i][j], out_len);
                    
                    dim3 block_dim(2, 16, 16);
                    dim3 blocks(AlignBlock(ic * kd, block_dim.x), AlignBlock(kw, block_dim.y), AlignBlock(kh, block_dim.z));
                    
                    KernelConvBackprop1<<<blocks, block_dim, 0, stream>>>(
                        BatchCurWeightGradients[i], BatchCurBiasGradients[i], BatchActivations[i][j],
                        iw, ih, ic, padding,
                        kw, kh, kd);
                    
                    block_dim = dim3(16, 16, 2);
                    blocks = dim3(AlignBlock(iw, block_dim.x), AlignBlock(ih, block_dim.y), AlignBlock(ic, block_dim.z));
                    
                    KernelConvBackprop2<<<blocks, block_dim, 0, stream>>>(
                        BatchPartialDeltaOut[i], conv->DeviceWeights, BatchCurBiasGradients[i],
                        iw, ih, ic, padding,
                        kw, kh, kd);
                    break;
                }
                    
                case LayerTypePool: {
                    pool = (PoolLayer*)layer;
                    
                    int w = pool->InputW, h = pool->InputH, ks = pool->KernelSize, c = pool->Channels;

                    dim3 block_dim(16, 16, 2);
                    dim3 blocks(AlignBlock(w, block_dim.x), AlignBlock(h, block_dim.y), AlignBlock(c, block_dim.z));
                    
                    KernelPoolBackprop<<<blocks, block_dim, 0, stream>>>(BatchPartialDeltaOut[i], BatchPartialDeltaIn[i], w, h, c, ks);
                    break;
                }
            }
            
            if (layer->Type != LayerTypePool) {
                int m = std::max(bias_length, weights_length);
                
                n_blocks = LinearBlockCountEx(m, MmTileDim);
                block_size = LinearBlockSizeEx(m, MmTileDim);
                
                KernelUpdateGradients<<<n_blocks, block_size, 0, stream>>>(
                    DevWeightGradients[j], DevBiasGradients[j],
                    BatchCurWeightGradients[i], BatchCurBiasGradients[i], BatchSize, bias_length, weights_length);
            }
            
            std::swap(BatchPartialDeltaOut[i], BatchPartialDeltaIn[i]);
        }
    }
    
    checkCudaErrors(cudaMemcpy(out_losses, d_out_losses, BatchSize * sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaDeviceSynchronize();
    
    int sid = 0;
    
    float if1 = 1.f / (1 - (float)pow(b1, t));
    float if2 = 1.f / (1 - (float)pow(b2, t));
    
    for (int i = 0; i < Layers.size(); i++) {
        Layer *layer = Layers[i];
        
        if (layer->Type != LayerTypePool) {
            int bias_length = layer->OutLayerLen();
            int weights_length = layer->WeightsLen();
            
            int m = std::max(bias_length, weights_length);
                    
            int n_blocks = LinearBlockCountEx(m, MmTileDim);
            int block_size = LinearBlockSizeEx(m, MmTileDim);
                       
            KernelAdamStep<<<n_blocks, block_size, 0, CudaStreams[sid].Stream>>>
                (layer->DeviceWeights, layer->DeviceBias,
                 DevWeightGradients[i], DevWeightGradientsM[i], DevWeightGradientsV[i],
                 DevBiasGradients[i], DevBiasGradientsM[i], DevBiasGradientsV[i],
                 b1, b2, if1, if2, learning_rate, weights_length, bias_length);
            
            sid++;
            sid %= ConcurrentKernelCount;
        }
    }
    
    cudaDeviceSynchronize();
        
    *out_avg_loss = 0;
    *out_correct_count = 0;

    int k = 0;
    
    for (int i = 0; i < BatchSize; i++) {
        *out_avg_loss += out_losses[i] / BatchSize;
        
        float ovmx = -1e12; int oimx = -1;
        float evmx = -1e12; int eimx = -1;
                
        for (int j = 0; j < one_example_out_length; j++, k++) {
            if (outputs[k] > ovmx) {
                ovmx = outputs[k];
                oimx = j;
            }
            
            if (example_out_data[k] > evmx) {
                evmx = example_out_data[k];
                eimx = j;
            }
        }
        
        if (eimx == oimx)
            (*out_correct_count)++;
    }
    
    checkCudaErrors(cudaFree(d_example_in_data));
    checkCudaErrors(cudaFree(d_example_out_data));    
    checkCudaErrors(cudaFree(d_out_losses));    
    checkCudaErrors(cudaFree(d_softmax_sums));    
}

/**
 * Program main
 */
int main(int argc, char **argv)
{
    return 0;
}