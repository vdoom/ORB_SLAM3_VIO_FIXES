/**
 * CUDA-accelerated ORB extraction kernels for ORB-SLAM3
 *
 * Contains:
 *  - ICAngleKernel:              keypoint orientation via intensity centroid
 *  - ComputeOrbDescriptorKernel: 256-bit rotated BRIEF descriptor
 *  - GpuPyramidBuilder:          image pyramid + Gaussian blur on GPU
 *  - Host wrapper functions for the above
 *
 * Build with: cmake -DUSE_CUDA=ON
 */

#include "cuda/ORBextractor_cuda.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

namespace ORB_SLAM3 {
namespace cuda {

// ---------------------------------------------------------------------------
// Constant memory: loaded once, read by all threads
// ---------------------------------------------------------------------------

// BRIEF sampling pattern: 512 points stored as 1024 ints (x0,y0, x1,y1, ...)
// Each descriptor byte uses 8 point-pairs = 16 points = 32 ints.
// Total: 32 bytes * 32 ints = 1024 ints.
__constant__ int c_bit_pattern[1024];

// Circular patch boundary for IC_Angle: umax[v] = max u at row v in the 31x31 patch
// Size: HALF_PATCH_SIZE + 1 = 16 entries
__constant__ int c_umax[16];

// Module-level init flag
static bool g_initialized = false;

// ---------------------------------------------------------------------------
// Error checking helper
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[ORB-CUDA] %s:%d CUDA error: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

#define CUDA_CHECK_BOOL(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[ORB-CUDA] %s:%d CUDA error: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return false; \
    } \
} while(0)

// ---------------------------------------------------------------------------
// CUDA Kernel: IC_Angle (keypoint orientation)
// ---------------------------------------------------------------------------
// One thread per keypoint. Computes intensity centroid angle over a
// 31x31 circular patch. Matches CPU IC_Angle() in ORBextractor.cc:76-103.

__global__ void ICAngleKernel(
    const unsigned char* __restrict__ image,
    int img_step,
    const float* __restrict__ kp_x,
    const float* __restrict__ kp_y,
    float* __restrict__ angles,
    int num_keypoints,
    int half_patch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_keypoints) return;

    int cx = __float2int_rn(kp_x[idx]);
    int cy = __float2int_rn(kp_y[idx]);
    const unsigned char* center = image + cy * img_step + cx;

    int m_01 = 0, m_10 = 0;

    // Center line (v=0): sum u * pixel[u]
    for (int u = -half_patch_size; u <= half_patch_size; ++u)
        m_10 += u * (int)center[u];

    // Symmetric lines v=1..HALF_PATCH_SIZE
    for (int v = 1; v <= half_patch_size; ++v) {
        int v_sum = 0;
        int d = c_umax[v];
        for (int u = -d; u <= d; ++u) {
            int val_plus  = (int)center[u + v * img_step];
            int val_minus = (int)center[u - v * img_step];
            v_sum += (val_plus - val_minus);
            m_10  += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    // atan2 in degrees to match OpenCV's fastAtan2 output
    angles[idx] = atan2f((float)m_01, (float)m_10) * (180.0f / 3.14159265358979323846f);
}

// ---------------------------------------------------------------------------
// CUDA Kernel: ORB Descriptor computation
// ---------------------------------------------------------------------------
// One thread per keypoint. Computes 32-byte (256-bit) rotated BRIEF descriptor.
// Matches CPU computeOrbDescriptor() in ORBextractor.cc:107-146.
//
// Pattern layout in c_bit_pattern[1024]:
//   512 cv::Points stored as (x0,y0, x1,y1, ...) = 1024 ints.
//   Per descriptor byte: 8 point-pairs = 16 points = 32 ints.
//   Byte i uses c_bit_pattern[i*32 .. i*32+31].
//
// CPU GET_VALUE(idx):
//   row = cvRound(pattern[idx].x * b + pattern[idx].y * a)
//   col = cvRound(pattern[idx].x * a - pattern[idx].y * b)
//   value = center[row * step + col]
// Where a = cos(angle), b = sin(angle).

__global__ void ComputeOrbDescriptorKernel(
    const unsigned char* __restrict__ image,
    int img_step,
    const float* __restrict__ kp_x,
    const float* __restrict__ kp_y,
    const float* __restrict__ kp_angle,
    unsigned char* __restrict__ descriptors,
    int num_keypoints)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_keypoints) return;

    float angle = kp_angle[idx] * (3.14159265358979323846f / 180.0f);
    float a = cosf(angle), b = sinf(angle);

    int cx = __float2int_rn(kp_x[idx]);
    int cy = __float2int_rn(kp_y[idx]);
    const unsigned char* center = image + cy * img_step + cx;

    unsigned char* desc = descriptors + idx * 32;

    // 32 bytes per descriptor
    for (int i = 0; i < 32; ++i) {
        // Each byte uses 16 points (8 pairs). Pattern base = i * 32 ints.
        const int* pat = c_bit_pattern + i * 32;
        int val = 0;

        // 8 bit-pairs per byte
        for (int bit = 0; bit < 8; ++bit) {
            // Each pair = 2 points = 4 ints: (x0,y0, x1,y1)
            int p = bit * 4;
            int px0 = pat[p],   py0 = pat[p + 1];
            int px1 = pat[p + 2], py1 = pat[p + 3];

            // Rotate pattern point by keypoint orientation
            // Matches CPU: row = round(x*b + y*a), col = round(x*a - y*b)
            int row0 = __float2int_rn(px0 * b + py0 * a);
            int col0 = __float2int_rn(px0 * a - py0 * b);
            int row1 = __float2int_rn(px1 * b + py1 * a);
            int col1 = __float2int_rn(px1 * a - py1 * b);

            unsigned char t0 = center[row0 * img_step + col0];
            unsigned char t1 = center[row1 * img_step + col1];

            if (t0 < t1)
                val |= (1 << bit);
        }
        desc[i] = (unsigned char)val;
    }
}

// ---------------------------------------------------------------------------
// Host: Constant memory initialization
// ---------------------------------------------------------------------------

void InitConstantMemory(const int* bit_pattern, int pattern_size,
                        const int* umax, int umax_size)
{
    if (pattern_size > 1024) {
        fprintf(stderr, "[ORB-CUDA] bit_pattern too large: %d (max 1024)\n", pattern_size);
        return;
    }
    if (umax_size > 16) {
        fprintf(stderr, "[ORB-CUDA] umax too large: %d (max 16)\n", umax_size);
        return;
    }

    CUDA_CHECK(cudaMemcpyToSymbol(c_bit_pattern, bit_pattern,
                                   pattern_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_umax, umax,
                                   umax_size * sizeof(int)));

    g_initialized = true;
    fprintf(stdout, "[ORB-CUDA] Constant memory initialized (pattern=%d ints, umax=%d ints)\n",
            pattern_size, umax_size);
}

// ---------------------------------------------------------------------------
// Host: Availability check
// ---------------------------------------------------------------------------

bool IsAvailable()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        fprintf(stdout, "[ORB-CUDA] No CUDA devices found, falling back to CPU\n");
        return false;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    fprintf(stdout, "[ORB-CUDA] Using GPU: %s (SM %d.%d, %d cores)\n",
            prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    return true;
}

// ---------------------------------------------------------------------------
// GpuPyramidBuilder
// ---------------------------------------------------------------------------

GpuPyramidBuilder::GpuPyramidBuilder(int nlevels, float scaleFactor, int edgeThreshold)
    : nlevels_(nlevels), edgeThreshold_(edgeThreshold)
{
    gpuPyramid_.resize(nlevels);
    gpuPyramidBordered_.resize(nlevels);
    gpuBlurred_.resize(nlevels);

    // Pre-create the Gaussian filter (7x7, sigma=2, same as CPU path)
    gaussianFilter_ = cv::cuda::createGaussianFilter(
        CV_8UC1, CV_8UC1, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);
}

GpuPyramidBuilder::~GpuPyramidBuilder() {}

void GpuPyramidBuilder::Build(const cv::Mat& image,
                               std::vector<cv::Mat>& mvImagePyramid,
                               const std::vector<float>& invScaleFactors)
{
    // Upload input image to GPU
    gpuImage_.upload(image, stream_);

    // Temporary GpuMat for resize output (avoids aliasing with bordered mat)
    cv::cuda::GpuMat gpuResized;

    for (int level = 0; level < nlevels_; ++level)
    {
        float scale = invScaleFactors[level];
        cv::Size sz(cvRound((float)image.cols * scale),
                    cvRound((float)image.rows * scale));

        if (level == 0)
        {
            // Level 0: copyMakeBorder from source image into bordered mat
            cv::cuda::copyMakeBorder(gpuImage_, gpuPyramidBordered_[level],
                                     edgeThreshold_, edgeThreshold_,
                                     edgeThreshold_, edgeThreshold_,
                                     cv::BORDER_REFLECT_101, cv::Scalar(), stream_);
        }
        else
        {
            // Resize from previous level's ROI into a temporary
            cv::cuda::resize(gpuPyramid_[level - 1], gpuResized,
                             sz, 0, 0, cv::INTER_LINEAR, stream_);

            // copyMakeBorder from temporary into bordered mat
            cv::cuda::copyMakeBorder(gpuResized, gpuPyramidBordered_[level],
                                     edgeThreshold_, edgeThreshold_,
                                     edgeThreshold_, edgeThreshold_,
                                     cv::BORDER_REFLECT_101 | cv::BORDER_ISOLATED,
                                     cv::Scalar(), stream_);
        }

        // Set the pyramid level as ROI into the bordered mat (no copy)
        gpuPyramid_[level] = gpuPyramidBordered_[level](
            cv::Rect(edgeThreshold_, edgeThreshold_, sz.width, sz.height));

        // Gaussian blur for descriptor computation (keep on GPU)
        gaussianFilter_->apply(gpuPyramid_[level], gpuBlurred_[level], stream_);
    }

    // Download pyramid to CPU (preserving bordered-parent layout)
    // Frame.cc accesses mvImagePyramid[level] as ROI into a bordered parent.
    // We replicate this layout: allocate CPU temp with border, set mvImagePyramid as ROI.
    stream_.waitForCompletion();

    for (int level = 0; level < nlevels_; ++level)
    {
        float scale = invScaleFactors[level];
        cv::Size sz(cvRound((float)image.cols * scale),
                    cvRound((float)image.rows * scale));
        cv::Size wholeSize(sz.width + edgeThreshold_ * 2,
                           sz.height + edgeThreshold_ * 2);

        cv::Mat temp(wholeSize, image.type());
        gpuPyramidBordered_[level].download(temp);

        // mvImagePyramid[level] is the inner ROI (same as CPU ComputePyramid)
        mvImagePyramid[level] = temp(cv::Rect(edgeThreshold_, edgeThreshold_,
                                              sz.width, sz.height));
    }
}

// ---------------------------------------------------------------------------
// Host: Orientation computation (IC_Angle) on GPU
// ---------------------------------------------------------------------------

void ComputeOrientationGPU(
    const std::vector<cv::cuda::GpuMat>& gpuPyramid,
    std::vector<std::vector<cv::KeyPoint>>& allKeypoints,
    int nlevels)
{
    const int HALF_PATCH = 15;
    const int BLOCK_SIZE = 256;

    for (int level = 0; level < nlevels; ++level)
    {
        std::vector<cv::KeyPoint>& keypoints = allKeypoints[level];
        int nkp = (int)keypoints.size();
        if (nkp == 0) continue;

        // Gather keypoint coordinates into flat arrays
        std::vector<float> h_kp_x(nkp), h_kp_y(nkp);
        for (int i = 0; i < nkp; ++i) {
            h_kp_x[i] = keypoints[i].pt.x;
            h_kp_y[i] = keypoints[i].pt.y;
        }

        // Allocate device arrays
        float *d_kp_x = nullptr, *d_kp_y = nullptr, *d_angles = nullptr;
        cudaMalloc(&d_kp_x, nkp * sizeof(float));
        cudaMalloc(&d_kp_y, nkp * sizeof(float));
        cudaMalloc(&d_angles, nkp * sizeof(float));

        cudaMemcpy(d_kp_x, h_kp_x.data(), nkp * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kp_y, h_kp_y.data(), nkp * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        int gridSize = (nkp + BLOCK_SIZE - 1) / BLOCK_SIZE;
        ICAngleKernel<<<gridSize, BLOCK_SIZE>>>(
            gpuPyramid[level].ptr<unsigned char>(),
            (int)gpuPyramid[level].step,
            d_kp_x, d_kp_y, d_angles,
            nkp, HALF_PATCH);

        // Download angles
        std::vector<float> h_angles(nkp);
        cudaMemcpy(h_angles.data(), d_angles, nkp * sizeof(float), cudaMemcpyDeviceToHost);

        // Write back into keypoints
        for (int i = 0; i < nkp; ++i)
            keypoints[i].angle = h_angles[i];

        cudaFree(d_kp_x);
        cudaFree(d_kp_y);
        cudaFree(d_angles);
    }
}

// ---------------------------------------------------------------------------
// Host: Descriptor computation on GPU
// ---------------------------------------------------------------------------

void ComputeDescriptorsGPU(
    const std::vector<cv::cuda::GpuMat>& gpuBlurred,
    const std::vector<std::vector<cv::KeyPoint>>& allKeypoints,
    std::vector<cv::Mat>& levelDescriptors,
    int nlevels)
{
    const int BLOCK_SIZE = 256;

    levelDescriptors.resize(nlevels);

    for (int level = 0; level < nlevels; ++level)
    {
        const std::vector<cv::KeyPoint>& keypoints = allKeypoints[level];
        int nkp = (int)keypoints.size();

        if (nkp == 0) {
            levelDescriptors[level] = cv::Mat();
            continue;
        }

        // Gather keypoint data
        std::vector<float> h_kp_x(nkp), h_kp_y(nkp), h_kp_angle(nkp);
        for (int i = 0; i < nkp; ++i) {
            h_kp_x[i] = keypoints[i].pt.x;
            h_kp_y[i] = keypoints[i].pt.y;
            h_kp_angle[i] = keypoints[i].angle;
        }

        // Allocate device arrays
        float *d_kp_x = nullptr, *d_kp_y = nullptr, *d_kp_angle = nullptr;
        unsigned char *d_desc = nullptr;

        cudaMalloc(&d_kp_x, nkp * sizeof(float));
        cudaMalloc(&d_kp_y, nkp * sizeof(float));
        cudaMalloc(&d_kp_angle, nkp * sizeof(float));
        cudaMalloc(&d_desc, nkp * 32 * sizeof(unsigned char));
        cudaMemset(d_desc, 0, nkp * 32);

        cudaMemcpy(d_kp_x, h_kp_x.data(), nkp * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kp_y, h_kp_y.data(), nkp * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kp_angle, h_kp_angle.data(), nkp * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        int gridSize = (nkp + BLOCK_SIZE - 1) / BLOCK_SIZE;
        ComputeOrbDescriptorKernel<<<gridSize, BLOCK_SIZE>>>(
            gpuBlurred[level].ptr<unsigned char>(),
            (int)gpuBlurred[level].step,
            d_kp_x, d_kp_y, d_kp_angle,
            d_desc, nkp);

        // Download descriptors
        levelDescriptors[level] = cv::Mat(nkp, 32, CV_8UC1);
        cudaMemcpy(levelDescriptors[level].data, d_desc,
                   nkp * 32 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        cudaFree(d_kp_x);
        cudaFree(d_kp_y);
        cudaFree(d_kp_angle);
        cudaFree(d_desc);
    }
}

} // namespace cuda
} // namespace ORB_SLAM3
