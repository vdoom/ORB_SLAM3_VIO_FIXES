/**
 * CUDA-accelerated ORB extraction helpers for ORB-SLAM3
 *
 * Accelerates: image pyramid construction, GaussianBlur,
 * keypoint orientation (IC_Angle), and ORB descriptor computation.
 * FAST detection and OctTree distribution remain on CPU.
 *
 * Build with: cmake -DUSE_CUDA=ON
 */

#ifndef ORB_EXTRACTOR_CUDA_H
#define ORB_EXTRACTOR_CUDA_H

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace ORB_SLAM3 {
namespace cuda {

/**
 * Initialize CUDA constant memory with the BRIEF bit pattern and
 * the circular patch boundary table (umax) for IC_Angle.
 * Must be called once before any GPU extraction.
 *
 * @param bit_pattern  The 1024-int BRIEF sampling pattern (bit_pattern_31_)
 * @param pattern_size Number of ints (must be 1024)
 * @param umax         Circular patch boundary lookup table
 * @param umax_size    Number of entries in umax (HALF_PATCH_SIZE + 1 = 16)
 */
void InitConstantMemory(const int* bit_pattern, int pattern_size,
                        const int* umax, int umax_size);

/**
 * Check if a CUDA-capable GPU is available and the module is initialized.
 */
bool IsAvailable();

/**
 * Manages GPU-side image pyramid construction and Gaussian blurring.
 *
 * Uploads the input image once, builds all pyramid levels on GPU
 * (resize + copyMakeBorder), applies GaussianBlur for descriptor
 * computation, and downloads the pyramid back to CPU cv::Mat
 * (preserving the bordered-parent memory layout that Frame.cc expects).
 */
class GpuPyramidBuilder {
public:
    GpuPyramidBuilder(int nlevels, float scaleFactor, int edgeThreshold);
    ~GpuPyramidBuilder();

    /**
     * Build the image pyramid on GPU and download to CPU.
     *
     * After this call:
     *  - mvImagePyramid contains CPU Mats in the same bordered-parent
     *    layout as the original CPU ComputePyramid().
     *  - Blurred pyramid levels are retained on GPU for descriptor computation.
     *  - Unblurred pyramid levels are retained on GPU for IC_Angle computation.
     */
    void Build(const cv::Mat& image,
               std::vector<cv::Mat>& mvImagePyramid,
               const std::vector<float>& invScaleFactors);

    /** Access unblurred GPU pyramid (for IC_Angle kernel). */
    const std::vector<cv::cuda::GpuMat>& GetPyramidLevels() const { return gpuPyramid_; }

    /** Access blurred GPU pyramid (for descriptor kernel). */
    const std::vector<cv::cuda::GpuMat>& GetBlurredLevels() const { return gpuBlurred_; }

private:
    int nlevels_;
    int edgeThreshold_;
    cv::cuda::GpuMat gpuImage_;
    std::vector<cv::cuda::GpuMat> gpuPyramid_;
    std::vector<cv::cuda::GpuMat> gpuPyramidBordered_;
    std::vector<cv::cuda::GpuMat> gpuBlurred_;
    cv::Ptr<cv::cuda::Filter> gaussianFilter_;
    cv::cuda::Stream stream_;
};

/**
 * Compute keypoint orientations (IC_Angle) on GPU.
 *
 * Launches one CUDA thread per keypoint. Each thread computes the
 * intensity centroid angle over a 31x31 circular patch.
 * Results are written back into keypoint.angle fields.
 *
 * @param gpuPyramid    Unblurred pyramid levels on GPU
 * @param allKeypoints  Per-level keypoint vectors (angles are modified in-place)
 * @param nlevels       Number of pyramid levels
 */
void ComputeOrientationGPU(
    const std::vector<cv::cuda::GpuMat>& gpuPyramid,
    std::vector<std::vector<cv::KeyPoint>>& allKeypoints,
    int nlevels);

/**
 * Compute ORB descriptors on GPU.
 *
 * Launches one CUDA thread per keypoint. Each thread computes a
 * 32-byte (256-bit) rotated BRIEF descriptor using the bit_pattern
 * stored in GPU constant memory.
 *
 * @param gpuBlurred      Blurred pyramid levels on GPU
 * @param allKeypoints    Per-level keypoint vectors (must have valid angles)
 * @param levelDescriptors Output: per-level descriptor Mats (CPU, nkeypoints x 32, CV_8U)
 * @param nlevels         Number of pyramid levels
 */
void ComputeDescriptorsGPU(
    const std::vector<cv::cuda::GpuMat>& gpuBlurred,
    const std::vector<std::vector<cv::KeyPoint>>& allKeypoints,
    std::vector<cv::Mat>& levelDescriptors,
    int nlevels);

} // namespace cuda
} // namespace ORB_SLAM3

#endif // ORB_EXTRACTOR_CUDA_H
