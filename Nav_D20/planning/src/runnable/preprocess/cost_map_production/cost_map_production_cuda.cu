#include "src/common/util/config_helper.h"
#include "src/common/util/log.h"
#include "src/runnable/preprocess/cost_map_production/cost_map_production_cuda.cuh"
#include "src/runnable/preprocess/cost_map_production/cost_map_production_cuda.h"

#include <npp.h>
#include <nppi_filtering_functions.h>
#include <nppi_statistics_functions.h>

using planning::common::util::ConfigHelper;

__global__ void scaleDistanceFieldKernel(float *distance_field, int size,
                                         float resolution) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  distance_field[idx] *= resolution;
}

__global__ void normalizeGradientKernel(float *grad_x, float *grad_y,
                                        int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  float mag = sqrtf(grad_x[idx] * grad_x[idx] + grad_y[idx] * grad_y[idx]);
  grad_x[idx] /= (mag + 1e-6f);
  grad_y[idx] /= (mag + 1e-6f);
}

__global__ void updateRefOffsetCostKernel(const float *ref_offset_dist,
                                          float *ref_offset_cost, int size,
                                          float max_reference_offset) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  float dist = fminf(ref_offset_dist[idx], max_reference_offset);
  ref_offset_cost[idx] =
      1.0f + (dist - max_reference_offset) / max_reference_offset;
}

__global__ void normalizeCompositeCostKernel(float *cost_field, int size,
                                             float max_dist) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  cost_field[idx] = 1.0f - (cost_field[idx] / max_dist);
}

__global__ void BuildCompositeObstacleKernel(
    const unsigned char *obstacle, const float *obstacle_distance,
    const float *ref_offset, unsigned char *composite_obstacle, int size,
    float inflation_safe_margin, float max_reference_offset,
    bool has_ref_path) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  if (obstacle[idx] < 128 || obstacle_distance[idx] <= inflation_safe_margin) {
    composite_obstacle[idx] = 0;
  } else if (has_ref_path && ref_offset[idx] >= max_reference_offset) {
    composite_obstacle[idx] = 0;
  } else {
    composite_obstacle[idx] = 255;
  }
}

namespace planning {
namespace preprocess {

bool CostMapProductionCuda::Init(int map_rows, int map_cols, double resolution,
                                 double map_position_x, double map_position_y,
                                 double map_length_x, double map_length_y) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  cudaError_t cudaError = cudaGetDevice(&nppStreamCtx.nCudaDeviceId);
  if (cudaError != cudaSuccess) {
    AERROR << "CUDA error: no devices supporting CUDA:"
           << cudaGetErrorString(cudaError);
    return false;
  }
  const NppLibraryVersion *libVer = nppGetLibVersion();
  AINFO << "NPP Library Version: " << libVer->major << "." << libVer->minor
        << "." << libVer->build;
  int driverVersion, runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);
  AINFO << "CUDA Driver  Version: " << driverVersion / 1000 << "."
        << (driverVersion % 100) / 10;
  AINFO << "CUDA Runtime Version: " << runtimeVersion / 1000 << "."
        << (runtimeVersion % 100) / 10;

  cudaError = cudaDeviceGetAttribute(
      &nppStreamCtx.nCudaDevAttrComputeCapabilityMajor,
      cudaDevAttrComputeCapabilityMajor, nppStreamCtx.nCudaDeviceId);
  if (cudaError != cudaSuccess) {
    AERROR << "CUDA error: cudaDeviceGetAttribute failed: "
           << cudaGetErrorString(cudaError);
    return false;
  }
  cudaError = cudaDeviceGetAttribute(
      &nppStreamCtx.nCudaDevAttrComputeCapabilityMinor,
      cudaDevAttrComputeCapabilityMinor, nppStreamCtx.nCudaDeviceId);
  if (cudaError != cudaSuccess) {
    AERROR << "CUDA error: cudaDeviceGetAttribute failed: "
           << cudaGetErrorString(cudaError);
    return false;
  }
  cudaError =
      cudaStreamGetFlags(nppStreamCtx.hStream, &nppStreamCtx.nStreamFlags);
  cudaDeviceProp oDeviceProperties;
  cudaError =
      cudaGetDeviceProperties(&oDeviceProperties, nppStreamCtx.nCudaDeviceId);
  nppStreamCtx.nMultiProcessorCount = oDeviceProperties.multiProcessorCount;
  nppStreamCtx.nMaxThreadsPerMultiProcessor =
      oDeviceProperties.maxThreadsPerMultiProcessor;
  nppStreamCtx.nMaxThreadsPerBlock = oDeviceProperties.maxThreadsPerBlock;
  nppStreamCtx.nSharedMemPerBlock = oDeviceProperties.sharedMemPerBlock;

  size_t block_size = map_rows * map_cols;
  resources_.block_size = block_size;
  resources_.npp_size_width = map_rows;
  resources_.npp_size_height = map_cols;

  NppiSize npp_size = {map_rows, map_cols};

  cudaError =
      cudaMalloc((void **)&resources_.cuda_input, block_size * sizeof(Npp8u));
  if (cudaError != cudaSuccess) {
    AERROR << "CUDA error: cudaMalloc cuda_input failed: "
           << cudaGetErrorString(cudaError);
    return false;
  }
  cudaError =
      cudaMalloc((void **)&resources_.cuda_output, block_size * sizeof(Npp32f));
  if (cudaError != cudaSuccess) {
    AERROR << "CUDA error: cudaMalloc cuda_output failed: "
           << cudaGetErrorString(cudaError);
    return false;
  }
  cudaError =
      cudaMalloc((void **)&resources_.cuda_grad_x, block_size * sizeof(Npp32f));
  if (cudaError != cudaSuccess) {
    AERROR << "CUDA error: cudaMalloc cuda_grad_x failed: "
           << cudaGetErrorString(cudaError);
    return false;
  }
  cudaError =
      cudaMalloc((void **)&resources_.cuda_grad_y, block_size * sizeof(Npp32f));
  if (cudaError != cudaSuccess) {
    AERROR << "CUDA error: cudaMalloc cuda_grad_y failed: "
           << cudaGetErrorString(cudaError);
    return false;
  }
  cudaError = cudaMalloc((void **)&resources_.cuda_composite_cost,
                         block_size * sizeof(Npp32f));
  if (cudaError != cudaSuccess) {
    AERROR << "CUDA error: cudaMalloc cuda_composite_cost failed: "
           << cudaGetErrorString(cudaError);
    return false;
  }
  cudaError = cudaMalloc((void **)&resources_.cuda_composite_obstacle,
                         block_size * sizeof(Npp8u));
  if (cudaError != cudaSuccess) {
    AERROR << "CUDA error: cudaMalloc cuda_composite_obstacle failed: "
           << cudaGetErrorString(cudaError);
    return false;
  }
  cudaError = cudaMalloc((void **)&resources_.cuda_ref_offset_dist,
                         block_size * sizeof(Npp32f));
  if (cudaError != cudaSuccess) {
    AERROR << "CUDA error: cudaMalloc cuda_ref_offset_dist failed: "
           << cudaGetErrorString(cudaError);
    return false;
  }
  cudaError = cudaMalloc((void **)&resources_.cuda_ref_offset_cost,
                         block_size * sizeof(Npp32f));
  if (cudaError != cudaSuccess) {
    AERROR << "CUDA error: cudaMalloc cuda_ref_offset_cost failed: "
           << cudaGetErrorString(cudaError);
    return false;
  }

  size_t device_buffer_size = 0;
  NppStatus nppStatus =
      nppiDistanceTransformPBAGetBufferSize(npp_size, &device_buffer_size);
  if (nppStatus != NPP_SUCCESS) {
    cudaError = cudaGetLastError();
    AERROR << "NPP error: nppiDistanceTransformPBAGetBufferSize failed: "
           << cudaGetErrorString(cudaError);
    return false;
  }
  cudaError =
      cudaMalloc((void **)&resources_.device_buffer, device_buffer_size);
  if (cudaError != cudaSuccess) {
    AERROR << "CUDA error: cudaMalloc device_buffer failed: "
           << cudaGetErrorString(cudaError);
    return false;
  }

  resources_.npp_stream_ctx = new NppStreamContext(nppStreamCtx);

  resources_.query_context.distance_field = resources_.cuda_output;
  resources_.query_context.composite_cost_field =
      resources_.cuda_composite_cost;
  resources_.query_context.grad_x_field = resources_.cuda_grad_x;
  resources_.query_context.grad_y_field = resources_.cuda_grad_y;
  resources_.query_context.ref_offset_dist_field =
      resources_.cuda_ref_offset_dist;
  resources_.query_context.ref_offset_cost_field =
      resources_.cuda_ref_offset_cost;
  resources_.query_context.map_rows = map_rows;
  resources_.query_context.map_cols = map_cols;
  resources_.query_context.resolution = static_cast<float>(resolution);
  resources_.query_context.origin_x = static_cast<float>(
      map_position_x + map_length_x * 0.5 - resolution * 0.5);
  resources_.query_context.origin_y = static_cast<float>(
      map_position_y + map_length_y * 0.5 - resolution * 0.5);

  auto circles = ConfigHelper::GetBoundingCircles(0.0, 0.0, 0.0);
  auto &fp = resources_.h_footprint;
  fp.num_circles = std::min((int)circles.size(), 8);
  fp.bounding_radius = static_cast<float>(
      hypot(ConfigHelper::GetConfig().robot_param().length() / 2.0,
            ConfigHelper::GetConfig().robot_param().width() / 2.0));
  for (int i = 0; i < fp.num_circles; ++i) {
    fp.offsets_x[i] = static_cast<float>(circles[i].center_x);
    fp.offsets_y[i] = static_cast<float>(circles[i].center_y);
    fp.radii[i] = static_cast<float>(circles[i].radius);
  }
  cudaError = cudaMalloc(&resources_.d_footprint, sizeof(FootprintCircles));
  if (cudaError != cudaSuccess) {
    AERROR << "CUDA error: cudaMalloc d_footprint failed: "
           << cudaGetErrorString(cudaError);
    return false;
  }
  cudaError = cudaMemcpy(resources_.d_footprint, &fp, sizeof(FootprintCircles),
                         cudaMemcpyHostToDevice);
  if (cudaError != cudaSuccess) {
    AERROR << "CUDA error: cudaMemcpy d_footprint failed: "
           << cudaGetErrorString(cudaError);
    return false;
  }

  resources_.initialized = true;
  AINFO << "CostMapProductionCuda initialized, map: " << map_rows << "x"
        << map_cols << ", resolution: " << resolution << ", origin: ("
        << resources_.query_context.origin_x << ", "
        << resources_.query_context.origin_y << ")";
  return true;
}

void CostMapProductionCuda::Release() {
  if (resources_.cuda_input != nullptr) {
    cudaFree(resources_.cuda_input);
    resources_.cuda_input = nullptr;
  }
  if (resources_.cuda_output != nullptr) {
    cudaFree(resources_.cuda_output);
    resources_.cuda_output = nullptr;
  }
  if (resources_.cuda_grad_x != nullptr) {
    cudaFree(resources_.cuda_grad_x);
    resources_.cuda_grad_x = nullptr;
  }
  if (resources_.cuda_grad_y != nullptr) {
    cudaFree(resources_.cuda_grad_y);
    resources_.cuda_grad_y = nullptr;
  }
  if (resources_.cuda_composite_cost != nullptr) {
    cudaFree(resources_.cuda_composite_cost);
    resources_.cuda_composite_cost = nullptr;
  }
  if (resources_.cuda_composite_obstacle != nullptr) {
    cudaFree(resources_.cuda_composite_obstacle);
    resources_.cuda_composite_obstacle = nullptr;
  }
  if (resources_.cuda_ref_offset_dist != nullptr) {
    cudaFree(resources_.cuda_ref_offset_dist);
    resources_.cuda_ref_offset_dist = nullptr;
  }
  if (resources_.cuda_ref_offset_cost != nullptr) {
    cudaFree(resources_.cuda_ref_offset_cost);
    resources_.cuda_ref_offset_cost = nullptr;
  }
  if (resources_.device_buffer != nullptr) {
    cudaFree(resources_.device_buffer);
    resources_.device_buffer = nullptr;
  }
  if (resources_.d_footprint != nullptr) {
    cudaFree(resources_.d_footprint);
    resources_.d_footprint = nullptr;
  }
  if (resources_.npp_stream_ctx != nullptr) {
    delete static_cast<NppStreamContext *>(resources_.npp_stream_ctx);
    resources_.npp_stream_ctx = nullptr;
  }
  resources_.initialized = false;
}

bool CostMapProductionCuda::ProcObstacle(const unsigned char *host_obstacle,
                                         int rows, int cols, double safe_margin,
                                         double inflation_margin,
                                         double max_obstacle_dist,
                                         double resolution) {
  cudaGetLastError();

  NppiSize npp_size = {resources_.npp_size_width, resources_.npp_size_height};
  NppStreamContext &nppStreamCtx =
      *static_cast<NppStreamContext *>(resources_.npp_stream_ctx);

  cudaError_t cudaError =
      cudaMemcpy(resources_.cuda_input, host_obstacle,
                 resources_.block_size * sizeof(Npp8u), cudaMemcpyHostToDevice);
  if (cudaError != cudaSuccess) {
    AERROR << "CUDA error: cudaMemcpy input failed: "
           << cudaGetErrorString(cudaError);
    return false;
  }

  NppStatus nppStatus = nppiDistanceTransformPBA_8u32f_C1R_Ctx(
      resources_.cuda_input, npp_size.width * sizeof(Npp8u), 0, 0, 0, 0, 0, 0,
      0, 0, resources_.cuda_output, npp_size.width * sizeof(Npp32f), npp_size,
      resources_.device_buffer, nppStreamCtx);
  if (nppStatus != NPP_SUCCESS) {
    cudaError = cudaGetLastError();
    AERROR << "NPP error: nppiDistanceTransformPBA_8u32f_C1R_Ctx failed: "
           << cudaGetErrorString(cudaError);
    return false;
  }

  int block_dim = 256;
  int grid_dim = (resources_.block_size + block_dim - 1) / block_dim;
  scaleDistanceFieldKernel<<<grid_dim, block_dim>>>(
      resources_.cuda_output, resources_.block_size,
      static_cast<float>(resolution));

  return true;
}

void CostMapProductionCuda::DownloadDistanceField(float *host_distance,
                                                  int size) {
  cudaMemcpy(host_distance, resources_.cuda_output, size * sizeof(float),
             cudaMemcpyDeviceToHost);
}

void CostMapProductionCuda::DownloadCostField(float *host_cost, int size) {
  cudaMemcpy(host_cost, resources_.cuda_composite_cost, size * sizeof(float),
             cudaMemcpyDeviceToHost);
}

void CostMapProductionCuda::DownloadGradientXField(float *host_grad_x,
                                                   int size) {
  cudaMemcpy(host_grad_x, resources_.cuda_grad_x, size * sizeof(float),
             cudaMemcpyDeviceToHost);
}

void CostMapProductionCuda::DownloadGradientYField(float *host_grad_y,
                                                   int size) {
  cudaMemcpy(host_grad_y, resources_.cuda_grad_y, size * sizeof(float),
             cudaMemcpyDeviceToHost);
}

bool CostMapProductionCuda::ProcReference(const unsigned char *host_ref_path,
                                          int rows, int cols,
                                          double max_reference_offset,
                                          double resolution) {
  cudaGetLastError();

  NppiSize npp_size = {resources_.npp_size_width, resources_.npp_size_height};
  NppStreamContext &nppStreamCtx =
      *static_cast<NppStreamContext *>(resources_.npp_stream_ctx);

  cudaError_t cudaError =
      cudaMemcpy(resources_.cuda_input, host_ref_path,
                 resources_.block_size * sizeof(Npp8u), cudaMemcpyHostToDevice);
  if (cudaError != cudaSuccess) {
    AERROR << "CUDA error: cudaMemcpy ref_path input failed: "
           << cudaGetErrorString(cudaError);
    return false;
  }

  NppStatus nppStatus = nppiDistanceTransformPBA_8u32f_C1R_Ctx(
      resources_.cuda_input, npp_size.width * sizeof(Npp8u), 0, 0, 0, 0, 0, 0,
      0, 0, resources_.cuda_ref_offset_dist, npp_size.width * sizeof(Npp32f),
      npp_size, resources_.device_buffer, nppStreamCtx);
  if (nppStatus != NPP_SUCCESS) {
    cudaError = cudaGetLastError();
    AERROR << "NPP error: nppiDistanceTransformPBA ref_path failed: "
           << cudaGetErrorString(cudaError);
    return false;
  }

  int block_dim = 256;
  int grid_dim = (resources_.block_size + block_dim - 1) / block_dim;
  scaleDistanceFieldKernel<<<grid_dim, block_dim>>>(
      resources_.cuda_ref_offset_dist, resources_.block_size,
      static_cast<float>(resolution));

  updateRefOffsetCostKernel<<<grid_dim, block_dim>>>(
      resources_.cuda_ref_offset_dist, resources_.cuda_ref_offset_cost,
      resources_.block_size, static_cast<float>(max_reference_offset));

  return true;
}

void CostMapProductionCuda::DownloadRefOffsetDistField(
    float *host_ref_offset_dist, int size) {
  cudaMemcpy(host_ref_offset_dist, resources_.cuda_ref_offset_dist,
             size * sizeof(float), cudaMemcpyDeviceToHost);
}

void CostMapProductionCuda::DownloadRefOffsetCostField(
    float *host_ref_offset_cost, int size) {
  cudaMemcpy(host_ref_offset_cost, resources_.cuda_ref_offset_cost,
             size * sizeof(float), cudaMemcpyDeviceToHost);
}

bool CostMapProductionCuda::ProcCompositeObstacle(
    const unsigned char *host_obstacle, const float *host_obstacle_distance,
    const float *host_ref_offset, unsigned char *host_composite_obstacle,
    int rows, int cols, double inflation_safe_margin,
    double max_reference_offset, bool has_ref_path) {
  cudaGetLastError();

  cudaError_t cudaError =
      cudaMemcpy(resources_.cuda_input, host_obstacle,
                 resources_.block_size * sizeof(Npp8u), cudaMemcpyHostToDevice);
  if (cudaError != cudaSuccess) {
    AERROR << "CUDA error: cudaMemcpy obstacle failed: "
           << cudaGetErrorString(cudaError);
    return false;
  }
  cudaError = cudaMemcpy(resources_.cuda_output, host_obstacle_distance,
                         resources_.block_size * sizeof(Npp32f),
                         cudaMemcpyHostToDevice);
  if (cudaError != cudaSuccess) {
    AERROR << "CUDA error: cudaMemcpy obstacle_distance failed: "
           << cudaGetErrorString(cudaError);
    return false;
  }
  if (has_ref_path) {
    cudaError = cudaMemcpy(resources_.cuda_ref_offset_dist, host_ref_offset,
                           resources_.block_size * sizeof(Npp32f),
                           cudaMemcpyHostToDevice);
    if (cudaError != cudaSuccess) {
      AERROR << "CUDA error: cudaMemcpy ref_offset failed: "
             << cudaGetErrorString(cudaError);
      return false;
    }
  }

  int block_dim = 256;
  int grid_dim = (resources_.block_size + block_dim - 1) / block_dim;
  BuildCompositeObstacleKernel<<<grid_dim, block_dim>>>(
      resources_.cuda_input, resources_.cuda_output,
      has_ref_path ? resources_.cuda_ref_offset_dist : nullptr,
      resources_.cuda_composite_obstacle, resources_.block_size,
      static_cast<float>(inflation_safe_margin),
      static_cast<float>(max_reference_offset), has_ref_path);
  cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    AERROR << "CUDA error: BuildCompositeObstacleKernel failed: "
           << cudaGetErrorString(cudaError);
    return false;
  }

  if (host_composite_obstacle != nullptr) {
    cudaError = cudaMemcpy(
        host_composite_obstacle, resources_.cuda_composite_obstacle,
        resources_.block_size * sizeof(Npp8u), cudaMemcpyDeviceToHost);
    if (cudaError != cudaSuccess) {
      AERROR << "CUDA error: cudaMemcpy composite_obstacle failed: "
             << cudaGetErrorString(cudaError);
      return false;
    }
  }

  return true;
}

bool CostMapProductionCuda::ProcComposite(
    unsigned char *host_composite_obstacle, int rows, int cols) {
  cudaGetLastError();

  NppiSize npp_size = {resources_.npp_size_width, resources_.npp_size_height};
  NppStreamContext &nppStreamCtx =
      *static_cast<NppStreamContext *>(resources_.npp_stream_ctx);
  cudaError_t cudaError = cudaSuccess;

  if (host_composite_obstacle != nullptr) {
    cudaError = cudaMemcpy(resources_.cuda_input, host_composite_obstacle,
                           resources_.block_size * sizeof(Npp8u),
                           cudaMemcpyHostToDevice);
    if (cudaError != cudaSuccess) {
      AERROR << "CUDA error: cudaMemcpy composite_obstacle failed: "
             << cudaGetErrorString(cudaError);
      return false;
    }
  }

  Npp8u *composite_input = host_composite_obstacle != nullptr
                               ? resources_.cuda_input
                               : resources_.cuda_composite_obstacle;

  NppStatus nppStatus = nppiDistanceTransformPBA_8u32f_C1R_Ctx(
      composite_input, npp_size.width * sizeof(Npp8u), 0, 0, 0, 0, 0, 0, 0, 0,
      resources_.cuda_composite_cost, npp_size.width * sizeof(Npp32f), npp_size,
      resources_.device_buffer, nppStreamCtx);
  if (nppStatus != NPP_SUCCESS) {
    cudaError = cudaGetLastError();
    AERROR << "NPP error: nppiDistanceTransformPBA composite failed: "
           << cudaGetErrorString(cudaError);
    return false;
  }

  int block_dim = 256;
  int grid_dim = (resources_.block_size + block_dim - 1) / block_dim;
  scaleDistanceFieldKernel<<<grid_dim, block_dim>>>(
      resources_.cuda_composite_cost, resources_.block_size,
      resources_.query_context.resolution);

  std::vector<float> host_temp(resources_.block_size);
  cudaMemcpy(host_temp.data(), resources_.cuda_composite_cost,
             resources_.block_size * sizeof(float), cudaMemcpyDeviceToHost);
  float max_dist = *std::max_element(host_temp.begin(), host_temp.end());

  if (max_dist <= 0.0f) {
    cudaMemset(resources_.cuda_composite_cost, 0,
               resources_.block_size * sizeof(float));
    cudaMemset(resources_.cuda_grad_x, 0,
               resources_.block_size * sizeof(float));
    cudaMemset(resources_.cuda_grad_y, 0,
               resources_.block_size * sizeof(float));
    return true;
  }

  normalizeCompositeCostKernel<<<grid_dim, block_dim>>>(
      resources_.cuda_composite_cost, resources_.block_size, max_dist);

  NppiPoint src_offset = {0, 0};
  nppStatus = nppiFilterSobelVertBorder_32f_C1R_Ctx(
      resources_.cuda_composite_cost, npp_size.width * sizeof(Npp32f), npp_size,
      src_offset, resources_.cuda_grad_x, npp_size.width * sizeof(Npp32f),
      npp_size, NPP_BORDER_REPLICATE, nppStreamCtx);
  if (nppStatus != NPP_SUCCESS) {
    cudaError = cudaGetLastError();
    AERROR << "NPP error: nppiFilterSobelVertBorder_32f_C1R_Ctx failed: "
           << cudaGetErrorString(cudaError);
    return false;
  }

  nppStatus = nppiFilterSobelHorizBorder_32f_C1R_Ctx(
      resources_.cuda_composite_cost, npp_size.width * sizeof(Npp32f), npp_size,
      src_offset, resources_.cuda_grad_y, npp_size.width * sizeof(Npp32f),
      npp_size, NPP_BORDER_REPLICATE, nppStreamCtx);
  if (nppStatus != NPP_SUCCESS) {
    cudaError = cudaGetLastError();
    AERROR << "NPP error: nppiFilterSobelHorizBorder_32f_C1R_Ctx failed: "
           << cudaGetErrorString(cudaError);
    return false;
  }

  normalizeGradientKernel<<<grid_dim, block_dim>>>(
      resources_.cuda_grad_x, resources_.cuda_grad_y, resources_.block_size);

  return true;
}

void CostMapProductionCuda::DownloadCompositeCostField(
    float *host_composite_cost, int size) {
  cudaMemcpy(host_composite_cost, resources_.cuda_composite_cost,
             size * sizeof(float), cudaMemcpyDeviceToHost);
}

} // namespace preprocess
} // namespace planning