////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

#include "lbann/layers/misc/dist_embedding.hpp"
#include "lbann/utils/cuda.hpp"

#include <layers.pb.h>

#ifdef LBANN_HAS_NVSHMEM
#include "nvshmem.h"
#include "nvshmemx.h"
#endif // LBANN_HAS_NVSHMEM

namespace lbann {

namespace {

using RequestType = dist_embedding_layer_impl::vector_request;
using Size2 = cuda::array<size_t, 2>;

/// @todo This would be fun to optimize further.
template <typename T> __device__ __forceinline__
T* memcpy_warp(T* __restrict__ dest, const T* __restrict__ src, size_t n) {
  constexpr size_t warp_size = 32;
  const size_t tid = threadIdx.x;
  for (size_t i = tid; i < n; i += warp_size) {
    dest[i] = src[i];
  }
  return dest;
}

/** See El::AbstractDistMatrix::ColOwner. */
__device__ __forceinline__
size_t distmat_index_owner(size_t global_index, size_t align, size_t stride) {
  return (global_index + align) % stride;
}

/** See El::AbstractDistMatrix::GlobalCol. */
__device__ __forceinline__
size_t distmat_global_index(size_t local_index, size_t shift, size_t stride) {
  return shift + local_index * stride;
}

/** See El::AbstractDistMatrix::LocalCol. */
__device__ __forceinline__
size_t distmat_local_index(size_t global_index, size_t rank, size_t align, size_t stride) {
  size_t shift = (long(rank) - align) % stride;
  if (shift < 0) {
    shift += stride;
  }
  if (global_index > shift) {
    return (global_index - shift - 1) / stride + 1;
  }
  else {
    return 0;
  }
}

template <typename Kernel, typename... ArgTs>
inline void launch_cuda_kernel(
  const Kernel& kernel,
  dim3 grid_dims,
  dim3 block_dims,
  size_t shared_mem,
  cudaStream_t stream,
  ArgTs... args) {
  void *arg_list[] = {
    const_cast<void*>(reinterpret_cast<const void*>(&args))...
  };
  CHECK_CUDA(
    cudaLaunchKernel(
      reinterpret_cast<const void*>(&kernel),
      grid_dims,
      block_dims,
      arg_list,
      shared_mem,
      stream));
}

template <typename Kernel, typename... ArgTs>
inline void launch_nvshmem_collective_kernel(
  const Kernel& kernel,
  dim3 grid_dims,
  dim3 block_dims,
  size_t shared_mem,
  cudaStream_t stream,
  ArgTs... args) {
  if (grid_dims.x == 0) {
    grid_dims.y = 0;
    grid_dims.z = 0;
  }
  void *arg_list[] = {
    const_cast<void*>(reinterpret_cast<const void*>(&args))...
  };
  auto status = nvshmemx_collective_launch(
    reinterpret_cast<const void*>(&kernel),
    grid_dims,
    block_dims,
    arg_list,
    shared_mem,
    stream);
  if (status != 0) {
    LBANN_ERROR(
      "Failed to launch NVSHMEM collective kernel ",
      "(error ",status,")");
  }
}

} // namespace <anon>

// =============================================
// Life cycle functions
// =============================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
dist_embedding_layer<TensorDataType,Layout,Device>::~dist_embedding_layer()
{
#ifdef LBANN_HAS_NVSHMEM
  nvshmem_free(m_workspace_buffer);
  nvshmem_free(m_requests_buffer);
#endif // LBANN_HAS_NVSHMEM
}

// =============================================
// Forward prop
// =============================================

namespace {

#ifdef LBANN_HAS_NVSHMEM
/** Request embedding vectors from owner processes.
 *
 *  Block dimensions: 32 x 1 x 1
 *
 *  Grid dimensions: input_dims[1] x input_dims[0] x 1
 */
template <typename TensorDataType>
__global__ void send_requests_kernel(
  Size2 input_dims,
  const TensorDataType* __restrict__ input,
  Size2 input_strides,
  RequestType* __restrict__ requests,
  Size2 requests_strides,
  size_t rank,
  size_t input_rowshift,
  size_t input_rowstride,
  size_t embeddings_rowalign,
  size_t embeddings_rowstride) {

  // Indices
  const size_t tid = threadIdx.x;
  const size_t bidx = blockIdx.x;
  const size_t bidy = blockIdx.y;
  const size_t nblocksx = gridDim.x;
  const size_t nblocksy = gridDim.y;

  const size_t i_per_block = (input_dims[1] + nblocksx - 1) / nblocksx;
  const size_t i_start = bidx * i_per_block;
  const size_t i_end = cuda::min((bidx+1) * i_per_block, input_dims[1]);
  for (size_t j = bidy; j < input_dims[0]; j += nblocksy) {
    for (size_t i = i_start; i < i_end; ++i) {
      const auto& global_j = distmat_global_index(j, input_rowshift, input_rowstride);

      // Get embedding vector index
      const auto& global_index_float
        = input[i*input_strides[1] + j*input_strides[0]];
      const auto& global_index = static_cast<size_t>(cuda::floor(global_index_float));

      // Figure out which process owns embedding vector
      auto& req = requests[i*requests_strides[1] + global_j*requests_strides[0]];
      if (tid == 0) {
        req.source_rank = distmat_index_owner(global_index, embeddings_rowalign, embeddings_rowstride);
        req.source_index = distmat_local_index(global_index, req.source_rank, embeddings_rowalign, embeddings_rowstride);
        req.target_rank = rank;
        req.target_index = i + global_j*input_dims[1];
        req.is_active = true;
      }

      // Send request to owner process
      __syncwarp();
      nvshmemx_putmem_nbi_warp(
        &req,
        &req,
        sizeof(RequestType),
        req.source_rank);

    }
  }

}
#endif // LBANN_HAS_NVSHMEM

#ifdef LBANN_HAS_NVSHMEM
/** Send my embedding vectors to requesting processes.
 *
 *  Block dimensions: 32 x 1 x 1
 *
 *  Grid dimensions: num_requests x 1 x 1
 */
template <typename TensorDataType>
__global__ void send_embeddings_kernel(
  size_t embedding_dim,
  size_t num_requests,
  RequestType* __restrict__ requests,
  const TensorDataType* __restrict__ embeddings,
  Size2 embeddings_strides,
  TensorDataType* __restrict__ workspace,
  Size2 workspace_strides,
  size_t rank) {

  // Indices
  const size_t tid = threadIdx.x;
  const size_t bid = blockIdx.x;
  const size_t nblocks = gridDim.x;

  // Assign requests to CUDA blocks
  const size_t requests_per_block = (num_requests + nblocks - 1) / nblocks;
  const size_t i_start = bid * requests_per_block;
  const size_t i_end = cuda::min((bid+1) * requests_per_block, num_requests);

  // Send my embedding vectors to requesting processes
  for (size_t i = i_start; i < i_end; ++i) {
    const auto& req = requests[i];
    if (req.is_active && req.source_rank == rank) {
      nvshmemx_putmem_nbi_warp(
        &workspace[req.target_index * workspace_strides[0]],
        &embeddings[req.source_index * embeddings_strides[0]],
        embedding_dim*sizeof(TensorDataType),
        req.target_rank);
    }
  }

  // Notify requesting processes that they have recieved my embedding vectors
  __syncwarp();
  if (tid == 0) {
    nvshmem_fence();
  }
  __syncwarp();
  const long flag_val{1};
  for (size_t i = i_start; i < i_end; ++i) {
    auto& req = requests[i];
    if (req.is_active && req.source_rank == rank) {
      nvshmemx_long_put_warp(
        &req.is_completed,
        &flag_val,
        1,
        req.target_rank);
    }
  }

}
#endif // LBANN_HAS_NVSHMEM

#ifdef LBANN_HAS_NVSHMEM
/** Wait for embedding vectors from owner processes.
 *
 *  Block dimensions: 32 x 1 x 1
 *
 *  Grid dimensions: Max allowed by NVSHMEM
 */
template <typename TensorDataType>
__global__ void wait_for_embeddings_kernel(
  size_t embedding_dim,
  Size2 input_dims,
  RequestType* __restrict__ requests,
  Size2 requests_strides,
  const TensorDataType* __restrict__ workspace,
  Size2 workspace_strides,
  TensorDataType* __restrict__ output,
  Size2 output_strides,
  size_t input_rowshift,
  size_t input_rowstride) {

  // Indices
  const size_t tid = threadIdx.x;
  const size_t bidx = blockIdx.x;
  const size_t bidy = blockIdx.y;
  const size_t nblocksx = gridDim.x;
  const size_t nblocksy = gridDim.y;

  // Assign requests to CUDA blocks
  const size_t i_per_block = (input_dims[1] + nblocksx - 1) / nblocksx;
  const size_t i_start = bidx * i_per_block;
  const size_t i_end = cuda::min((bidx+1) * i_per_block, input_dims[1]);

  for (size_t j = bidy; j < input_dims[0]; j += nblocksy) {
    for (size_t i = i_start; i < i_end; ++i) {
      const auto& global_j = distmat_global_index(j, input_rowshift, input_rowstride);

      // Wait for embedding vector to arrive
      auto& req = requests[i*requests_strides[1] + global_j*requests_strides[0]];
      if (tid == 0) {
        nvshmem_wait(&req.is_completed, 0);
        req.is_completed = 0;
      }
      __syncwarp();

      // Copy embedding vector to output tensor
      memcpy_warp(
        &output[i*embedding_dim + j*output_strides[0]],
        &workspace[req.target_index*workspace_strides[0]],
        embedding_dim);

    }
  }

}
#endif // LBANN_HAS_NVSHMEM

} // namespace <anon>

template <typename TensorDataType, data_layout Layout, El::Device Device>
void dist_embedding_layer<TensorDataType,Layout,Device>::fp_compute() {
#ifndef LBANN_HAS_NVSHMEM
  LBANN_ERROR(
    "dist_embedding_layer with ",
    "(TensorDataType=",TypeName<TensorDataType>(),", ",
    "Layout=",to_string(Layout),", ",
    "Device=",to_string(Device),") ",
    "requires NVSHMEM, but LBANN has not been built with NVSHMEM");
  return;
#else // LBANN_HAS_NVSHMEM

  // Data matrices
  using LocalMat = El::Matrix<TensorDataType, Device>;
  const auto& embeddings = this->get_data_type_weights(0).get_values();
  const auto& input = this->get_prev_activations();
  const auto& local_input = dynamic_cast<const LocalMat&>(input.LockedMatrix());
  auto& local_output = dynamic_cast<LocalMat&>(this->get_local_activations());

  // Dimensions
  const size_t input_size = this->get_input_size();
  const size_t output_size = this->get_output_size();
  const size_t mini_batch_size = input.Width();
  const size_t local_mini_batch_size = local_input.Width();

  // GPU objects
  auto&& stream = El::GPUManager::Stream();

  // SHMEM processing element
  const size_t rank = this->get_comm()->get_rank_in_trainer();

  // Initialize NVSHMEM buffer for embedding vectors
  if (m_workspace_buffer_size < output_size * mini_batch_size) {
    nvshmem_free(m_workspace_buffer);
    m_workspace_buffer_size = output_size * mini_batch_size;
    m_workspace_buffer = reinterpret_cast<TensorDataType*>(
      nvshmem_malloc(m_workspace_buffer_size*sizeof(RequestType)));
  }
  LocalMat workspace(
    m_embedding_dim,
    input_size * mini_batch_size,
    m_workspace_buffer,
    m_embedding_dim);

  // Initialize NVSHMEM buffer for shmem_put requests
  if (m_requests_buffer_size < input_size * mini_batch_size) {
    nvshmem_free(m_requests_buffer);
    m_requests_buffer_size = input_size * mini_batch_size;
    m_requests_buffer = reinterpret_cast<RequestType*>(
      nvshmem_malloc(m_requests_buffer_size*sizeof(RequestType)));
  }
  CHECK_CUDA(
    cudaMemsetAsync(
      m_requests_buffer,
      0,
      m_requests_buffer_size*sizeof(RequestType),
      stream));

  // Request embedding vectors from owner processes
  nvshmemx_barrier_all_on_stream(stream);
  if (!local_input.IsEmpty()) {
    constexpr size_t block_size = 32;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = input_size;
    grid_dims.y = local_mini_batch_size;
    launch_cuda_kernel(
      send_requests_kernel<TensorDataType>,
      grid_dims,
      block_dims,
      0,
      stream,
      Size2{local_mini_batch_size, input_size},
      size_t(local_input.LockedBuffer()),
      Size2{size_t(local_input.LDim()), 1},
      m_requests_buffer,
      Size2{input_size, 1},
      size_t(rank),
      size_t(input.RowShift()),
      size_t(input.RowStride()),
      size_t(embeddings.RowAlign()),
      size_t(embeddings.RowStride()));
  }
  nvshmemx_barrier_all_on_stream(stream);

  // Send my embedding vectors to requesting processes
  {
    constexpr size_t block_size = 32;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = input_size * mini_batch_size;
    launch_cuda_kernel(
      send_embeddings_kernel<TensorDataType>,
      grid_dims,
      block_dims,
      0,
      stream,
      m_embedding_dim,
      input_size * mini_batch_size,
      m_requests_buffer,
      embeddings.LockedBuffer(),
      Size2{size_t(embeddings.LDim()), 1},
      workspace.Buffer(),
      Size2{size_t(workspace.LDim()), 1},
      rank);
  }

  // Copy embedding vectors from workspace to output tensor
  {
    constexpr size_t block_size = 32;
    launch_nvshmem_collective_kernel(
      wait_for_embeddings_kernel<TensorDataType>,
      0,
      block_size,
      0,
      stream,
      m_embedding_dim,
      Size2{local_mini_batch_size, input_size},
      m_requests_buffer,
      Size2{input_size, 1},
      workspace.LockedBuffer(),
      Size2{size_t(workspace.LDim()), 1},
      local_output.Buffer(),
      Size2{size_t(local_output.LDim()), 1},
      size_t(input.RowShift()),
      size_t(input.RowStride()));
  }

#endif // LBANN_HAS_NVSHMEM
}

// =============================================
// Backprop
// =============================================

namespace {

#ifdef LBANN_HAS_NVSHMEM
/** Send gradients to owner processes.
 *
 *  Block dimensions: 32 x 1 x 1
 *
 *  Grid dimensions: input_dims[1] x input_dims[0] x 1
 */
template <typename TensorDataType>
__global__ void send_gradients_kernel(
  size_t embedding_dim,
  Size2 input_dims,
  RequestType* __restrict__ requests,
  Size2 requests_strides,
  const TensorDataType* __restrict__ output_grad,
  Size2 output_grad_strides,
  TensorDataType* __restrict__ workspace,
  Size2 workspace_strides,
  size_t rank,
  size_t input_rowshift,
  size_t input_rowstride) {

  // Indices
  const size_t tid = threadIdx.x;
  const size_t bidx = blockIdx.x;
  const size_t bidy = blockIdx.y;
  const size_t nblocksx = gridDim.x;
  const size_t nblocksy = gridDim.y;

  // Assign requests to CUDA blocks
  const size_t i_per_block = (input_dims[1] + nblocksx - 1) / nblocksx;
  const size_t i_start = bidx * i_per_block;
  const size_t i_end = cuda::min((bidx+1) * i_per_block, input_dims[1]);

  // Send gradients to owner processes
  for (size_t j = bidy; j < input_dims[0]; j += nblocksy) {
    for (size_t i = i_start; i < i_end; ++i) {
      const auto& global_j = distmat_global_index(j, input_rowshift, input_rowstride);
      auto& req = requests[i*requests_strides[1] + global_j*requests_strides[0]];
      if (req.is_active && req.target_rank == rank) {
        nvshmemx_putmem_nbi_warp(
          &workspace[req.target_index * workspace_strides[0]],
          &output_grad[i*embedding_dim + j*output_grad_strides[0]],
          embedding_dim*sizeof(TensorDataType),
          req.source_rank);
      }
    }
  }

  // Notify owner processes that they have recieved gradients
  if (tid == 0) {
    nvshmem_fence();
  }
  __syncwarp();
  const long flag_val{1};
  for (size_t j = bidy; j < input_dims[0]; j += nblocksy) {
    for (size_t i = i_start; i < i_end; ++i) {
      const auto& global_j = distmat_global_index(j, input_rowshift, input_rowstride);
      auto& req = requests[i*requests_strides[1] + global_j*requests_strides[0]];
      nvshmemx_long_put_warp(
        &req.is_completed,
        &flag_val,
        1,
        req.source_rank);
    }
  }

}
#endif // LBANN_HAS_NVSHMEM

#ifdef LBANN_HAS_NVSHMEM
/** Sparse SGD on local embeddings.
 *
 *  Block dimensions: 32 x 1 x 1
 *
 *  Grid dimensions: Max allowed by NVSHMEM
 */
template <typename TensorDataType>
__global__ void sgd_kernel(
  TensorDataType learning_rate,
  size_t embedding_dim,
  size_t num_requests,
  RequestType* __restrict__ requests,
  const TensorDataType* __restrict__ workspace,
  Size2 workspace_strides,
  TensorDataType* __restrict__ embeddings,
  Size2 embeddings_strides,
  size_t rank) {

  // Indices
  const size_t tid = threadIdx.x;
  const size_t bid = blockIdx.x;
  const size_t nblocks = gridDim.x;
  constexpr size_t warp_size = 32;

  // Assign requests to CUDA blocks
  const size_t requests_per_block = (num_requests + nblocks - 1) / nblocks;
  const size_t i_start = bid * requests_per_block;
  const size_t i_end = cuda::min((bid+1) * requests_per_block, num_requests);

  // Send my embedding vectors to requesting processes
  for (size_t i = i_start; i < i_end; ++i) {
    auto& req = requests[i];
    if (req.is_active && req.source_rank == rank) {

      // Wait until gradient has been recieved
      if (tid == 0) {
        nvshmem_wait(&req.is_completed, 0);
        req.is_completed = 0;
      }
      __syncwarp();

      // Update embedding with gradient
      const auto* __restrict__ dw = &workspace[req.target_index * workspace_strides[0]];
      auto* __restrict__ w = &embeddings[req.source_index * embeddings_strides[0]];
      for (size_t k = tid; k < embedding_dim; k += warp_size) {
        cuda::atomic_add(&w[k], -learning_rate * dw[k]);
      }

    }
  }

}
#endif // LBANN_HAS_NVSHMEM

} // namespace <anon>

template <typename TensorDataType, data_layout Layout, El::Device Device>
void dist_embedding_layer<TensorDataType,Layout,Device>::bp_compute() {
#ifndef LBANN_HAS_NVSHMEM
  LBANN_ERROR(
    "dist_embedding_layer with ",
    "(TensorDataType=",TypeName<TensorDataType>(),", ",
    "Layout=",to_string(Layout),", ",
    "Device=",to_string(Device),") ",
    "requires NVSHMEM, but LBANN has not been built with NVSHMEM");
  return;
#else // LBANN_HAS_NVSHMEM

  // Data matrices
  using LocalMat = El::Matrix<TensorDataType, Device>;
  auto& embeddings = this->get_data_type_weights(0).get_values();
  auto& local_embeddings = dynamic_cast<LocalMat&>(embeddings.Matrix());
  const auto& input = this->get_prev_activations();
  const auto& local_output_grad = dynamic_cast<const LocalMat&>(this->get_local_prev_error_signals());

  // Dimensions
  const size_t input_size = this->get_input_size();
  const size_t mini_batch_size = input.Width();
  const size_t local_mini_batch_size = local_output_grad.Width();

  // GPU objects
  auto&& stream = El::GPUManager::Stream();

  // SHMEM processing element
  const size_t rank = this->get_comm()->get_rank_in_trainer();

  // Initialize NVSHMEM buffer for gradient w.r.t. embeddings
  LocalMat workspace(
    m_embedding_dim,
    input_size * mini_batch_size,
    m_workspace_buffer,
    m_embedding_dim);

  // Send gradients to owner processes
  if (!local_output_grad.IsEmpty()) {
    constexpr size_t block_size = 32;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = input_size;
    grid_dims.y = local_mini_batch_size;
    launch_cuda_kernel(
      send_gradients_kernel<TensorDataType>,
      grid_dims,
      block_dims,
      0,
      stream,
      m_embedding_dim,
      Size2{local_mini_batch_size, input_size},
      m_requests_buffer,
      Size2{input_size, 1},
      local_output_grad.LockedBuffer(),
      Size2{static_cast<size_t>(local_output_grad.LDim()), 1},
      workspace.Buffer(),
      Size2{static_cast<size_t>(workspace.LDim()), 1},
      rank,
      size_t(input.RowShift()),
      size_t(input.RowStride()));
  }

  // Configure local embeddings for sparse SGD
  // Note: If we are not doing sparse SGD, then we initialize
  // embeddings_v as a tensor of zeros. Applying sparse SGD to this
  // tensor results in the full gradient tensor, which can then be
  // sent to a dense optimizer.
  LocalMat local_embeddings_v;
  std::unique_ptr<El::AbstractDistMatrix<TensorDataType>> embeddings_grad;
  if (m_sparse_sgd) {
    El::View(local_embeddings_v, local_embeddings);
  }
  else {
    embeddings_grad.reset(
      embeddings.Construct(embeddings.Grid(), embeddings.Root()));
    embeddings_grad->AlignWith(embeddings);
    El::Zeros(*embeddings_grad, embeddings.Height(), embeddings.Width());
    El::View(local_embeddings_v, embeddings_grad->Matrix());
  }

  // Sparse SGD on local embeddings
  {
    constexpr size_t block_size = 32;
    launch_nvshmem_collective_kernel(
      sgd_kernel<TensorDataType>,
      0,
      block_size,
      0,
      stream,
      m_learning_rate,
      m_embedding_dim,
      input_size * mini_batch_size,
      m_requests_buffer,
      workspace.LockedBuffer(),
      Size2{size_t(workspace.LDim()), 1},
      local_embeddings_v.Buffer(),
      Size2{size_t(local_embeddings_v.LDim()), 1},
      rank);
  }

  // Send gradients to dense optimizer if needed
  auto&& opt = this->get_data_type_weights(0).get_optimizer();
  if (!m_sparse_sgd && opt != nullptr) {
    opt->add_to_gradient(*embeddings_grad);
  }

#endif // LBANN_HAS_NVSHMEM
}

// =============================================
// Explicit template instantiation
// =============================================

/// @todo fp16
template class dist_embedding_layer<
  float, data_layout::DATA_PARALLEL, El::Device::GPU>;

} // namespace lbann
