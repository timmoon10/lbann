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

#define LBANN_CONCATENATE_LAYER_INSTANTIATE
#include "lbann/layers/transform/concatenate.hpp"
#include "lbann/utils/cuda.hpp"

namespace lbann {

namespace {

using dim4 = cuda::array<size_t, 4>;

/**
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimensions: (max_input_size / bsize) x num_inputs x 1
 */
template <typename T>
__global__ void concat4d_kernel(
  size_t num_inputs,
  const T* __restrict__ * __restrict__ input_buffer_list,
  const dim4* __restrict__ input_dims_list,
  const dim4* __restrict__ input_strides_list,
  T* __restrict__ output_buffer,
  dim4 output_strides,
  const size_t* __restrict__ output_offset_list) {

  // Indices
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t nthreadsx = gridDim.x * blockDim.x;
  const size_t nthreadsy = gridDim.y * blockDim.y;

  for (size_t j=gidy; j<num_inputs; j+=nthreadsy) {

    // Objects for current input tensor
    const auto& input_buffer = input_buffer_list[j];
    const auto& input_dims = input_dims_list[j];
    const auto& input_strides = input_strides_list[j];
    const auto& output_offset = output_offset_list[j];
    const auto& input_size = (input_dims[0] * input_dims[1]
                              * input_dims[2] * input_dims[3]);

    for (size_t i=gidx; i<input_size; i+=nthreadsx) {

      // Get position in input tensor
      dim4 pos;
      size_t pos_flat = i;
      #pragma unroll
      for (int d=3; d>=0; --d) {
        pos[d] = pos_flat % input_dims[d];
        pos_flat = pos_flat / input_dims[d];
      }

      // Copy value to output tensor
      const auto& x = input_buffer[pos[0] * input_strides[0]
                                   + pos[1] * input_strides[1]
                                   + pos[2] * input_strides[2]
                                   + pos[3] * input_strides[3]];
      auto& y = output_buffer[output_offset
                              + pos[0] * output_strides[0]
                              + pos[1] * output_strides[1]
                              + pos[2] * output_strides[2]
                              + pos[3] * output_strides[3]];
      y = x;

    }

  }

}

/**
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimensions: (max_input_size / bsize) x num_inputs x 1
 */
template <typename T>
__global__ void slice4d_kernel(
  size_t num_outputs,
  const T* __restrict__ input_buffer,
  dim4 input_strides,
  const size_t* __restrict__ input_offset_list,
  T* __restrict__ * __restrict__ output_buffer_list,
  const dim4* __restrict__ output_dims_list,
  const dim4* __restrict__ output_strides_list) {

  // Indices
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t nthreadsx = gridDim.x * blockDim.x;
  const size_t nthreadsy = gridDim.y * blockDim.y;

  for (size_t j=gidy; j<num_outputs; j+=nthreadsy) {

    // Objects for current output tensor
    const auto& input_offset = input_offset_list[j];
    auto& output_buffer = output_buffer_list[j];
    const auto& output_dims = output_dims_list[j];
    const auto& output_strides = output_strides_list[j];
    const auto& output_size = (output_dims[0] * output_dims[1]
                              * output_dims[2] * output_dims[3]);

    for (size_t i=gidx; i<output_size; i+=nthreadsx) {

      // Get position in output tensor
      dim4 pos;
      size_t pos_flat = i;
      #pragma unroll
      for (int d=3; d>=0; --d) {
        pos[d] = pos_flat % output_dims[d];
        pos_flat = pos_flat / output_dims[d];
      }

      // Copy value to input tensor
      const auto& x = input_buffer[input_offset
                                   + pos[0] * input_strides[0]
                                   + pos[1] * input_strides[1]
                                   + pos[2] * input_strides[2]
                                   + pos[3] * input_strides[3]];
      auto& y = output_buffer[pos[0] * output_strides[0]
                              + pos[1] * output_strides[1]
                              + pos[2] * output_strides[2]
                              + pos[3] * output_strides[3]];
      y = x;

    }

  }

}

} // namespace <anon>

template <typename TensorDataType>
void fp_compute_impl(
  concatenate_layer<TensorDataType,data_layout::MODEL_PARALLEL,El::Device::GPU>& l,
  size_t concat_dim) {

  // Stack Elemental matrices on top of each other
  // Note: Assume each mini-batch sample is flat.
  auto& output = l.get_activations();
  std::unique_ptr<El::AbstractDistMatrix<TensorDataType>> output_v(
    output.Construct(output.Grid(), output.Root()));
  size_t offset = 0;
  for (size_t i=0; i<static_cast<size_t>(l.get_num_parents()); ++i) {
    const auto& input = l.get_prev_activations(i);
    El::View(*output_v, output,
             El::IR(offset, offset+input.Height()), El::ALL);
    El::Copy(input, *output_v);
    offset += input.Height();
  }

}

template <typename TensorDataType>
void bp_compute_impl(
  concatenate_layer<TensorDataType,data_layout::MODEL_PARALLEL,El::Device::GPU>& l,
  size_t concat_dim) {
  // Tensor views have already been setup in
  // bp_setup_gradient_wrt_inputs
}

template <typename TensorDataType>
void fp_compute_impl(
  concatenate_layer<TensorDataType,data_layout::DATA_PARALLEL,El::Device::GPU>& l,
  size_t concat_dim) {

  // Check that number of dimensions is valid
  /// @todo Support tensors with arbitrary number of dimensions
  const size_t num_dims = l.get_output_dims().size();
  if (num_dims > 3) {
    LBANN_ERROR(l.get_type()," layer \"",l.get_name(),"\" ",
                "is operating on ",num_dims,"-D tensors, ",
                "but only 3-D tensors are currently supported");
  }

  // Get dimensions and strides for each input tensor
  const size_t num_inputs = l.get_num_parents();
  std::vector<const TensorDataType*> input_buffer_list;
  std::vector<dim4> input_dims_list, input_strides_list;
  size_t max_input_size = 0;
  for (size_t j=0; j<num_inputs; ++j) {
    const auto& input = l.get_prev_activations(j);
    const auto& input_dims = l.get_input_dims(j);

    // Construct dimensions and strides in reverse order
    // Note: Assume each mini-batch sample is fully packed.
    std::vector<size_t> rdims(input_dims.rbegin(), input_dims.rend());
    std::vector<size_t> rstrides(input_dims.size(), 1);
    for (size_t d=1; d<input_dims.size(); ++d) {
      rstrides[d] = rdims[d-1] * rstrides[d-1];
    }
    rdims.push_back(input.LocalWidth());
    rstrides.push_back(input.LDim());

    // Pad tensor dimensions to 4D
    rdims.resize(4, 1);
    rstrides.resize(4, rstrides.back());

    input_buffer_list.push_back(input.LockedBuffer());
    input_dims_list.push_back({rdims[3], rdims[2], rdims[1], rdims[0]});
    input_strides_list.push_back(
      {rstrides[3], rstrides[2], rstrides[1], rstrides[0]});
    max_input_size = std::max(max_input_size,
                              rdims[3]*rdims[2]*rdims[1]*rdims[0]);
  }

  // Get strides for output tensor
  dim4 output_strides;
  auto& output = l.get_activations();
  {
    const auto& output_dims = l.get_output_dims();

    // Construct dimensions and strides in reverse order
    // Note: Assume each mini-batch sample is fully packed.
    std::vector<size_t> rdims(output_dims.rbegin(), output_dims.rend());
    std::vector<size_t> rstrides(output_dims.size(), 1);
    for (size_t d=1; d<output_dims.size(); ++d) {
      rstrides[d] = rdims[d-1] * rstrides[d-1];
    }
    rdims.push_back(output.LocalWidth());
    rstrides.push_back(output.LDim());

    // Pad tensor dimensions to 4D
    rdims.resize(4, 1);
    rstrides.resize(4, rstrides.back());

    output_strides = {rstrides[3], rstrides[2], rstrides[1], rstrides[0]};
  }

  // Compute each input tensor's offset in output tensor
  concat_dim += 4 - num_dims;   // Tensor has been padded to 4-D
  std::vector<size_t> output_offset_list;
  output_offset_list.push_back(0);
  for (const auto& input_dims : input_dims_list) {
    auto offset = output_offset_list.back();
    offset += input_dims[concat_dim] * output_strides[concat_dim];
    output_offset_list.push_back(offset);
  }

  // Pack tensor data into a CPU buffer
  l.m_workspace_event.synchronize();
  l.m_workspace.resize(
    sizeof(TensorDataType*) * input_buffer_list.size()
    + sizeof(dim4) * input_dims_list.size()
    + sizeof(dim4) * input_strides_list.size()
    + sizeof(size_t) * output_offset_list.size());
  size_t pos = 0;
  std::memcpy(&l.m_workspace[pos], input_buffer_list.data(),
              sizeof(TensorDataType*) * input_buffer_list.size());
  pos += sizeof(TensorDataType*) * input_buffer_list.size();
  std::memcpy(&l.m_workspace[pos], input_dims_list.data(),
              sizeof(dim4) * input_dims_list.size());
  pos += sizeof(dim4) * input_dims_list.size();
  std::memcpy(&l.m_workspace[pos], input_strides_list.data(),
              sizeof(dim4) * input_strides_list.size());
  pos += sizeof(dim4) * input_strides_list.size();
  std::memcpy(&l.m_workspace[pos], output_offset_list.data(),
              sizeof(size_t) * output_offset_list.size());
  pos += sizeof(size_t) * output_offset_list.size();

  // Copy tensor data to GPU
  auto&& stream = El::GPUManager::Stream();
  cuda::thrust::vector<unsigned char> device_workspace(l.m_workspace.size());
  unsigned char* device_workspace_ptr = device_workspace.data().get();
  cudaMemcpyAsync(device_workspace_ptr,
                  l.m_workspace.data(),
                  l.m_workspace.size(),
                  cudaMemcpyHostToDevice,
                  stream);
  l.m_workspace_event.record(stream);
  pos = 0;
  auto&& device_input_buffer_list
    = reinterpret_cast<const TensorDataType**>(device_workspace_ptr+pos);
  pos += sizeof(TensorDataType*) * input_buffer_list.size();
  auto&& device_input_dims_list
    = reinterpret_cast<const dim4*>(device_workspace_ptr+pos);
  pos += sizeof(dim4) * input_dims_list.size();
  auto&& device_input_strides_list
    = reinterpret_cast<const dim4*>(device_workspace_ptr+pos);
  pos += sizeof(dim4) * input_strides_list.size();
  auto&& device_output_offset_list
    = reinterpret_cast<const size_t*>(device_workspace_ptr+pos);
  pos += sizeof(size_t) * output_offset_list.size();

  // Launch CUDA kernel
  if (max_input_size > 0) {
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (max_input_size + block_size - 1) / block_size;
    grid_dims.y = num_inputs;
    concat4d_kernel<<<grid_dims, block_dims, 0, stream>>>(
      num_inputs,
      device_input_buffer_list,
      device_input_dims_list,
      device_input_strides_list,
      output.Buffer(),
      output_strides,
      device_output_offset_list);
  }

}

template <typename TensorDataType>
void bp_compute_impl(
  concatenate_layer<TensorDataType,data_layout::DATA_PARALLEL,El::Device::GPU>& l,
  size_t concat_dim) {

  // Check that number of dimensions is valid
  /// @todo Support tensors with arbitrary number of dimensions
  const size_t num_dims = l.get_output_dims().size();
  if (num_dims > 3) {
    LBANN_ERROR(l.get_type()," layer \"",l.get_name(),"\" ",
                "is operating on ",num_dims,"-D tensors, ",
                "but only 3-D tensors are currently supported");
  }

  // Get dimensions and strides for each input gradient tensor
  const size_t num_inputs = l.get_num_parents();
  std::vector<TensorDataType*> input_grad_buffer_list;
  std::vector<dim4> input_grad_dims_list, input_grad_strides_list;
  size_t max_input_grad_size = 0;
  for (size_t j=0; j<num_inputs; ++j) {
    auto& input_grad = l.get_error_signals(j);
    const auto& input_grad_dims = l.get_input_dims(j);

    // Construct dimensions and strides in reverse order
    // Note: Assume each mini-batch sample is fully packed.
    std::vector<size_t> rdims(input_grad_dims.rbegin(), input_grad_dims.rend());
    std::vector<size_t> rstrides(input_grad_dims.size(), 1);
    for (size_t d=1; d<input_grad_dims.size(); ++d) {
      rstrides[d] = rdims[d-1] * rstrides[d-1];
    }
    rdims.push_back(input_grad.LocalWidth());
    rstrides.push_back(input_grad.LDim());

    // Pad tensor dimensions to 4D
    rdims.resize(4, 1);
    rstrides.resize(4, rstrides.back());

    input_grad_buffer_list.push_back(input_grad.Buffer());
    input_grad_dims_list.push_back({rdims[3], rdims[2], rdims[1], rdims[0]});
    input_grad_strides_list.push_back(
      {rstrides[3], rstrides[2], rstrides[1], rstrides[0]});
    max_input_grad_size = std::max(max_input_grad_size,
                                   rdims[3]*rdims[2]*rdims[1]*rdims[0]);
  }

  // Get strides for output gradient tensor
  dim4 output_grad_strides;
  const auto& output_grad = l.get_prev_error_signals();
  {
    const auto& output_grad_dims = l.get_output_dims();

    // Construct dimensions and strides in reverse order
    // Note: Assume each mini-batch sample is fully packed.
    std::vector<size_t> rdims(output_grad_dims.rbegin(), output_grad_dims.rend());
    std::vector<size_t> rstrides(output_grad_dims.size(), 1);
    for (size_t d=1; d<output_grad_dims.size(); ++d) {
      rstrides[d] = rdims[d-1] * rstrides[d-1];
    }
    rdims.push_back(output_grad.LocalWidth());
    rstrides.push_back(output_grad.LDim());

    // Pad tensor dimensions to 4D
    rdims.resize(4, 1);
    rstrides.resize(4, rstrides.back());

    output_grad_strides = {rstrides[3], rstrides[2], rstrides[1], rstrides[0]};
  }

  // Compute each input gradient tensor's offset in output gradient tensor
  concat_dim += 4 - num_dims;   // Tensor has been padded to 4-D
  std::vector<size_t> output_grad_offset_list;
  output_grad_offset_list.push_back(0);
  for (const auto& input_grad_dims : input_grad_dims_list) {
    auto offset = output_grad_offset_list.back();
    offset += input_grad_dims[concat_dim] * output_grad_strides[concat_dim];
    output_grad_offset_list.push_back(offset);
  }

  // Pack tensor data into a CPU buffer
  l.m_workspace_event.synchronize();
  l.m_workspace.resize(
    sizeof(size_t) * output_grad_offset_list.size()
    + sizeof(TensorDataType*) * input_grad_buffer_list.size()
    + sizeof(dim4) * input_grad_dims_list.size()
    + sizeof(dim4) * input_grad_strides_list.size());
  size_t pos = 0;
  std::memcpy(&l.m_workspace[pos], output_grad_offset_list.data(),
              sizeof(size_t) * output_grad_offset_list.size());
  pos += sizeof(size_t) * output_grad_offset_list.size();
  std::memcpy(&l.m_workspace[pos], input_grad_buffer_list.data(),
              sizeof(TensorDataType*) * input_grad_buffer_list.size());
  pos += sizeof(TensorDataType*) * input_grad_buffer_list.size();
  std::memcpy(&l.m_workspace[pos], input_grad_dims_list.data(),
              sizeof(dim4) * input_grad_dims_list.size());
  pos += sizeof(dim4) * input_grad_dims_list.size();
  std::memcpy(&l.m_workspace[pos], input_grad_strides_list.data(),
              sizeof(dim4) * input_grad_strides_list.size());
  pos += sizeof(dim4) * input_grad_strides_list.size();

  // Copy tensor data to GPU
  auto&& stream = El::GPUManager::Stream();
  cuda::thrust::vector<unsigned char> device_workspace(l.m_workspace.size());
  unsigned char* device_workspace_ptr = device_workspace.data().get();
  cudaMemcpyAsync(device_workspace_ptr,
                  l.m_workspace.data(),
                  l.m_workspace.size(),
                  cudaMemcpyHostToDevice,
                  stream);
  l.m_workspace_event.record(stream);
  pos = 0;
  auto&& device_output_grad_offset_list
    = reinterpret_cast<const size_t*>(device_workspace_ptr+pos);
  pos += sizeof(size_t) * output_grad_offset_list.size();
  auto&& device_input_grad_buffer_list
    = reinterpret_cast<TensorDataType**>(device_workspace_ptr+pos);
  pos += sizeof(TensorDataType*) * input_grad_buffer_list.size();
  auto&& device_input_grad_dims_list
    = reinterpret_cast<const dim4*>(device_workspace_ptr+pos);
  pos += sizeof(dim4) * input_grad_dims_list.size();
  auto&& device_input_grad_strides_list
    = reinterpret_cast<const dim4*>(device_workspace_ptr+pos);
  pos += sizeof(dim4) * input_grad_strides_list.size();

  // Launch CUDA kernel
  if (max_input_grad_size > 0) {
    constexpr size_t block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (max_input_grad_size + block_size - 1) / block_size;
    grid_dims.y = num_inputs;
    slice4d_kernel<<<grid_dims, block_dims, 0, stream>>>(
      num_inputs,
      output_grad.LockedBuffer(),
      output_grad_strides,
      device_output_grad_offset_list,
      device_input_grad_buffer_list,
      device_input_grad_dims_list,
      device_input_grad_strides_list);
  }

}

// Explicit instantiation
#define PROTO(T)                                                        \
  template class concatenate_layer<                                     \
    T, data_layout::DATA_PARALLEL, El::Device::GPU>;                    \
  template class concatenate_layer<                                     \
    T, data_layout::MODEL_PARALLEL, El::Device::GPU>

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
