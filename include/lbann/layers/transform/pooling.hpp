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

#ifndef LBANN_LAYER_POOLING_HPP_INCLUDED
#define LBANN_LAYER_POOLING_HPP_INCLUDED

#include <utility>
#include <vector>
#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/cudnn.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/im2col.hpp"

namespace lbann {

// Forward declaration
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class unpooling_layer;

template <typename TensorDataType,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class pooling_layer : public transform_layer<TensorDataType> {
  static_assert(T_layout == data_layout::DATA_PARALLEL,
                "pooling only supports DATA_PARALLEL");
private:

  /** Pooling mode. */
  pool_mode m_pool_mode;

  /** Pooling window dimensions. */
  std::vector<int> m_pool_dims;
  /** Size of pooling window. */
  int m_pool_size;
  /** Pooling padding. */
  std::vector<int> m_pads;
  /** Pooling strides. */
  std::vector<int> m_strides;

  /** Input indices for max pooling.
   *  Each entry corresponds to a local entry in the activations
   *  matrix. The entry gives the index of the maximum entry within
   *  the pooling window.
   */
  std::vector<int> m_max_pool_indices;

#ifdef LBANN_HAS_CUDNN
  /** Pooling descriptor. */
  cudnnPoolingDescriptor_t m_pooling_cudnn_desc;
  /** Tensor cuDNN descriptors. */
  cudnn::data_parallel_layer_tensor_manager<TensorDataType> m_tensors_cudnn_desc;
#endif // LBANN_HAS_CUDNN

  friend class unpooling_layer<TensorDataType, T_layout, Dev>;

public:

  pooling_layer(lbann_comm *comm,
                int num_data_dims,
                int pool_dim,
                int pad,
                int stride,
                pool_mode mode)
    : pooling_layer(comm,
                    num_data_dims,
                    std::vector<int>(num_data_dims, pool_dim),
                    std::vector<int>(num_data_dims, pad),
                    std::vector<int>(num_data_dims, stride),
                    mode) {}

  pooling_layer(lbann_comm *comm,
                int num_data_dims,
                std::vector<int> pool_dims,
                std::vector<int> pads,
                std::vector<int> strides,
                pool_mode mode)
    : transform_layer<TensorDataType>(comm),
      m_pool_mode(mode),
      m_pool_dims(pool_dims),
      m_pads(pads),
      m_strides(strides)
#ifdef LBANN_HAS_CUDNN
    , m_pooling_cudnn_desc(nullptr),
      m_tensors_cudnn_desc(this)
#endif // LBANN_HAS_CUDNN
  {
    // Initialize input dimensions and pooling parameters
    m_pool_size = std::accumulate(m_pool_dims.begin(),
                                  m_pool_dims.end(),
                                  1,
                                  std::multiplies<int>());

  }

  pooling_layer(const pooling_layer& other)
    : transform_layer<TensorDataType>(other),
      m_pool_mode(other.m_pool_mode),
      m_pool_dims(other.m_pool_dims),
      m_pool_size(other.m_pool_size),
      m_pads(other.m_pads),
      m_strides(other.m_strides),
      m_max_pool_indices(other.m_max_pool_indices)
#ifdef LBANN_HAS_CUDNN
    , m_pooling_cudnn_desc(nullptr),
      m_tensors_cudnn_desc(other.m_tensors_cudnn_desc)
#endif // LBANN_HAS_CUDNN
  {
#ifdef LBANN_HAS_CUDNN
    copy_pooling_cudnn_desc(other.m_pooling_cudnn_desc, m_pooling_cudnn_desc);
    m_tensors_cudnn_desc.set_layer(this);
#endif // LBANN_HAS_CUDNN
  }

  pooling_layer& operator=(const pooling_layer& other){
    transform_layer<TensorDataType>::operator=(other);
    m_pool_mode = other.m_pool_mode;
    m_pool_dims = other.m_pool_dims;
    m_pool_size = other.m_pool_size;
    m_pads = other.m_pads;
    m_strides = other.m_strides;
    m_max_pool_indices = other.m_max_pool_indices;
#ifdef LBANN_HAS_CUDNN
    copy_pooling_cudnn_desc(other.m_pooling_cudnn_desc, m_pooling_cudnn_desc);
    m_tensors_cudnn_desc = other.m_tensors_cudnn_desc;
    m_tensors_cudnn_desc.set_layer(this);
#endif // LBANN_HAS_CUDNN
    return *this;
  }

  ~pooling_layer() {
#ifdef LBANN_HAS_CUDNN
    if (m_pooling_cudnn_desc != nullptr) {
      cudnnDestroyPoolingDescriptor(m_pooling_cudnn_desc);
    }
#endif // LBANN_HAS_CUDNN
  }

  pooling_layer* copy() const override { return new pooling_layer(*this); }
  std::string get_type() const override { return "pooling"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  description get_description() const override {
    auto desc = transform_layer<TensorDataType>::get_description();
    std::stringstream ss;

    // Pool mode
    ss.str(std::string{});
    ss.clear();
    switch (m_pool_mode) {
    case pool_mode::max:            ss << "max";              break;
    case pool_mode::average:        ss << "average";          break;
    case pool_mode::average_no_pad: ss << "average (no pad)"; break;
    case pool_mode::invalid:
    default:
      ss << "invalid";
    }
    desc.add("Pool mode", ss.str());

    // Pool dimensions
    ss.str(std::string{});
    ss.clear();
    for (size_t i = 0; i < m_pool_dims.size(); ++i) {
      ss << (i > 0 ? ", " : "" ) << m_pool_dims[i];
    }
    desc.add("Pool dimensions", ss.str());

    // Strides
    ss.str(std::string{});
    ss.clear();
    for (size_t i = 0; i < m_strides.size(); ++i) {
      ss << (i > 0 ? ", " : "" ) << m_strides[i];
    }
    desc.add("Strides", ss.str());

    // Pads
    ss.str(std::string{});
    ss.clear();
    for (size_t i = 0; i < m_pads.size(); ++i) {
      ss << (i > 0 ? ", " : "" ) << m_pads[i];
    }
    desc.add("Pads", ss.str());

    // Result
    return desc;

  }

protected:

  void setup_dims() override {
    transform_layer<TensorDataType>::setup_dims();
    const auto& input_dims = this->get_input_dims();
    auto output_dims = input_dims;
    for(size_t i = 0; i < output_dims.size() - 1; ++i) {
      const int effective_dim = (input_dims[i+1] + 2 * m_pads[i]
                                 - m_pool_dims[i] + 1);
      output_dims[i+1] = (effective_dim + m_strides[i] - 1) / m_strides[i];
    }
    this->set_output_dims(output_dims);
  }

  /// Initialize GPU objects
  void setup_gpu() override {
    transform_layer<TensorDataType>::setup_gpu();
#ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
#else

    // Set pooling descriptor
    cudnnPoolingMode_t cudnn_pool_mode;
    switch(m_pool_mode) {
    case pool_mode::max:
    #ifndef LBANN_DETERMINISTIC
      cudnn_pool_mode = CUDNN_POOLING_MAX; break;
    #else
      cudnn_pool_mode = CUDNN_POOLING_MAX_DETERMINISTIC; break;
    #endif
    case pool_mode::average:
      cudnn_pool_mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING; break;
    case pool_mode::average_no_pad:
      cudnn_pool_mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING; break;
    default:
      std::stringstream err;
      err << "no GPU implementation for pooling mode " << static_cast<int>(m_pool_mode);
      LBANN_ERROR(err.str());
      cudnn_pool_mode = CUDNN_POOLING_MAX;
    }
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&m_pooling_cudnn_desc));
    CHECK_CUDNN(cudnnSetPoolingNdDescriptor(m_pooling_cudnn_desc,
                                            cudnn_pool_mode,
                                            CUDNN_PROPAGATE_NAN,
                                            m_pool_dims.size(),
                                            m_pool_dims.data(),
                                            m_pads.data(),
                                            m_strides.data()));

#endif // #ifndef LBANN_HAS_CUDNN
  }

  void fp_compute() override {
    if(this->using_gpus()) {
      fp_compute_cudnn();
    } else {
      fp_compute_im2col();
    }
  }

  void bp_compute() override {
    if(this->using_gpus()) {
      bp_compute_cudnn();
    } else {
      bp_compute_im2col();
    }
  }

private:

  /// Pooling forward propagation with cuDNN
  void fp_compute_cudnn() {
#ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
#else
    using ScalingType = cudnn::ScalingParamType<TensorDataType>;
    const auto& local_input = this->get_local_prev_activations();
    auto& local_output = this->get_local_activations();
    if (local_input.Height() > 0 && local_input.Width() > 0) {
      const auto zero = El::TypeTraits<ScalingType>::Zero();
      const auto one = El::TypeTraits<ScalingType>::One();
      CHECK_CUDNN(cudnnPoolingForward(cudnn::get_handle(),
                                      m_pooling_cudnn_desc,
                                      &one,
                                      m_tensors_cudnn_desc.get_prev_activations(),
                                      local_input.LockedBuffer(),
                                      &zero,
                                      m_tensors_cudnn_desc.get_activations(),
                                      local_output.Buffer()));
    }
#endif // #ifndef LBANN_HAS_CUDNN
  }

  /// Pooling backward propagation with cuDNN
  void bp_compute_cudnn() {
#ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
#else
    using ScalingType = cudnn::ScalingParamType<TensorDataType>;
    const auto& local_input = this->get_local_prev_activations();
    const auto& local_output = this->get_local_activations();
    const auto& local_gradient_wrt_output = this->get_local_prev_error_signals();
    auto& local_gradient_wrt_input = this->get_local_error_signals();
    if (local_input.Height() > 0 && local_input.Width() > 0) {

      // Useful constants
      const auto one = El::TypeTraits<ScalingType>::One();
      const auto zero = El::TypeTraits<ScalingType>::Zero();

      // Perform backprop on GPU
      CHECK_CUDNN(cudnnPoolingBackward(cudnn::get_handle(),
                                       m_pooling_cudnn_desc,
                                       &one,
                                       m_tensors_cudnn_desc.get_activations(),
                                       local_output.LockedBuffer(),
                                       m_tensors_cudnn_desc.get_prev_error_signals(),
                                       local_gradient_wrt_output.LockedBuffer(),
                                       m_tensors_cudnn_desc.get_prev_activations(),
                                       local_input.LockedBuffer(),
                                       &zero,
                                       m_tensors_cudnn_desc.get_error_signals(),
                                       local_gradient_wrt_input.Buffer()));

    }
#endif // #ifndef LBANN_HAS_CUDNN
  }

  /// Pooling forward propagation with im2col
  void fp_compute_im2col() {
    if(m_pool_mode != pool_mode::max && m_pool_mode != pool_mode::average) {
      LBANN_ERROR("CPU pooling layer only supports max and average pooling");
    }

    // Local matrices
    const auto& local_input = this->get_local_prev_activations();
    auto& local_output = this->get_local_activations();

    // Pool parameters
    const int local_width = local_input.Width();
    const auto& input_dims = this->get_input_dims();
    const int num_channels = input_dims[0];
    const int num_per_output_channel = this->get_output_size() / num_channels;

    // Initialize max pool indices if needed
    if(m_pool_mode == pool_mode::max) {
      m_max_pool_indices.assign(this->get_output_size() * local_width, 0);
    }

    // Initialize matrices
    El::Matrix<TensorDataType, Dev> im2col_mat(m_pool_size * num_channels, num_per_output_channel);
    El::Matrix<TensorDataType, Dev> input_mat;

    // Iterate through data samples
    for(int sample = 0; sample < local_width; ++sample) {

      // Construct im2col matrix from input
      El::LockedView(input_mat, local_input,
                     El::ALL, El::IR(sample));
      im2col<TensorDataType>(input_mat,
             im2col_mat,
             num_channels,
             input_dims.size() - 1,
             &input_dims[1],
             m_pads.data(),
             m_pool_dims.data(),
             m_strides.data());

      if(m_pool_mode == pool_mode::max) {
        // Apply max pooling
        TensorDataType *output_buffer = local_output.Buffer(0, sample);
        int *indices_buffer = &m_max_pool_indices[sample * this->get_output_size()];
        LBANN_OMP_PARALLEL_FOR
        for(int channel = 0; channel < num_channels; ++channel) {
          for(int j = 0; j < num_per_output_channel; ++j) {
            TensorDataType *im2col_buffer = im2col_mat.Buffer(channel*m_pool_size, j);
            TensorDataType max_entry = im2col_buffer[0];
            int max_index = 0;
            for(int i = 1; i < m_pool_size; ++i) {
              const TensorDataType current_entry = im2col_buffer[i];
              if(current_entry > max_entry) {
                max_entry = current_entry;
                max_index = i;
              }
            }
            const int output_index = j + channel * num_per_output_channel;
            output_buffer[output_index] = max_entry;
            indices_buffer[output_index] = max_index;
          }
        }
      }

      if(m_pool_mode == pool_mode::average) {
        // Apply average pooling
        TensorDataType *output_buffer = local_output.Buffer(0, sample);
        LBANN_OMP_PARALLEL_FOR
        for(int channel = 0; channel < num_channels; ++channel) {
          for(int j = 0; j < num_per_output_channel; ++j) {
            const TensorDataType *im2col_buffer
              = im2col_mat.LockedBuffer(channel*m_pool_size, j);
            TensorDataType output_entry = El::TypeTraits<TensorDataType>::Zero();
            for(int i = 0; i < m_pool_size; ++i) {
              output_entry += im2col_buffer[i];
            }
            output_entry /= m_pool_size;
            const int output_index = j + channel * num_per_output_channel;
            output_buffer[output_index] = output_entry;
          }
        }
      }

    }

  }

  /// Pooling forward propagation with im2col
  void bp_compute_im2col() {
    using CPUMatType = El::Matrix<TensorDataType, El::Device::CPU>;
    if(m_pool_mode != pool_mode::max && m_pool_mode != pool_mode::average) {
      LBANN_ERROR("CPU pooling layer only supports max and average pooling");
    }

    // Local matrices
    const auto& local_gradient_wrt_output = this->get_local_prev_error_signals();
    auto& local_gradient_wrt_input = this->get_local_error_signals();

    // Pool parameters
    const int local_width = local_gradient_wrt_output.Width();
    const auto& input_dims = this->get_input_dims();
    const int num_channels = input_dims[0];
    const int num_per_input_channel = this->get_output_size() / num_channels;

    // Initialize matrices
    CPUMatType im2col_mat(m_pool_size * num_channels, num_per_input_channel);
    CPUMatType gradient_wrt_input_col;

    // Iterate through data samples
    for(int sample = 0; sample < local_width; ++sample) {

      // Compute gradient w.r.t. im2col matrix for max pooling
      if(m_pool_mode == pool_mode::max) {

        // Clear im2col matrix
        El::Zero(im2col_mat);

        // Copy previous error signal to im2col matrix entries
        // corresponding to max
        const TensorDataType *gradient_wrt_output_buffer
          = local_gradient_wrt_output.LockedBuffer(0, sample);
        const int *indices_buffer
          = &m_max_pool_indices[sample * this->get_output_size()];
        LBANN_OMP_PARALLEL_FOR
        for(int channel = 0; channel < num_channels; ++channel) {
          for(int j = 0; j < num_per_input_channel; ++j) {
            const int input_index = j + channel * num_per_input_channel;
            const int max_index = indices_buffer[input_index];
            TensorDataType *im2col_buffer = im2col_mat.Buffer(channel*m_pool_size, j);
            im2col_buffer[max_index]
              = gradient_wrt_output_buffer[input_index];
          }
        }

      }

      // Compute gradient w.r.t. im2col matrix for average pooling
      if(m_pool_mode == pool_mode::average) {
        const TensorDataType *gradient_wrt_output_buffer
          = local_gradient_wrt_output.LockedBuffer(0, sample);
        LBANN_OMP_PARALLEL_FOR
        for(int channel = 0; channel < num_channels; ++channel) {
          for(int j = 0; j < num_per_input_channel; ++j) {
            TensorDataType *im2col_buffer = im2col_mat.Buffer(channel*m_pool_size, j);
            const int input_index = j + channel * num_per_input_channel;
            const TensorDataType output_entry
              = gradient_wrt_output_buffer[input_index] / El::To<TensorDataType>(m_pool_size);
            for(int i = 0; i < m_pool_size; ++i) {
              im2col_buffer[i] = output_entry;
            }
          }
        }

      }

      // Compute error signal (i.e. gradient w.r.t. input)
      El::View(gradient_wrt_input_col, local_gradient_wrt_input,
               El::ALL, El::IR(sample));
      col2im<TensorDataType>(im2col_mat,
             gradient_wrt_input_col,
             num_channels,
             input_dims.size() - 1,
             &input_dims[1],
             m_pads.data(),
             m_pool_dims.data(),
             m_strides.data());

    }

  }

#ifdef LBANN_HAS_CUDNN
  /** Copy pooling cuDNN descriptor. */
  static void copy_pooling_cudnn_desc(const cudnnPoolingDescriptor_t& src,
                                      cudnnPoolingDescriptor_t& dst) {

    // Create or destroy descriptor if needed
    if(src != nullptr && dst == nullptr) {
        CHECK_CUDNN(cudnnCreatePoolingDescriptor(&dst));
    }
    else if(src == nullptr && dst != nullptr) {
        CHECK_CUDNN(cudnnDestroyPoolingDescriptor(dst));
        dst = nullptr;
    }

    // Copy descriptor data if needed
    if(src != nullptr) {
        cudnnPoolingMode_t mode;
        cudnnNanPropagation_t nan_propagation;
        int num_dims;
        CHECK_CUDNN(cudnnGetPoolingNdDescriptor(src,
                                                0,
                                                &mode,
                                                &nan_propagation,
                                                &num_dims,
                                                nullptr,
                                                nullptr,
                                                nullptr));
        std::vector<int> dims(num_dims), pads(num_dims), strides(num_dims);
        CHECK_CUDNN(cudnnGetPoolingNdDescriptor(src,
                                                num_dims,
                                                &mode,
                                                &nan_propagation,
                                                &num_dims,
                                                dims.data(),
                                                pads.data(),
                                                strides.data()));
        CHECK_CUDNN(cudnnSetPoolingNdDescriptor(dst,
                                                mode,
                                                nan_propagation,
                                                num_dims,
                                                dims.data(),
                                                pads.data(),
                                                strides.data()));
    }

  }
#endif // LBANN_HAS_CUDNN

};

LBANN_DEFINE_LAYER_BUILDER(pooling);

#ifndef LBANN_POOLING_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device) \
  extern template class pooling_layer<T, data_layout::DATA_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_POOLING_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_POOLING_HPP_INCLUDED
