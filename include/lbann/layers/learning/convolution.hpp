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

#ifndef LBANN_LAYERS_LEARNING_CONVOLUTION_HPP_INCLUDED
#define LBANN_LAYERS_LEARNING_CONVOLUTION_HPP_INCLUDED

#include "lbann/layers/learning/base_convolution.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

// Forward declaration.
namespace callback {
class imcomm;
}

/** @brief Standard deep learning convolution.
 *
 *  Applies convolution (more precisely, cross-correlation) to input
 *  tensors. This is primarily optimized for image data in NCHW
 *  format.
 */
template <typename TensorDataType,
          data_layout Layout = data_layout::DATA_PARALLEL,
          El::Device Device = El::Device::CPU>
class convolution_layer : public base_convolution_layer<TensorDataType, Device> {
  static_assert(Layout == data_layout::DATA_PARALLEL,
                "convolution layer only supports DATA_PARALLEL");
private:

  friend class callback::imcomm;

public:

  convolution_layer(lbann_comm *comm,
                    int num_data_dims,
                    int num_output_channels,
                    int conv_dim,
                    int pad,
                    int stride,
                    int dilation,
                    int groups,
                    bool has_bias = true)
    : convolution_layer(comm,
                        num_data_dims,
                        num_output_channels,
                        std::vector<int>(num_data_dims, conv_dim),
                        std::vector<int>(num_data_dims, pad),
                        std::vector<int>(num_data_dims, stride),
                        std::vector<int>(num_data_dims, dilation),
                        groups,
                        has_bias) {}

  convolution_layer(lbann_comm *comm,
                    int num_data_dims,
                    int num_output_channels,
                    std::vector<int> conv_dims,
                    std::vector<int> pads,
                    std::vector<int> strides,
                    std::vector<int> dilations,
                    int groups,
                    bool has_bias = true)
    : base_convolution_layer<TensorDataType, Device>(
        comm,
        num_data_dims,
        num_output_channels,
        std::move(conv_dims),
        std::move(pads),
        std::move(strides),
        std::move(dilations),
        groups,
        has_bias) {
  }

  convolution_layer* copy() const override { return new convolution_layer(*this); }

  std::string get_type() const override { return "convolution"; }

  data_layout get_data_layout() const override { return Layout; }

  El::Device get_device_allocation() const override { return Device; }

protected:

  void setup_dims() override {
    base_convolution_layer<TensorDataType, Device>::setup_dims();

    // Get tensor dimensions
    const auto& input_dims = this->get_input_dims();
    auto output_dims = input_dims;

    // Initialize output tensor dimensions
    output_dims[0] = this->m_output_channels;
    for (size_t i = 0; i < output_dims.size() - 1; ++i) {
      const auto& input_dim = input_dims[i+1];
      const auto& kernel_dim = this->m_conv_dims[i];
      const auto& stride = this->m_strides[i];
      const auto& pad = this->m_pads[i];
      const auto& dilation = this->m_dilations[i];
      const auto& effective_dim = (input_dim
                                   + 2 * pad
                                   - dilation * (kernel_dim-1));
      output_dims[i+1] = (effective_dim + stride - 1) / stride;
    }
    this->set_output_dims(output_dims);

  }

  std::vector<int> get_kernel_dims() const override {
    std::vector<int> dims;
    dims.push_back(this->m_output_channels);
    dims.push_back(this->get_input_dims()[0] / this->m_groups);
    dims.insert(dims.end(),
                this->m_conv_dims.begin(),
                this->m_conv_dims.end());
    return dims;
  }

  void fp_compute() override {
    if(this->using_gpus()) {
      base_convolution_layer<TensorDataType, Device>::apply_convolution_cudnn(true);
      base_convolution_layer<TensorDataType, Device>::apply_bias_cudnn();
    } else {
      base_convolution_layer<TensorDataType, Device>::apply_convolution_im2col(true);
      base_convolution_layer<TensorDataType, Device>::apply_bias_cpu();
    }
  }

  void bp_compute() override {
    if(this->using_gpus()) {
      base_convolution_layer<TensorDataType, Device>::compute_gradients_cudnn(false);
      base_convolution_layer<TensorDataType, Device>::apply_transposed_convolution_cudnn(false);
    } else {
      base_convolution_layer<TensorDataType, Device>::compute_gradients_im2col(false);
      base_convolution_layer<TensorDataType, Device>::apply_transposed_convolution_im2col(false);
    }
  }

};

// Builder function
LBANN_DEFINE_LAYER_BUILDER(convolution);

#ifndef LBANN_CONVOLUTION_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device) \
  extern template class convolution_layer<T, data_layout::DATA_PARALLEL, Device>;

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_CONVOLUTION_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_LEARNING_CONVOLUTION_HPP_INCLUDED
