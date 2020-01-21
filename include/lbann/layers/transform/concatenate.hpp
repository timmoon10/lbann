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

#ifndef LBANN_LAYERS_TRANSFORM_CONCATENATE_HPP_INCLUDED
#define LBANN_LAYERS_TRANSFORM_CONCATENATE_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/exception.hpp"

#include <lbann/proto/proto_common.hpp>
#include <layers.pb.h>

namespace lbann {

/** @brief Concatenate tensors along specified dimension. */
template <typename TensorDataType,
          data_layout Layout = data_layout::DATA_PARALLEL,
          El::Device Device = El::Device::CPU>
class concatenate_layer : public data_type_layer<TensorDataType> {
public:

  concatenate_layer(lbann_comm *comm, size_t concat_dim);
  concatenate_layer(const concatenate_layer& other) = default;
  concatenate_layer& operator=(const concatenate_layer& other) = default;

  concatenate_layer* copy() const override;
  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;

  description get_description() const override;

protected:

  void setup_pointers() override;
  void setup_dims() override;

  void fp_setup_outputs(El::Int mini_batch_size) override;
  void bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) override;
  void fp_compute() override;
  void bp_compute() override;

private:

  /** @brief Tensor dimension to concatenate along. */
  size_t m_concat_dim;

#ifdef LBANN_HAS_GPU
  /** @brief Workspace buffer.
   *
   *  Parameters for CUDA kernels are copied into this buffer and
   *  asynchronously transferred to GPU.
   */
  std::vector<unsigned char> m_workspace;
  /** @brief CUDA event for workspace buffer.
   *
   *  Makes sure asynchronous GPU memory transfers are completed
   *  before modifying workspace buffer.
   */
  cuda::event_wrapper m_workspace_event;
#endif // LBANN_HAS_GPU

  template <typename U>
  friend void fp_compute_impl(concatenate_layer<U,Layout,Device>&, size_t);
  template <typename U, El::Device D>
  friend void bp_setup_gradient_wrt_inputs_impl(concatenate_layer<U,Layout,D>&);
  template <typename U>
  friend void bp_compute_impl(concatenate_layer<U,Layout,Device>&, size_t);

};

// =========================================================
// Implementation
// =========================================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
concatenate_layer<TensorDataType,Layout,Device>::concatenate_layer(
  lbann_comm *comm,
  size_t concat_dim)
  : data_type_layer<TensorDataType>(comm),
    m_concat_dim{concat_dim} {
  this->m_expected_num_parent_layers = -1; // No limit on parents
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
concatenate_layer<TensorDataType, Layout,Device>* concatenate_layer<TensorDataType,Layout,Device>::copy() const {
  return new concatenate_layer(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::string concatenate_layer<TensorDataType,Layout,Device>::get_type() const {
  return "concatenate";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
data_layout concatenate_layer<TensorDataType,Layout,Device>::get_data_layout() const {
  return Layout;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
El::Device concatenate_layer<TensorDataType,Layout,Device>::get_device_allocation() const {
  return Device;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
description concatenate_layer<TensorDataType,Layout,Device>::get_description() const {
  auto desc = data_type_layer<TensorDataType>::get_description();
  desc.add("Concatenation dimension", m_concat_dim);
  return desc;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void concatenate_layer<TensorDataType,Layout,Device>::setup_pointers() {
  data_type_layer<TensorDataType>::setup_pointers();
  if (this->get_num_parents() < 1) {
    LBANN_ERROR(get_type()," layer \"",this->get_name(),"\" ",
                "has no parents");
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void concatenate_layer<TensorDataType,Layout,Device>::setup_dims() {
  data_type_layer<TensorDataType>::setup_dims();

  // Dimensions of first input tensor
  auto output_dims = this->get_input_dims(0);
  if (m_concat_dim >= output_dims.size()) {
    std::ostringstream err;
    err << get_type() << " layer \"" << this->get_name() << "\" "
        << "is concatenating along dimension " << m_concat_dim << ", "
        << "but it has a " << output_dims.size() << "-D input tensor "
        << "(parent layer \"" << this->get_parent_layers()[0]->get_name() << "\" "
        << "outputs with dimensions ";
    for (size_t d=0; d<output_dims.size(); ++d) {
      err << (d>0 ? " x " : "") << output_dims[d];
    }
    err << ")";
    LBANN_ERROR(err.str());
  }

  // Dimensions of remaining input tensors
  for (int j=1; j<this->get_num_parents(); ++j) {
    const auto& input_dims = this->get_input_dims(j);
    if (input_dims.size() != output_dims.size()
        || !std::equal(input_dims.begin(),
                       input_dims.begin() + m_concat_dim,
                       output_dims.begin())
        || !std::equal(input_dims.begin() + m_concat_dim + 1,
                       input_dims.end(),
                       output_dims.begin() + m_concat_dim + 1)) {
      std::ostringstream err;
      err << get_type() << " layer \"" << this->get_name() << "\" "
          << "expects input tensors with dimensions ";
      for (size_t d=0; d<output_dims.size(); ++d) {
        err << (d>0 ? " x " : "");
        if (d == m_concat_dim) { err << "X"; }
        else { err << output_dims[d]; }
      }
      err << ", but parent layer "
          << "\"" << this->get_parent_layers()[j]->get_name() << "\" "
          << "outputs with dimensions ";
      for (size_t d=0; d < input_dims.size(); ++d) {
        err << (d>0 ? " x " : "") << input_dims[d];
      }
      LBANN_ERROR(err.str());
    }
    output_dims[m_concat_dim] += input_dims[m_concat_dim];
  }

  // Model-parallel implementation only supports flat data
  if (Layout == data_layout::MODEL_PARALLEL
      && std::accumulate(&output_dims[0], &output_dims[m_concat_dim], 1, std::multiplies<int>()) > 1) {
    LBANN_ERROR(this->get_type()," layer \"",this->get_name(),"\" ",
                "attempted to concatenate along dimension ",m_concat_dim,", ",
                "but model-parallel concatenate layer "
                "only supports flat data");
  }

  // Update output dimensions
  this->set_output_dims(output_dims);

}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void concatenate_layer<TensorDataType,Layout,Device>::fp_setup_outputs(El::Int mini_batch_size) {
  const auto& input0 = this->get_prev_activations(0);
  auto& output = this->get_activations();
  output.Empty(false);
  if (this->get_num_parents() == 1) {
    El::LockedView(output, input0);
  }
  else {
    output.AlignWith(input0);
    output.Resize(this->get_output_size(), input0.Width());
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void concatenate_layer<TensorDataType,Layout,Device>::fp_compute() {

  // Just make a view if there is one input
  if (this->get_num_parents() == 1) {
    El::LockedView(this->get_activations(), this->get_prev_activations(0));
    return;
  }

  // Perform concatenation
  fp_compute_impl(*this, m_concat_dim);

}

template <typename TensorDataType, El::Device Device>
void bp_setup_gradient_wrt_inputs_impl(
  concatenate_layer<TensorDataType,data_layout::MODEL_PARALLEL,Device>& l) {

  // Slice Elemental matrices
  // Note: Assume each mini-batch sample is flat.
  const size_t num_inputs = l.get_num_parents();
  const auto& output_grad = l.get_prev_error_signals();
  size_t offset = 0;
  for (size_t j=0; j<num_inputs; ++j) {
    auto& input_grad = l.get_error_signals(j);
    const auto& input_size = l.get_input_size(j);
    El::LockedView(input_grad, output_grad,
                   El::IR(offset, offset+input_size), El::ALL);
    offset += input_size;
  }

}

template <typename TensorDataType, El::Device Device>
void bp_setup_gradient_wrt_inputs_impl(
  concatenate_layer<TensorDataType,data_layout::DATA_PARALLEL,Device>& l) {

  const size_t num_inputs = l.get_num_parents();
  const auto& output_grad = l.get_prev_error_signals();
  if (num_inputs == 1) {
    El::LockedView(l.get_error_signals(0), output_grad);
  }
  else {
    for (size_t j=0; j<num_inputs; ++j) {
      auto& input_grad = l.get_error_signals(j);
      input_grad.AlignWith(output_grad);
      input_grad.Resize(l.get_input_size(j), output_grad.Width());
    }
  }

}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void concatenate_layer<TensorDataType,Layout,Device>::bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) {
  bp_setup_gradient_wrt_inputs_impl(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void concatenate_layer<TensorDataType,Layout,Device>::bp_compute() {

  // Just make a view if there is one input
  if (this->get_num_parents() == 1) {
    El::LockedView(this->get_error_signals(0), this->get_prev_error_signals());
    return;
  }

  // Perform slice
  bp_compute_impl(*this, m_concat_dim);

}

LBANN_DEFINE_LAYER_BUILDER(concatenate);

#ifndef LBANN_CONCATENATE_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device)             \
  extern template class concatenate_layer<  \
    T, data_layout::DATA_PARALLEL, Device>; \
  extern template class concatenate_layer<  \
    T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_CONCATENATE_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_TRANSFORM_CONCATENATE_HPP_INCLUDED
