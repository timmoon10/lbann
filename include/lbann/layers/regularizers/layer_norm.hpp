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

#ifndef LBANN_LAYERS_REGULARIZERS_LAYER_NORM_HPP_INCLUDED
#define LBANN_LAYERS_REGULARIZERS_LAYER_NORM_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"

#include <memory>

namespace lbann {

/** @brief
 *
 *  Each data sample is normalized to have zero mean and unit standard
 *  deviation. See:
 *
 *  Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer
 *  normalization." arXiv preprint arXiv:1607.06450 (2016).
 *
 *  Note that this layer does not apply an entry-wise scale and bias
 *  like in the paper. Use the entry-wise scale/bias layer to
 *  reproduce that functionality.
 *
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class layer_norm_layer : public data_type_layer<TensorDataType> {
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  ///@}

public:

  /**
   *  @param comm       LBANN communicator
   *  @param epsilon    Small number to avoid division by zero
   */
  layer_norm_layer(lbann_comm* comm, TensorDataType epsilon=1e-5);

  layer_norm_layer(const layer_norm_layer& other);
  layer_norm_layer& operator=(const layer_norm_layer& other);
  layer_norm_layer* copy() const override;

  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;
  description get_description() const override;

protected:

  void setup_dims() override;
  void setup_matrices(const El::Grid& grid) override;
  void fp_setup_outputs(El::Int mini_batch_size) override;
  void bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) override;

  void fp_compute() override;
  void bp_compute() override;

private:

  using AbsDistMatType = El::AbstractDistMatrix<TensorDataType>;

  /** Small number to avoid division by zero. */
  TensorDataType m_epsilon;

  /** @brief Per-sample statistics.
   *
   *  The means and variances are fused for performance.
   */
  std::unique_ptr<AbsDistMatType> m_statistics;
  /** @brief Gradients w.r.t. per-sample statistics.
   *
   *  The means and variances are fused for performance.
   */
  std::unique_ptr<AbsDistMatType> m_statistics_gradient;

};

// =========================================================
// Implementation
// =========================================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
layer_norm_layer<TensorDataType,Layout,Device>::layer_norm_layer(
  lbann_comm* comm,
  TensorDataType epsilon)
  : data_type_layer<TensorDataType>(comm), m_epsilon(epsilon)
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
layer_norm_layer<TensorDataType,Layout,Device>::layer_norm_layer(
  const layer_norm_layer<TensorDataType,Layout,Device>& other)
  : data_type_layer<TensorDataType>(other),
    m_epsilon(other.m_epsilon),
    m_statistics(other.m_statistics
                 ? other.m_statistics->Copy()
                 : nullptr),
    m_statistics_gradient(other.m_statistics_gradient
                          ? other.m_statistics_gradient->Copy()
                          : nullptr)
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
layer_norm_layer<TensorDataType,Layout,Device>& layer_norm_layer<TensorDataType,Layout,Device>::operator=(
  const layer_norm_layer<TensorDataType,Layout,Device>& other) {
  data_type_layer<TensorDataType>::operator=(other);
  m_epsilon = other.m_epsilon;
  m_statistics.reset(other.m_statistics
                     ? other.m_statistics->Copy()
                     : nullptr);
  m_statistics_gradient.reset(other.m_statistics_gradient
                              ? other.m_statistics_gradient->Copy()
                              : nullptr);
  return *this;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
layer_norm_layer<TensorDataType,Layout,Device>* layer_norm_layer<TensorDataType,Layout,Device>::copy() const {
  return new layer_norm_layer(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::string layer_norm_layer<TensorDataType,Layout,Device>::get_type() const {
  return "layer norm";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
data_layout layer_norm_layer<TensorDataType,Layout,Device>::get_data_layout() const {
  return Layout;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
El::Device layer_norm_layer<TensorDataType,Layout,Device>::get_device_allocation() const {
  return Device;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
description layer_norm_layer<TensorDataType,Layout,Device>::get_description() const {
  auto desc = data_type_layer<TensorDataType>::get_description();
  desc.add("Epsilon", m_epsilon);
  return desc;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void layer_norm_layer<TensorDataType,Layout,Device>::setup_dims() {
  data_type_layer<TensorDataType>::setup_dims();
  this->set_output_dims(this->get_input_dims());
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void layer_norm_layer<TensorDataType,Layout,Device>::setup_matrices(const El::Grid& grid) {
  data_type_layer<TensorDataType>::setup_matrices(grid);
  auto dist = this->get_prev_activations().DistData();
  dist.colDist = El::STAR;
  m_statistics.reset(AbsDistMatrixType::Instantiate(dist));
  m_statistics_gradient.reset(AbsDistMatrixType::Instantiate(dist));
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void layer_norm_layer<TensorDataType,Layout,Device>::fp_setup_outputs(El::Int mini_batch_size) {
  data_type_layer<TensorDataType>::fp_setup_outputs(mini_batch_size);
  const auto& input = this->get_prev_activations();
  m_statistics->Empty(false);
  m_statistics->AlignWith(input);
  m_statistics->Resize(2, input.Width());
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void layer_norm_layer<TensorDataType,Layout,Device>::bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) {
  data_type_layer<TensorDataType>::bp_setup_gradient_wrt_inputs(mini_batch_size);
  const auto& input = this->get_prev_activations();
  m_statistics_gradient->Empty(false);
  m_statistics_gradient->AlignWith(input);
  m_statistics_gradient->Resize(2, input.Width());
}

// =========================================================
// Explicit template instantiation
// =========================================================

#ifndef LBANN_LAYER_NORM_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device) \
  extern template class layer_norm_layer<   \
    T, data_layout::DATA_PARALLEL, Device>; \
  extern template class layer_norm_layer<   \
    T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_LAYER_NORM_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_REGULARIZERS_LAYER_NORM_HPP_INCLUDED
