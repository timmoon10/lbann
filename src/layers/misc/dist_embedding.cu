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
#include "lbann/models/model.hpp"
#include "lbann/optimizers/sgd.hpp"
#include "lbann/utils/memory.hpp"

#include <layers.pb.h>

namespace lbann {

// =============================================
// Life-cycle and utility functions
// =============================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
dist_embedding_layer<TensorDataType,Layout,Device>::dist_embedding_layer(
  lbann_comm* comm,
  size_t num_embeddings,
  size_t embedding_dim,
  DataType learning_rate)
  : data_type_layer<TensorDataType>(comm),
    m_num_embeddings{num_embeddings},
    m_embedding_dim{embedding_dim},
    m_learning_rate{learning_rate}
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
dist_embedding_layer<TensorDataType,Layout,Device>* dist_embedding_layer<TensorDataType,Layout,Device>::copy() const {
  return new dist_embedding_layer(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::string dist_embedding_layer<TensorDataType,Layout,Device>::get_type() const {
  return "distributed embedding";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
data_layout dist_embedding_layer<TensorDataType,Layout,Device>::get_data_layout() const {
  return Layout;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
El::Device dist_embedding_layer<TensorDataType,Layout,Device>::get_device_allocation() const {
  return Device;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
description dist_embedding_layer<TensorDataType,Layout,Device>::get_description() const {
  auto desc = data_type_layer<TensorDataType>::get_description();
  desc.add("Num embeddings", m_num_embeddings);
  desc.add("Embedding dim", m_embedding_dim);
  return desc;
}

// =============================================
// Setup functions
// =============================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
void dist_embedding_layer<TensorDataType,Layout,Device>::setup_matrices(const El::Grid& grid) {
  data_type_layer<TensorDataType>::setup_matrices(grid);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void dist_embedding_layer<TensorDataType,Layout,Device>::setup_dims() {
  data_type_layer<TensorDataType>::setup_dims();
  auto dims = this->get_input_dims();
  dims.push_back(static_cast<int>(m_embedding_dim));
  this->set_output_dims(dims);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void dist_embedding_layer<TensorDataType,Layout,Device>::setup_data() {
  data_type_layer<TensorDataType>::setup_data();

  // Create random embeddings on trainer master rank
  m_local_embeddings.Resize(m_embedding_dim, m_num_embeddings);
  if (this->get_comm()->am_trainer_master()) {
    El::Gaussian(m_local_embeddings,
                 m_embedding_dim,
                 m_num_embeddings,
                 TensorDataType{0.},
                 TensorDataType(1. / m_embedding_dim));
  }
  El::Broadcast(
    reinterpret_cast<El::AbstractMatrix<TensorDataType>&>(m_local_embeddings),
    this->get_comm()->get_trainer_comm(),
    0);

  // Create dummy weights
  // Note: Prevents model from optimizing away this layer during
  // backprop.
  auto w = make_unique<data_type_weights<TensorDataType>>(this->get_comm());
  w->set_name(this->get_name() + "_dummy_weights");
  w->set_optimizer(make_unique<sgd<TensorDataType>>(0.));
  this->set_data_type_weights(0, w.get());
  this->m_model->add_weights(std::move(w));

}

// =============================================
// Forward prop
// =============================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
void dist_embedding_layer<TensorDataType,Layout,Device>::fp_compute() {
  using LocalMat = El::Matrix<TensorDataType, El::Device::GPU>;
  using CPUMat = El::Matrix<TensorDataType, El::Device::CPU>;

  // Local data
  const auto& local_input = dynamic_cast<const LocalMat&>(this->get_local_prev_activations());
  auto& local_output = dynamic_cast<LocalMat&>(this->get_local_activations());
  const size_t input_size = this->get_input_size();
  const size_t local_mini_batch_size = local_input.Width();

  // Copy input to CPU
  const CPUMat local_input_cpu(local_input);

  // Populate output matrix with values from embedding matrix
  LocalMat embeddings_v, output_v;
  for (size_t j=0; j<local_mini_batch_size; ++j) {
    for (size_t i=0; i<input_size; ++i) {
      const auto& ind = static_cast<size_t>(std::floor(local_input_cpu(i,j)));
      El::View(output_v, local_output,
               El::IR(i*m_embedding_dim, (i+1)*m_embedding_dim),
               El::IR(j));
      El::LockedView(embeddings_v, m_local_embeddings, El::ALL, El::IR(ind));
      El::Copy(embeddings_v, output_v);
    }
  }

}

// =============================================
// Backprop
// =============================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
void dist_embedding_layer<TensorDataType,Layout,Device>::bp_compute() {
  using LocalMat = El::Matrix<TensorDataType, El::Device::GPU>;
  using CPUMat = El::Matrix<TensorDataType, El::Device::CPU>;

  // Local data
  const auto& local_input = dynamic_cast<const LocalMat&>(this->get_local_prev_activations());
  const auto& local_output_grad = dynamic_cast<const LocalMat&>(this->get_local_prev_activations());
  const size_t input_size = this->get_input_size();
  const size_t local_mini_batch_size = local_input.Width();

  // Copy input to CPU
  const CPUMat local_input_cpu(local_input);

  // Compute gradient w.r.t. embeddings
  LocalMat local_embeddings_grad;
  El::Zeros(local_embeddings_grad, m_embedding_dim, m_num_embeddings);
  LocalMat embeddings_grad_v, output_grad_v;
  for (size_t j=0; j<local_mini_batch_size; ++j) {
    for (size_t i=0; i<input_size; ++i) {
      const auto& ind = static_cast<size_t>(std::floor(local_input_cpu(i,j)));
      El::LockedView(output_grad_v, local_output_grad,
                     El::IR(i*m_embedding_dim, (i+1)*m_embedding_dim),
                     El::IR(j));
      El::View(embeddings_grad_v, local_embeddings_grad, El::ALL, El::IR(ind));
      El::Copy(output_grad_v, embeddings_grad_v);
    }
  }
  El::AllReduce(
    local_embeddings_grad,
    this->get_comm()->get_trainer_comm());

  // Stochastic gradient descent
  El::Axpy(-m_learning_rate, local_embeddings_grad, m_local_embeddings);

}

// =============================================
// Builder function
// =============================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer> build_dist_embedding_layer_from_pbuf(
  lbann_comm* comm,
  const lbann_data::Layer& proto_layer) {
  LBANN_ERROR("distributed embedding layer is only supported with ",
              "float datatype, data-parallel layout, and GPU");
}

template <>
std::unique_ptr<Layer> build_dist_embedding_layer_from_pbuf<float,data_layout::DATA_PARALLEL,El::Device::GPU>(
  lbann_comm* comm,
  const lbann_data::Layer& proto_layer) {
  const auto& params = proto_layer.dist_embedding();
  return make_unique<dist_embedding_layer<float,data_layout::DATA_PARALLEL,El::Device::GPU>>(
    comm,
    params.num_embeddings(),
    params.embedding_dim(),
    params.learning_rate());
}

// =============================================
// Explicit template instantiation
// =============================================

/// @todo fp16
template class dist_embedding_layer<
  float, data_layout::DATA_PARALLEL, El::Device::GPU>;

#define PROTO(T)                                                        \
  template std::unique_ptr<Layer>                                       \
  build_dist_embedding_layer_from_pbuf<T,data_layout::DATA_PARALLEL,El::Device::GPU>( \
    lbann_comm*, lbann_data::Layer const&);                             \
  template std::unique_ptr<Layer>                                       \
  build_dist_embedding_layer_from_pbuf<T,data_layout::MODEL_PARALLEL,El::Device::GPU>( \
    lbann_comm*, lbann_data::Layer const&)
#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
