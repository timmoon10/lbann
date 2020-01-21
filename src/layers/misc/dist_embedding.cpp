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
#include "lbann/utils/memory.hpp"

#include <layers.pb.h>

namespace lbann {

// =============================================
// Forward prop
// =============================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
void dist_embedding_layer<TensorDataType,Layout,Device>::fp_compute() {
  using LocalMat = El::Matrix<TensorDataType, Device>;

  // Local data
  const auto& local_embeddings = dynamic_cast<const LocalMat&>(this->get_data_type_weights(0).get_values().LockedMatrix());
  const auto& local_input = dynamic_cast<const LocalMat&>(this->get_local_prev_activations());
  auto& local_output = dynamic_cast<LocalMat&>(this->get_local_activations());
  const size_t input_size = this->get_input_size();
  const size_t local_mini_batch_size = local_input.Width();

  // Populate output matrix with values from embedding matrix
  LocalMat embeddings_v, output_v;
  for (size_t j=0; j<local_mini_batch_size; ++j) {
    for (size_t i=0; i<input_size; ++i) {
      const auto& ind = static_cast<size_t>(std::floor(local_input(i,j)));
      El::View(output_v, local_output,
               El::IR(i*m_embedding_dim, (i+1)*m_embedding_dim),
               El::IR(j));
      El::LockedView(embeddings_v, local_embeddings, El::ALL, El::IR(ind));
      El::Copy(embeddings_v, output_v);
    }
  }

}

// =============================================
// Backprop
// =============================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
void dist_embedding_layer<TensorDataType,Layout,Device>::bp_compute() {
  using DistMat_ = El::DistMatrix<TensorDataType,El::STAR,El::STAR,El::ELEMENT,Device>;
  using LocalMat = El::Matrix<TensorDataType, Device>;

  // Local data
  auto& embeddings = dynamic_cast<DistMat_&>(this->get_data_type_weights(0).get_values());
  const auto& local_input = dynamic_cast<const LocalMat&>(this->get_local_prev_activations());
  const auto& local_output_grad = dynamic_cast<const LocalMat&>(this->get_local_prev_error_signals());
  const size_t input_size = this->get_input_size();
  const size_t local_mini_batch_size = local_input.Width();

  // Embedding layer is not differentiable w.r.t. inputs
  El::Zero(this->get_error_signals());

  // Compute gradient w.r.t. embeddings
  DistMat_ embeddings_grad(embeddings.Grid());
  El::Zeros(embeddings_grad, m_embedding_dim, m_num_embeddings);
  LocalMat& local_embeddings_grad = embeddings_grad.Matrix();
  LocalMat embeddings_grad_v, output_grad_v;
  for (size_t j=0; j<local_mini_batch_size; ++j) {
    for (size_t i=0; i<input_size; ++i) {
      const auto& ind = static_cast<size_t>(std::floor(local_input(i,j)));
      El::LockedView(output_grad_v, local_output_grad,
                     El::IR(i*m_embedding_dim, (i+1)*m_embedding_dim),
                     El::IR(j));
      El::View(embeddings_grad_v, local_embeddings_grad, El::ALL, El::IR(ind));
      El::Axpy(TensorDataType{1.}, output_grad_v, embeddings_grad_v);
    }
  }
  El::AllReduce(embeddings_grad, embeddings_grad.RedundantComm());

#ifdef LBANN_DIST_EMBEDDING_SPARSE_SGD

  // Stochastic gradient descent
  El::Axpy(-m_learning_rate, embeddings_grad, embeddings);

#else // LBANN_DIST_EMBEDDING_SPARSE_SGD

  // Send gradient to optimizer
  auto& opt = *this->get_data_type_weights(0).get_optimizer();
  opt.add_to_gradient(embeddings_grad);

#endif // LBANN_DIST_EMBEDDING_SPARSE_SGD

}

// =============================================
// Builder function
// =============================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer> build_dist_embedding_layer_from_pbuf(
  lbann_comm* comm,
  const lbann_data::Layer& proto_layer) {
  LBANN_ERROR(
    "Attempted to construct dist_embedding_layer ",
    "with invalid parameters ",
    "(TensorDataType=",TypeName<TensorDataType>(),", ",
    "Layout=",to_string(Layout),", ",
    "Device=",to_string(Device),")");
  return nullptr;
}

template <>
std::unique_ptr<Layer> build_dist_embedding_layer_from_pbuf<float,data_layout::DATA_PARALLEL,El::Device::CPU>(
  lbann_comm* comm,
  const lbann_data::Layer& proto_layer) {
  const auto& params = proto_layer.dist_embedding();
  return make_unique<dist_embedding_layer<float,data_layout::DATA_PARALLEL,El::Device::CPU>>(
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
  float, data_layout::DATA_PARALLEL, El::Device::CPU>;

#define PROTO(T)                                                        \
  template std::unique_ptr<Layer>                                       \
  build_dist_embedding_layer_from_pbuf<T,data_layout::DATA_PARALLEL,El::Device::CPU>( \
    lbann_comm*, lbann_data::Layer const&);                             \
  template std::unique_ptr<Layer>                                       \
  build_dist_embedding_layer_from_pbuf<T,data_layout::MODEL_PARALLEL,El::Device::CPU>( \
    lbann_comm*, lbann_data::Layer const&)
#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
