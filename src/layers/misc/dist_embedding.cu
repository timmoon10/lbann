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

namespace lbann {

// =============================================
// Life-cycle and utility functions
// =============================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
dist_embedding_layer<TensorDataType,Layout,Device>::dist_embedding_layer(
  lbann_comm* comm,
  size_t num_embeddings,
  size_t embedding_dim)
  : data_type_layer<TensorDataType>(comm),
    m_num_embeddings{num_embeddings},
    m_embedding_dim{embedding_dim}
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
dist_embedding_layer<TensorDataType,Layout,Device>::dist_embedding_layer(
 const dist_embedding_layer<TensorDataType,Layout,Device>& other)
  : data_type_layer<TensorDataType>(other),
    m_num_embeddings{other.m_num_embeddings},
    m_embedding_dim{other.m_embedding_dim}
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
dist_embedding_layer<TensorDataType,Layout,Device>& dist_embedding_layer<TensorDataType,Layout,Device>::operator=(
  const dist_embedding_layer<TensorDataType,Layout,Device>& other) {
  data_type_layer<TensorDataType>::operator=(other);
  m_num_embeddings = other.m_num_embeddings;
  m_embedding_dim = other.m_embedding_dim;
  return *this;
}

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

#ifdef LBANN_HAS_GPU_FP16

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
}

#endif // LBANN_HAS_GPU_FP16

// =============================================
// Forward prop
// =============================================

#ifdef LBANN_HAS_GPU_FP16

template <typename TensorDataType, data_layout Layout, El::Device Device>
void dist_embedding_layer<TensorDataType,Layout,Device>::fp_compute() {
}

#endif // LBANN_HAS_GPU_FP16

// =============================================
// Backprop
// =============================================

#ifdef LBANN_HAS_GPU_FP16

template <typename TensorDataType, data_layout Layout, El::Device Device>
void dist_embedding_layer<TensorDataType,Layout,Device>::bp_compute() {
}

#endif // LBANN_HAS_GPU_FP16

// =============================================
// Builder function
// =============================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer> build_dist_embedding_layer_from_pbuf(
  lbann_comm* comm,
  const lbann_data::Layer& proto_layer) {
  LBANN_ERROR("distributed embedding layer is only supported with ",
              "fp16 datatype, data-parallel layout, and GPU");
}

#ifdef LBANN_HAS_GPU_FP16
template <>
std::unique_ptr<Layer> build_dist_embedding_layer_from_pbuf<fp16,data_layout::DATA_PARALLEL,El::Device::GPU>(
  lbann_comm* comm,
  const lbann_data::Layer& proto_layer) {
  const auto& params = proto_layer.dist_embedding();
  const size_t num_embeddings = params.num_embeddings();
  const size_t embedding_dim = params.embedding_dim();
  return lbann::make_unique<dist_embedding_layer<fp16,data_layout::DATA_PARALLEL,El::Device::GPU>>(
    comm, num_embeddings, embedding_dim);
}
#endif // LBANN_HAS_GPU_FP16

// =============================================
// Explicit template instantiation
// =============================================

#ifdef LBANN_HAS_GPU_FP16
template class dist_embedding_layer<
  fp16, data_layout::DATA_PARALLEL, El::Device::GPU>;
#endif // LBANN_HAS_GPU_FP16

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
