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

#ifndef LBANN_LAYERS_MISC_DIST_EMBEDDING_HPP_INCLUDED
#define LBANN_LAYERS_MISC_DIST_EMBEDDING_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/optimizers/sgd.hpp"
#include "lbann/utils/memory.hpp"

namespace lbann {

namespace dist_embedding_layer_impl {

  /** Request for an embedding vector from a remote process. */
  struct vector_request {
    size_t source_rank{0};
    size_t source_index{0};
    size_t target_rank{0};
    size_t target_index{0};
    long is_completed{0};
    bool is_active{false};
  };

} // namespace dist_embedding_layer_impl

/** @brief Embedding layer with distributed weights.
 *
 *  @warning This is extremely experimental.
 *
 *  @todo Arbitrary unbalanced distributions
 *  @todo Sparse SGD with optimizer class
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class dist_embedding_layer : public data_type_layer<TensorDataType> {
  static_assert(Layout == data_layout::DATA_PARALLEL,
                "distributed embedding layer only supports data parallel layout");

public:

  dist_embedding_layer(
    lbann_comm* comm,
    size_t num_embeddings,
    size_t embedding_dim,
    bool sparse_sgd,
    DataType learning_rate);

  dist_embedding_layer(const dist_embedding_layer& other);
  dist_embedding_layer& operator=(const dist_embedding_layer& other);
  ~dist_embedding_layer();

  dist_embedding_layer* copy() const override;
  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;

  description get_description() const override;

protected:

  void setup_dims() override;
  void setup_data() override;

  void fp_compute() override;
  void bp_compute() override;

private:

  using RequestType = dist_embedding_layer_impl::vector_request;

  /** Size of dictionary of embeddings. */
  size_t m_num_embeddings;
  /** Size of embedding vectors. */
  size_t m_embedding_dim;

  /** Perform sparse SGD during backprop.
   *
   *  Bypasses optimizer class.
   */
  bool m_sparse_sgd;
  /** SGD learning rate. */
  DataType m_learning_rate;

  /** SHMEM buffer to communicate embedding vectors. */
  TensorDataType* m_workspace_buffer{nullptr};
  /** Allocated size of @c m_workspace_buffer. */
  size_t m_workspace_buffer_size{0};

  /** SHMEM buffer to communicate requests for embedding vectors. */
  RequestType* m_requests_buffer{nullptr};
  /** Allocated size of @c m_requests_buffer. */
  size_t m_requests_buffer_size{0};

};

// Builder function
LBANN_DEFINE_LAYER_BUILDER(dist_embedding);

// =============================================
// Implementation
// =============================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
dist_embedding_layer<TensorDataType,Layout,Device>::dist_embedding_layer(
  lbann_comm* comm,
  size_t num_embeddings,
  size_t embedding_dim,
  bool sparse_sgd,
  DataType learning_rate)
  : data_type_layer<TensorDataType>(comm),
    m_num_embeddings{num_embeddings},
    m_embedding_dim{embedding_dim},
    m_sparse_sgd{sparse_sgd},
    m_learning_rate{learning_rate} {

  // Learning rate is only used for sparse SGD
  if (!m_sparse_sgd) {
    m_learning_rate = -1.0;
  }

}

template <typename TensorDataType, data_layout Layout, El::Device Device>
dist_embedding_layer<TensorDataType,Layout,Device>::dist_embedding_layer(
  const dist_embedding_layer& other)
  : data_type_layer<TensorDataType>(other) {
  LBANN_ERROR("copy constructor is invalid for dist_embedding_layer");
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
dist_embedding_layer<TensorDataType,Layout,Device>& dist_embedding_layer<TensorDataType,Layout,Device>::operator=(
  const dist_embedding_layer& other) {
  LBANN_ERROR("copy assignment operator is invalid for dist_embedding_layer");
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
  desc.add("Using sparse SGD", m_sparse_sgd);
  desc.add("SGD learning rate", m_learning_rate);
  return desc;
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

  // Construct default weights if needed
  // Note: Randomly drawn from normal distribution with mean 0 and
  // standard deviation 1.
  if (!this->has_weights()) {
    auto w = make_unique<data_type_weights<TensorDataType>>(this->get_comm());
    auto init = make_unique<normal_initializer<TensorDataType>>(0,1);
    auto opt = this->m_model->template create_optimizer<TensorDataType>();
    w->set_name(this->get_name() + "_weights");
    w->set_initializer(std::move(init));
    w->set_optimizer(std::move(opt));
    this->add_weights(w.get());
    this->m_model->add_weights(std::move(w));
  }
  if (this->num_weights() != 1) {
    LBANN_ERROR("attempted to setup ",
                this->get_type()," layer \"",this->get_name(),"\" ",
                "with an invalid number of weights ",
                "(expected 1, found ",this->num_weights(),")");
  }

  // Configure embedding weights
  auto& embeddings = this->get_data_type_weights(0);
  {
    auto dist = this->get_prev_activations().DistData();
    dist.colDist = El::STAR;
    dist.rowDist = El::VC;
    embeddings.set_dims(
      {static_cast<int>(m_embedding_dim)},
      {static_cast<int>(m_num_embeddings)});
    embeddings.set_matrix_distribution(dist);
  }

  // Destroy embedding optimizer and create dummy weights
  // Note: This layer manually performs sparse SGD on embedding
  // weights during backprop, so the embedding optimizer isn't needed.
  // However, the layer must send gradients to some optimizer to
  // prevent the model from optimizing the layer out of compute graph
  // during backprop. We get around this by creating dummy weights
  // with no entries.
  if (m_sparse_sgd) {
    embeddings.set_optimizer(nullptr);
    auto w = make_unique<data_type_weights<TensorDataType>>(this->get_comm());
    auto opt = make_unique<sgd<TensorDataType>>(0.);
    w->set_name(this->get_name() + "_dummy_weights");
    w->set_optimizer(std::move(opt));
    w->set_dims(1);
    w->set_matrix_distribution(embeddings.get_matrix_distribution());
    w->setup();
    this->add_weights(w.get());
    this->m_model->add_weights(std::move(w));
  }

  // Setup embedding weights
  embeddings.setup();

}

// =============================================
// Explicit template instantiation
// =============================================

extern template class dist_embedding_layer<
  float, data_layout::DATA_PARALLEL, El::Device::CPU>;
extern template class dist_embedding_layer<
  float, data_layout::DATA_PARALLEL, El::Device::GPU>;

} // namespace lbann

#endif // LBANN_LAYERS_MISC_DIST_EMBEDDING_HPP_INCLUDED
