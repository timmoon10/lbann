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

#ifdef LBANN_HAS_SHMEM
#include <shmem.h>
#endif // LBANN_HAS_SHMEM

namespace lbann {

// =============================================
// Setup functions
// =============================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
dist_embedding_layer<TensorDataType,Layout,Device>::~dist_embedding_layer()
{
#ifdef LBANN_HAS_SHMEM
  if (m_embeddings_buffer != nullptr) {
    shmem_free(m_embeddings_buffer);
  }
  if (m_embeddings_grad_buffer != nullptr) {
    shmem_free(m_embeddings_grad_buffer);
  }
  if (m_requests_buffer != nullptr) {
    shmem_free(m_requests_buffer);
  }
#endif // LBANN_HAS_SHMEM
}

namespace {

template <typename T>
T* attach_dist_matrix_to_shmem_buffer(
  El::AbstractDistMatrix<T>& mat,
  size_t height,
  size_t width) {

  // Make sure matrix distribution is valid
  const auto dist = mat.DistData();
  if (dist.device != El::Device::CPU) {
    LBANN_ERROR("attempted to attach non-CPU matrix to OpenSHMEM buffer");
  }

  // Allocate SHMEM buffer
  const size_t col_comm_size = El::mpi::Size(mat.ColComm());
  const size_t row_comm_size = El::mpi::Size(mat.RowComm());
  const size_t local_height = (height + col_comm_size - 1) / col_comm_size;
  const size_t local_width = (width + row_comm_size - 1) / row_comm_size;
  auto* buffer = reinterpret_cast<T*>(
    shmem_malloc(local_height * local_width * sizeof(T)));
  if (buffer == nullptr) {
    LBANN_ERROR("failed to allocate OpenSHMEM buffer");
  }

  // Attach matrix to SHMEM buffer
  std::unique_ptr<El::AbstractDistMatrix<T>> orig_mat(
    mat.Construct(mat.Grid(), mat.Root()));
  *orig_mat = std::move(mat);
  mat.Empty();
  mat.AlignWith(dist);
  dynamic_cast<El::ElementalMatrix<T>&>(mat).Attach(
    height, width,
    *dist.grid, dist.colAlign, dist.rowAlign,
    buffer, local_height, dist.root);
  if (mat.Height() == orig_mat->Height()
      && mat.Width() == orig_mat->Width()) {
    El::Copy(*orig_mat, mat);
  }
  return buffer;

}

} // namespace <anon>

template <typename TensorDataType, data_layout Layout, El::Device Device>
void dist_embedding_layer<TensorDataType,Layout,Device>::setup_data() {
  data_type_layer<TensorDataType>::setup_data();
#ifndef LBANN_HAS_SHMEM
  LBANN_ERROR(
    "dist_embedding_layer with ",
    "(TensorDataType=",TypeName<TensorDataType>(),", ",
    "Layout=",to_string(Layout),", ",
    "Device=",to_string(Device),") ",
    "requires SHMEM, but LBANN has not been built with SHMEM");
  return;
#else // LBANN_HAS_SHMEM

  // Construct default weights if needed
  // Note: Randomly drawn from normal distribution with mean 0 and
  // standard deviation 1.
  if (!this->has_weights()) {
    auto w = make_unique<data_type_weights<TensorDataType>>(this->get_comm());
    auto init = make_unique<normal_initializer<TensorDataType>>(0,1);
    w->set_name(this->get_name() + "_weights");
    w->set_initializer(std::move(init));
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

  // Set dummy optimizer
  // Note: This layer manually performs sparse SGD during backprop.
  // However, the weights must have an optimizer to prevent the model
  // from optimizing out the layer during the backprop.
  /// @todo Sparse optimizers
  this->get_data_type_weights(0).set_optimizer(
    make_unique<sgd<TensorDataType>>(0.));

  // Setup embedding weights
  embeddings.setup();

  // Reset SHMEM buffers
  if (m_embeddings_buffer != nullptr) {
    shmem_free(m_embeddings_buffer);
  }
  if (m_embeddings_grad_buffer != nullptr) {
    shmem_free(m_embeddings_grad_buffer);
  }
  if (m_requests_buffer != nullptr) {
    shmem_free(m_requests_buffer);
  }

  // Attach embeddings to SHMEM buffer
  auto& embeddings_mat = embeddings.get_values();
  m_embeddings_buffer = attach_dist_matrix_to_shmem_buffer(
    embeddings_mat,
    embeddings.get_matrix_height(),
    embeddings.get_matrix_width());

  // Construct gradient w.r.t. embeddings
  {
    auto dist = this->get_activations().DistData();
    dist.colDist = El::STAR;
    dist.rowDist = El::STAR;
    m_embeddings_grad.reset(
      El::AbstractDistMatrix<TensorDataType>::Instantiate(dist));
    m_embeddings_grad_buffer = nullptr;
    m_embeddings_grad_buffer_size = 0;
  }

#endif // LBANN_HAS_SHMEM
}

// =============================================
// Forward prop
// =============================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
void dist_embedding_layer<TensorDataType,Layout,Device>::fp_compute() {
#ifndef LBANN_HAS_SHMEM
  LBANN_ERROR(
    "dist_embedding_layer with ",
    "(TensorDataType=",TypeName<TensorDataType>(),", ",
    "Layout=",to_string(Layout),", ",
    "Device=",to_string(Device),") ",
    "requires SHMEM, but LBANN has not been built with SHMEM");
  return;
#else // LBANN_HAS_SHMEM
  using LocalMat = El::Matrix<TensorDataType, Device>;

  // Local data
  const auto& embeddings = this->get_data_type_weights(0).get_values();
  const auto& input = this->get_prev_activations();
  const auto& local_input = dynamic_cast<const LocalMat&>(input.LockedMatrix());
  auto& local_output = dynamic_cast<LocalMat&>(this->get_local_activations());
  const size_t input_size = this->get_input_size();
  const size_t mini_batch_size = input.Width();
  const size_t local_mini_batch_size = local_input.Width();
  const size_t rank = this->get_comm()->get_rank_in_trainer();

  // Initialize workspace for shmem_put requests
  if (m_requests_buffer_size < input_size * mini_batch_size) {
    m_requests_buffer_size = input_size * mini_batch_size;
    m_requests_buffer = reinterpret_cast<shmem_put_request*>(
      shmem_realloc(
        m_requests_buffer,
        m_requests_buffer_size*sizeof(shmem_put_request)
        )
      );
  }
  std::fill(
    m_requests_buffer,
    m_requests_buffer+m_requests_buffer_size,
    shmem_put_request());
  shmem_barrier_all();

  // Get embedding vectors from remote processes
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (size_t j=0; j<local_mini_batch_size; ++j) {
    for (size_t i=0; i<input_size; ++i) {
      const auto& global_index = static_cast<size_t>(std::floor(local_input(i,j)));
      const auto& global_j = input.GlobalCol(j);

      // Figure out which rank owns embedding vector
      auto& req = m_requests_buffer[i + global_j*input_size];
      req.source_rank = embeddings.Owner(0, global_index);
      req.source_index = embeddings.LocalCol(global_index, req.source_rank);
      req.target_rank = rank;
      req.target_index = i + j*input_size;
      req.is_active = true;
      shmem_putmem(
        &req,
        &req,
        sizeof(shmem_put_request),
        req.source_rank);

      // Get embedding vector from owner
      shmem_getmem_nbi(
        &local_output(i*m_embedding_dim, j),
        embeddings.LockedBuffer() + req.source_index*m_embedding_dim,
        m_embedding_dim*sizeof(TensorDataType),
        req.source_rank);

    }
  }
  shmem_quiet();

#endif // LBANN_HAS_SHMEM
}

// =============================================
// Backprop
// =============================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
void dist_embedding_layer<TensorDataType,Layout,Device>::bp_compute() {
#ifndef LBANN_HAS_SHMEM
  LBANN_ERROR(
    "dist_embedding_layer with ",
    "(TensorDataType=",TypeName<TensorDataType>(),", ",
    "Layout=",to_string(Layout),", ",
    "Device=",to_string(Device),") ",
    "requires SHMEM, but LBANN has not been built with SHMEM");
  return;
#else // LBANN_HAS_SHMEM
  using LocalMat = El::Matrix<TensorDataType, Device>;

  // Local data
  auto& embeddings = this->get_data_type_weights(0).get_values();
  auto& embeddings_grad = dynamic_cast<El::ElementalMatrix<TensorDataType>&>(*m_embeddings_grad);
  const auto& input = this->get_prev_activations();
  const auto& local_input = dynamic_cast<const LocalMat&>(input.LockedMatrix());
  const auto& local_output_grad = dynamic_cast<const LocalMat&>(this->get_local_prev_error_signals());
  const size_t input_size = this->get_input_size();
  const size_t output_size = this->get_output_size();
  const size_t mini_batch_size = input.Width();
  const size_t local_mini_batch_size = local_input.Width();
  const size_t rank = this->get_comm()->get_rank_in_trainer();

  // Initialize workspace for gradient w.r.t. embeddings
  const auto& dist = embeddings.DistData();
  embeddings_grad.Empty();
  if (m_embeddings_grad_buffer_size < output_size * mini_batch_size) {
    m_embeddings_grad_buffer_size = output_size * mini_batch_size;
    m_embeddings_grad_buffer = reinterpret_cast<TensorDataType*>(
      shmem_realloc(
        m_embeddings_grad_buffer,
        m_embeddings_grad_buffer_size*sizeof(TensorDataType)
        )
      );
  }
  embeddings_grad.Attach(
    output_size, mini_batch_size,
    *dist.grid, dist.colAlign, dist.rowAlign,
    m_embeddings_grad_buffer, output_size, dist.root);

  // Send gradients w.r.t. embedding vectors to remote processes
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (size_t j=0; j<local_mini_batch_size; ++j) {
    for (size_t i=0; i<input_size; ++i) {
      const auto& global_j = input.GlobalCol(j);
      auto& req = m_requests_buffer[i + global_j*input_size];
      shmem_putmem(
        embeddings_grad.Buffer(i*m_embedding_dim, global_j),
        local_output_grad.LockedBuffer(i*m_embedding_dim, j),
        m_embedding_dim*sizeof(TensorDataType),
        req.source_rank);
    }
  }
  shmem_barrier_all();

#ifdef LBANN_DIST_EMBEDDING_SPARSE_SGD

  // Stochastic gradient descent
  /// @todo Implement
  // El::Axpy(-m_learning_rate, embeddings_grad, embeddings);

#else // LBANN_DIST_EMBEDDING_SPARSE_SGD

  // Send gradient to optimizer
  auto& opt = *this->get_data_type_weights(0).get_optimizer();
  std::unique_ptr<El::AbstractDistMatrix<TensorDataType>> embeddings_grad_full(
    embeddings.Construct(embeddings.Grid(), embeddings.Root()));
  embeddings_grad_full->AlignWith(embeddings);
  El::Zeros(*embeddings_grad_full, embeddings.Height(), embeddings.Width());
  LocalMat embeddings_grad_full_v, embeddings_grad_sparse_v;
  for (size_t global_j=0; global_j<mini_batch_size; ++global_j) {
    for (size_t i=0; i<input_size; ++i) {
      const auto& req = m_requests_buffer[i + global_j*input_size];
      if (req.is_active && req.source_rank == rank) {
        El::LockedView(
          embeddings_grad_sparse_v,
          embeddings_grad.LockedMatrix(),
          El::IR(i*m_embedding_dim, (i+1)*m_embedding_dim),
          El::IR(global_j));
        El::View(
          embeddings_grad_full_v,
          embeddings_grad_full->Matrix(),
          El::ALL, El::IR(req.source_index));
        El::Axpy(
          TensorDataType{1.},
          embeddings_grad_sparse_v,
          embeddings_grad_full_v);
      }
    }
  }
  opt.add_to_gradient(*embeddings_grad_full);

#endif // LBANN_DIST_EMBEDDING_SPARSE_SGD

#endif // LBANN_HAS_SHMEM
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
