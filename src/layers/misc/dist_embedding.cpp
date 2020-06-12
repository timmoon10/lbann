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

#include <layers.pb.h>

#ifdef LBANN_HAS_SHMEM
#include <shmem.h>
#endif // LBANN_HAS_SHMEM

namespace lbann {

namespace {

/** Value to set SHMEM flags. */
constexpr long flag_val = 1;

} // namespace <anon>

// =============================================
// Life cycle and setup
// =============================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
dist_embedding_layer<TensorDataType,Layout,Device>::~dist_embedding_layer()
{
#ifdef LBANN_HAS_SHMEM
  shmem_free(m_embeddings_buffer);
  shmem_free(m_workspace_buffer);
  shmem_free(m_metadata_buffer);
#endif // LBANN_HAS_SHMEM
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void dist_embedding_layer<TensorDataType,Layout,Device>::attach_embeddings_to_shmem_buffer() {
#ifndef LBANN_HAS_SHMEM
  LBANN_ERROR(
    "dist_embedding_layer with ",
    "(TensorDataType=",TypeName<TensorDataType>(),", ",
    "Layout=",to_string(Layout),", ",
    "Device=",to_string(Device),") ",
    "requires SHMEM, but LBANN has not been built with SHMEM");
  return;
#else
  if (m_embeddings_buffer != nullptr || m_embeddings_buffer_size != 0) {
    LBANN_ERROR("attempted to attach embedding matrix ",
                "to OpenSHMEM buffer multiple times");
  }

  // Embedding weights matrix
  auto& embeddings = this->get_data_type_weights(0).get_values();
  const auto dist = embeddings.DistData();
  if (dist.device != El::Device::CPU) {
    LBANN_ERROR("attempted to attach non-CPU matrix to OpenSHMEM buffer");
  }
  if (shmem_addr_accessible(embeddings.LockedBuffer(), shmem_my_pe())) {
    return;
  }

  // Calculate size of SHMEM buffer
  const auto col_comm_size = El::mpi::Size(embeddings.ColComm());
  const auto row_comm_size = El::mpi::Size(embeddings.RowComm());
  const auto height = embeddings.Height();
  const auto width = embeddings.Width();
  const auto local_height = (height + col_comm_size - 1) / col_comm_size;
  const auto local_width = (width + row_comm_size - 1) / row_comm_size;
  m_embeddings_buffer_size = local_height * local_width * sizeof(TensorDataType);
  if (m_embeddings_buffer_size == 0) {
    return;
  }

  // Allocate SHMEM buffer
  m_embeddings_buffer = reinterpret_cast<TensorDataType*>(
    shmem_malloc(m_embeddings_buffer_size));
  if (m_embeddings_buffer == nullptr) {
    LBANN_ERROR("failed to allocate OpenSHMEM buffer");
  }

  // Attach matrix to SHMEM buffer
  std::unique_ptr<El::AbstractDistMatrix<TensorDataType>> orig_mat(
    embeddings.Construct(embeddings.Grid(), embeddings.Root()));
  *orig_mat = std::move(embeddings);
  embeddings.Empty();
  embeddings.AlignWith(dist);
  dynamic_cast<El::ElementalMatrix<TensorDataType>&>(embeddings).Attach(
    height, width,
    *dist.grid, dist.colAlign, dist.rowAlign,
    m_embeddings_buffer, local_height, dist.root);
  El::Copy(*orig_mat, embeddings);

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

  // Data matrices
  const auto& embeddings = this->get_data_type_weights(0).get_values();
  const auto& input = this->get_prev_activations();
  const auto& local_input = dynamic_cast<const LocalMat&>(input.LockedMatrix());
  auto& local_output = dynamic_cast<LocalMat&>(this->get_local_activations());

  // Dimensions
  const size_t input_size = this->get_input_size();
  const size_t output_size = this->get_output_size();
  const size_t mini_batch_size = input.Width();
  const size_t local_mini_batch_size = local_input.Width();

  // Barrier to handle gradient checking
  /// @todo Think of a way to avoid this synchronization
  if (m_barrier_in_forward_prop) {
    shmem_barrier_all();
  }

  // Synchronize non-blocking barrier
  // Note: Make sure embeddings are up-to-date and SHMEM workspaces
  // are safe to reset.
  auto& comm = *this->get_comm();
  comm.wait(m_nb_barrier_request);

  // Initialize SHMEM buffer for communicating embedding vectors
  if (m_workspace_buffer_size < output_size * mini_batch_size) {
    m_workspace_buffer_size = output_size * mini_batch_size;
    m_workspace_buffer = reinterpret_cast<TensorDataType*>(
      shmem_realloc(
        m_workspace_buffer,
        m_workspace_buffer_size*sizeof(MetadataType)
        )
      );
  }
  LocalMat workspace(
    m_embedding_dim,
    input_size * mini_batch_size,
    m_workspace_buffer,
    m_embedding_dim);

  // Initialize SHMEM buffer for embedding vector metadata
  if (m_metadata_buffer_size < input_size * mini_batch_size) {
    m_metadata_buffer_size = input_size * mini_batch_size;
    m_metadata_buffer = reinterpret_cast<MetadataType*>(
      shmem_realloc(
        m_metadata_buffer,
        m_metadata_buffer_size*sizeof(MetadataType))
      );
  }
  std::fill(
    m_metadata_buffer,
    m_metadata_buffer+m_metadata_buffer_size,
    MetadataType());

  // Get embedding vectors from owner processes
  const size_t rank = comm.get_rank_in_trainer();
  for (size_t j=0; j<local_mini_batch_size; ++j) {
    for (size_t i=0; i<input_size; ++i) {
      const auto& global_index = static_cast<size_t>(std::floor(local_input(i,j)));
      const auto& global_j = input.GlobalCol(j);

      // Figure out which process owns embedding vector
      auto& m = m_metadata_buffer[i + global_j*input_size];
      m.source_rank = embeddings.Owner(0, global_index);
      m.source_index = embeddings.LocalCol(global_index, m.source_rank);
      m.target_rank = rank;
      m.target_index = i + global_j*input_size;
      m.is_active = true;

      // Get embedding vector from owner process
      shmem_getmem_nbi(
        workspace.Buffer(0, m.target_index),
        embeddings.LockedBuffer(0, m.source_index),
        m_embedding_dim*sizeof(TensorDataType),
        m.source_rank);

    }
  }
  shmem_quiet();

  // Copy embedding vectors from workspace to output tensor
  for (size_t j=0; j<local_mini_batch_size; ++j) {
    for (size_t i=0; i<input_size; ++i) {
      const auto& global_j = input.GlobalCol(j);
      const auto* x = workspace.LockedBuffer(0, i + global_j*input_size);
      auto* y = local_output.Buffer(i*m_embedding_dim, j);
      std::copy(x, x+m_embedding_dim, y);
    }
  }

  // Non-blocking barrier
  // Note: SHMEM workspaces are ready to recieve gradients.
  nb_barrier(comm, comm.get_trainer_comm(), m_nb_barrier_request);

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

  // Data matrices
  const auto& input = this->get_prev_activations();
  const auto& local_output_grad = dynamic_cast<const LocalMat&>(this->get_local_prev_error_signals());

  // Dimensions
  const size_t input_size = this->get_input_size();
  const size_t mini_batch_size = input.Width();
  const size_t local_mini_batch_size = local_output_grad.Width();

  // Synchronize non-blocking barrier
  // Note: Make sure SHMEM workspaces are ready to recieve gradients.
  auto& comm = *this->get_comm();
  comm.wait(m_nb_barrier_request);

  // Initialize SHMEM buffer for gradient w.r.t. embeddings
  LocalMat workspace(
    m_embedding_dim,
    input_size * mini_batch_size,
    m_workspace_buffer,
    m_embedding_dim);

  // Send gradients to owner processes
  for (size_t j=0; j<local_mini_batch_size; ++j) {
    for (size_t i=0; i<input_size; ++i) {
      const auto& global_j = input.GlobalCol(j);
      auto& m = m_metadata_buffer[i + global_j*input_size];
      shmem_putmem_nbi(
        workspace.Buffer(0, i+global_j*input_size),
        local_output_grad.LockedBuffer(i*m_embedding_dim, j),
        m_embedding_dim*sizeof(TensorDataType),
        m.source_rank);
      shmem_putmem_nbi(
        &m,
        &m,
        sizeof(MetadataType),
        m.source_rank);
    }
  }
  shmem_quiet();

  // Non-blocking barrier
  // Note: Gradients have been sent.
  nb_barrier(comm, comm.get_trainer_comm(), m_nb_barrier_request);

  // Use dense optimizer if needed
  if (!m_sparse_sgd) {

    // Create buffer for dense gradients
    auto& embeddings = this->get_data_type_weights(0).get_values();
    std::unique_ptr<El::AbstractDistMatrix<TensorDataType>> embeddings_grad(
      embeddings.Construct(embeddings.Grid(), embeddings.Root()));
    embeddings_grad->AlignWith(embeddings);
    El::Zeros(*embeddings_grad, embeddings.Height(), embeddings.Width());
    auto& local_embeddings_grad = dynamic_cast<LocalMat&>(embeddings_grad->Matrix());

    // Apply SGD step to convert sparse gradients to dense gradients
    apply_sparse_sgd_step(
      input_size * mini_batch_size,
      local_embeddings_grad);

    // Send dense gradients to dense optimizer
    auto&& opt = this->get_data_type_weights(0).get_optimizer();
    if (opt != nullptr) {
      opt->add_to_gradient(*embeddings_grad);
    }

  }

#endif // LBANN_HAS_SHMEM
}

// =============================================
// Sparse SGD
// =============================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
void dist_embedding_layer<TensorDataType,Layout,Device>::apply_sparse_sgd_step(
  size_t num_gradients,
  LocalMat& local_embeddings) {
#ifndef LBANN_HAS_SHMEM
  LBANN_ERROR(
    "dist_embedding_layer with ",
    "(TensorDataType=",TypeName<TensorDataType>(),", ",
    "Layout=",to_string(Layout),", ",
    "Device=",to_string(Device),") ",
    "requires SHMEM, but LBANN has not been built with SHMEM");
  return;
#else // LBANN_HAS_SHMEM

  // Synchronize non-blocking barrier
  // Note: Make sure gradients have been received.
  auto& comm = *this->get_comm();
  comm.wait(m_nb_barrier_request);

  // Initialize SHMEM buffer for gradient w.r.t. embeddings
  LocalMat local_embeddings_grad(
    m_embedding_dim,
    num_gradients,
    m_workspace_buffer,
    m_embedding_dim);

  // Sparse SGD on local embeddings
  const size_t rank = comm.get_rank_in_trainer();
  const size_t num_omp_threads = omp_get_num_threads();
  const size_t embeddings_per_thread
    = (local_embeddings.Width() + num_omp_threads - 1) / num_omp_threads;
  LBANN_OMP_PARALLEL_FOR
  for (size_t thread = 0; thread < num_omp_threads; ++thread) {
    const size_t index_start = thread * embeddings_per_thread;
    const size_t index_end = (thread+1) * embeddings_per_thread;
    for (size_t i=0; i<num_gradients; ++i) {
      const auto& m = m_metadata_buffer[i];
      if (m.is_active
          && m.source_rank == rank
          && index_start <= m.source_index
          && m.source_index < index_end) {
        const auto* dw = local_embeddings_grad.LockedBuffer(0, m.target_index);
        auto* w = local_embeddings.Buffer(0, m.source_index);
        EL_SIMD
        for (size_t k = 0; k < m_embedding_dim; ++k) {
          w[k] -= m_learning_rate * dw[k];
        }
      }
    }
  }

#endif // LBANN_HAS_SHMEM
}

// =============================================
// Builder function
// =============================================

namespace
{

template <typename TensorDataType, data_layout Layout, El::Device Device>
struct Builder
{

  template <typename... Args>
  static std::unique_ptr<Layer> Build(Args&&...)
  {
    LBANN_ERROR(
      "Attempted to construct dist_embedding_layer ",
      "with invalid parameters ",
      "(TensorDataType=",TypeName<TensorDataType>(),", ",
      "Layout=",to_string(Layout),", ",
      "Device=",to_string(Device),")");
    return nullptr;
  }

};

template <El::Device Device>
struct Builder<float,data_layout::DATA_PARALLEL,Device>
{

  template <typename... Args>
  static std::unique_ptr<Layer> Build(Args&&... args)
  {
    using LayerType = dist_embedding_layer<float,data_layout::DATA_PARALLEL,Device>;
    return make_unique<LayerType>(std::forward<Args>(args)...);
  }

};

} // namespace <anon>

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer> build_dist_embedding_layer_from_pbuf(
  lbann_comm* comm,
  const lbann_data::Layer& proto_layer)
{
  using BuilderType = Builder<TensorDataType, Layout, Device>;
  LBANN_ASSERT_MSG_HAS_FIELD(proto_layer, dist_embedding);
  const auto& params = proto_layer.dist_embedding();
  return BuilderType::Build(
    comm,
    params.num_embeddings(),
    params.embedding_dim(),
    params.sparse_sgd(),
    params.learning_rate(),
    params.barrier_in_forward_prop());
}

// =============================================
// Explicit template instantiation
// =============================================

/// @todo fp16
template class dist_embedding_layer<
  float, data_layout::DATA_PARALLEL, El::Device::CPU>;
extern template class dist_embedding_layer<
  float, data_layout::DATA_PARALLEL, El::Device::GPU>;

#define PROTO_DEVICE(T, Device)                         \
  LBANN_LAYER_BUILDER_ETI(dist_embedding, T, Device)
#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
