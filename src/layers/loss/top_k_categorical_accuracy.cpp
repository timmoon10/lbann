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

#define LBANN_TOP_K_CATEGORICAL_ACCURACY_LAYER_INSTANTIATE
#include "lbann/layers/loss/top_k_categorical_accuracy.hpp"
#include <algorithm>
#include <limits>


namespace lbann {

namespace {

/** Sparse vector entry. */
template <typename TensorDataType>
struct entry {

  /** Vector entry value. */
  TensorDataType value = min_value;
  /** Vector entry index. */
  El::Int index = max_index;

  /** Minimum possible value. */
  static constexpr TensorDataType min_value = -std::numeric_limits<TensorDataType>::infinity();
  /** Maximum possible index. */
  static constexpr El::Int max_index = std::numeric_limits<El::Int>::max();

  /** Comparison operation to sort vector entries.
   *  Entries are sorted by value in decreasing order, with ties
   *  broken in favor of entries with smaller indices.
   */
  static bool compare(const entry<TensorDataType>& a, const entry<TensorDataType>& b) {
    return a.value > b.value || (a.value == b.value && a.index < b.index);
  }

};

/** CPU implementation of top-k categorical accuracy layer forward prop. */
template <typename TensorDataType>
void fp_cpu(lbann_comm& comm,
            El::Int k,
            const El::AbstractDistMatrix<TensorDataType>& predictions,
            const El::AbstractDistMatrix<TensorDataType>& labels,
            El::AbstractDistMatrix<TensorDataType>& loss) {

  // Local matrices
  const auto& local_predictions = predictions.LockedMatrix();
  const auto& local_labels = labels.LockedMatrix();
  auto& local_loss = loss.Matrix();
  const El::Int height = predictions.Height();
  const El::Int local_height = local_predictions.Height();
  const El::Int local_width = local_predictions.Width();

  // Trivial cases
  if (k < 1) {
    El::Zero(loss);
    return;
  } else if (k >= height) {
    El::Fill(loss, El::TypeTraits<TensorDataType>::One());
    return;
  } else if (local_width < 1) {
    return;
  }

  // Column communicator
  auto&& col_comm = predictions.ColComm();
  const auto& col_comm_rank = El::mpi::Rank(col_comm);
  const auto& col_comm_size = El::mpi::Size(col_comm);
  const auto& col_comm_root = loss.RowOwner(0);

  // Get label indices
  // Note: This may have race conditions if columns of labels matrix
  // are not one-hot vectors.
  std::vector<El::Int> label_indices(local_width, height);
  Al::request req;
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int col = 0; col < local_width; ++col) {
    for (El::Int row = 0; row < local_height; ++row) {
      if (local_labels(row, col) > El::TypeTraits<TensorDataType>::Zero()) {
        label_indices[col] = labels.GlobalRow(row);
      }
    }
  }
  comm.nb_allreduce(label_indices.data(),
                    label_indices.size(),
                    col_comm,
                    req,
                    El::mpi::MIN);

  // Find top-k entries in each column of local prediction matrix
  std::vector<entry<TensorDataType>> top_entries(local_width * k);
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    std::vector<entry<TensorDataType>> local_entries(std::max(local_height, k));
    for (El::Int row = 0; row < local_height; ++row) {
      local_entries[row].value = local_predictions(row, col);
      local_entries[row].index = predictions.GlobalRow(row);
    }
    std::partial_sort_copy(local_entries.begin(),
                           local_entries.end(),
                           &top_entries[col*k],
                           &top_entries[col*k] + k,
                           entry<TensorDataType>::compare);
  }

  // Find top-k entries in each column of global prediction matrix
  if (col_comm_size > 1) {
    if (col_comm_rank != col_comm_root) {
      comm.gather(reinterpret_cast<El::byte*>(top_entries.data()),
                  top_entries.size() * sizeof(entry<TensorDataType>),
                  col_comm_root,
                  col_comm, El::SyncInfo<El::Device::CPU>{});
    } else {
      std::vector<entry<TensorDataType>> global_top_entries(col_comm_size * local_width * k);
      comm.gather(reinterpret_cast<El::byte*>(top_entries.data()),
                  top_entries.size() * sizeof(entry<TensorDataType>),
                  reinterpret_cast<El::byte*>(global_top_entries.data()),
                  col_comm, El::SyncInfo<El::Device::CPU>{});
      LBANN_OMP_PARALLEL_FOR
      for (El::Int col = 0; col < local_width; ++col) {
        std::vector<entry<TensorDataType>> col_entries(col_comm_size * k);
        for (El::Int rank = 0; rank < col_comm_size; ++rank) {
          const auto* start = &global_top_entries[rank*local_width*k+col*k];
          std::copy(start, start + k, &col_entries[rank*k]);
        }
        std::partial_sort_copy(col_entries.begin(),
                               col_entries.end(),
                               &top_entries[col*k],
                               &top_entries[col*k] + k,
                               entry<TensorDataType>::compare);
      }
    }

  }

  // Compute categorical accuracy
  El::Zero(loss);
  comm.wait(req);
  if (col_comm_rank == col_comm_root) {
    LBANN_OMP_PARALLEL_FOR_COLLAPSE2
    for (El::Int col = 0; col < local_width; ++col) {
      for (El::Int i = 0; i < k; ++i) {
        const auto& label_index = label_indices[col];
        if (top_entries[col*k+i].index == label_index
            && label_index < height) {
          local_loss(0, col) = El::TypeTraits<TensorDataType>::One();
        }
      }
    }
  }

}

} // namespace

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void top_k_categorical_accuracy_layer<TensorDataType, T_layout, Dev>::fp_compute() {
  fp_cpu(*this->get_comm(),
         this->m_k,
         this->get_prev_activations(0),
         this->get_prev_activations(1),
         this->get_activations());
}

#define PROTO(T)                                      \
  template class top_k_categorical_accuracy_layer<    \
    T, data_layout::DATA_PARALLEL, El::Device::CPU>;  \
  template class top_k_categorical_accuracy_layer<    \
    T, data_layout::MODEL_PARALLEL, El::Device::CPU>

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
