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

namespace lbann {

/** @brief Embedding layer with distributed weights.
 *
 *  @warning This is extremely experimental.
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class dist_embedding_layer : public data_type_layer<TensorDataType> {
//   static_assert(
//     false, /// @todo Support fp16
// #else
//     false,
// #endif // LBANN_HAS_GPU_FP16
//     "distributed embedding layer only supports half-precision datatype");
  static_assert(Layout == data_layout::DATA_PARALLEL,
                "distributed embedding layer only supports data parallel layout");
  static_assert(Device == El::Device::GPU,
                "distributed embedding layer only supports GPU");

public:

  dist_embedding_layer(
    lbann_comm* comm,
    size_t num_embeddings,
    size_t embedding_dim);

  dist_embedding_layer(const dist_embedding_layer& other);
  dist_embedding_layer& operator=(const dist_embedding_layer& other);
  ~dist_embedding_layer() = default;

  dist_embedding_layer* copy() const override;
  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;

  description get_description() const override;

protected:

  void setup_matrices(const El::Grid& grid) override;
  void setup_dims() override;
  void setup_data() override;

  void fp_compute() override;
  void bp_compute() override;

private:

  /** Size of dictionary of embeddings. */
  size_t m_num_embeddings;
  /** Size of embedding vectors. */
  size_t m_embedding_dim;

};

// Builder function
LBANN_DEFINE_LAYER_BUILDER(dist_embedding);

// Explicit template instantiation
#ifdef LBANN_HAS_GPU_FP16
extern template class dist_embedding_layer<
  fp16, data_layout::DATA_PARALLEL, El::Device::GPU>;
#endif // LBANN_HAS_GPU_FP16

} // namespace lbann

#endif // LBANN_LAYERS_MISC_DIST_EMBEDDING_HPP_INCLUDED
