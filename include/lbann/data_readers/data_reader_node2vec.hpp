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

#ifndef LBANN_DATA_READERS_NODE2VEC_HPP_INCLUDED
#define LBANN_DATA_READERS_NODE2VEC_HPP_INCLUDED

#include "data_reader.hpp"
#ifdef LBANN_HAS_LARGESCALE_NODE2VEC

namespace lbann {

// Note (tym 4/8/20): Including largescale_node2vec in this header
// causes multiple definitions (I suspect it instantiates an object
// somewhere). However, node2vec_reader needs to store
// largescale_node2vec classes in unique_ptrs. To get around this, we
// implement derived classes in the source file and forward declare
// them in this header.
namespace node2vec_reader_impl {
  class DistributedDatabase;
  class EdgeWeightData;
  class RandomWalker;
} // namespace node2vec_reader_impl

class node2vec_reader : public generic_data_reader {
public:

  node2vec_reader(
    std::string graph_file,
    std::string backup_file,
    size_t walk_length,
    double return_param,
    double inout_param,
    size_t num_negative_samples);
  node2vec_reader(const node2vec_reader&) = delete;
  node2vec_reader& operator=(const node2vec_reader&) = delete;
  ~node2vec_reader() override;
  node2vec_reader* copy() const override;

  std::string get_type() const override;

  const std::vector<int> get_data_dims() const override;
  int get_num_labels() const override;
  int get_linearized_data_size() const override;
  int get_linearized_label_size() const override;

  void load() override;

protected:
  bool fetch_data_block(
    CPUMat& X,
    El::Int thread_id,
    El::Int mb_size,
    El::Matrix<El::Int>& indices_fetched) override;
  bool fetch_label(CPUMat& Y, int data_id, int mb_idx) override;

private:

  std::unique_ptr<node2vec_reader_impl::DistributedDatabase> m_distributed_database;
  std::unique_ptr<node2vec_reader_impl::EdgeWeightData> m_edge_weight_data;
  std::unique_ptr<node2vec_reader_impl::RandomWalker> m_random_walker;
  std::vector<size_t> m_local_vertices;

  std::string m_graph_file;
  std::string m_backup_file;
  size_t m_walk_length;
  /** @brief node2vec p parameter. */
  double m_return_param;
  /** @brief node2vec q parameter. */
  double m_inout_param;
  size_t m_num_negative_samples;

};

} // namespace lbann

#endif // LBANN_HAS_LARGESCALE_NODE2VEC
#endif // LBANN_DATA_READERS_NODE2VEC_HPP_INCLUDED
