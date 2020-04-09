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

#include "lbann/data_readers/data_reader_node2vec.hpp"
#ifdef LBANN_HAS_LARGESCALE_NODE2VEC
#include "lbann/utils/memory.hpp"
#include <dist/node2vec_rw/node2vec_rw.hpp>
#include <havoqgt/distributed_db.hpp>

namespace lbann {

namespace node2vec_reader_impl {

  class DistributedDatabase : public ::havoqgt::distributed_db {
  public:
    template <typename... Args>
    DistributedDatabase(Args... args)
      : ::havoqgt::distributed_db(args...) {}
  };

  class RandomWalker : public ::node2vec_rw::node2vec_rw<> {
  public:
    template <typename... Args>
    RandomWalker(Args... args)
      : ::node2vec_rw::node2vec_rw<>(args...) {}
  };

  class EdgeWeightData : public RandomWalker::edge_weight_data_type {
  public:
    template <typename... Args>
    EdgeWeightData(Args... args)
      : RandomWalker::edge_weight_data_type(args...) {}
  };

} // namespace node2vec_reader_impl

namespace {
  using node2vec_reader_impl::DistributedDatabase;
  using node2vec_reader_impl::RandomWalker;
  using node2vec_reader_impl::EdgeWeightData;
  using Graph = RandomWalker::graph_type;
} // namespace <anon>

node2vec_reader::node2vec_reader(
  std::string graph_file,
  std::string backup_file,
  size_t walk_length,
  double return_param,
  double inout_param)
  : generic_data_reader(true),
    m_graph_file(std::move(graph_file)),
    m_backup_file(std::move(backup_file)),
    m_walk_length{walk_length},
    m_return_param{return_param},
    m_inout_param{inout_param}
{}

node2vec_reader::~node2vec_reader() {
  // Deallocate objects in right order
  m_random_walker.reset();
  m_edge_weight_data.reset();
  m_distributed_database.reset();
}

node2vec_reader* node2vec_reader::copy() const {
  LBANN_ERROR("can not copy node2vec_reader");
}

std::string node2vec_reader::get_type() const {
  return "node2vec_reader";
}

const std::vector<int> node2vec_reader::get_data_dims() const {
  std::vector<int> dims;
  /// @todo Include random samples and walk windows
  dims.push_back(static_cast<int>(m_walk_length));
  return dims;
}
int node2vec_reader::get_num_labels() const {
  return 1;
}
int node2vec_reader::get_linearized_data_size() const {
  const auto& dims = get_data_dims();
  return std::accumulate(dims.begin(), dims.end(), 1,
                         std::multiplies<int>());
}
int node2vec_reader::get_linearized_label_size() const {
  return get_num_labels();
}

bool node2vec_reader::fetch_datum(CPUMat& Y, int data_id, int col) {
  /// @todo Implement
  return true;
}

bool node2vec_reader::fetch_label(CPUMat& Y, int data_id, int col) {
  return true;
}

void node2vec_reader::setup(int num_io_threads,
                            observer_ptr<thread_pool> io_thread_pool) {
  generic_data_reader::setup(num_io_threads, io_thread_pool);

  // Copy backup file if needed
  if (!m_backup_file.empty()) {
    ::havoqgt::distributed_db::transfer(
      m_backup_file.c_str(),
      m_graph_file.c_str());
  }

  // Load graph data
  m_distributed_database = make_unique<DistributedDatabase>(
    ::havoqgt::db_open(),
    m_graph_file.c_str());
  auto* graph = m_distributed_database->get_segment_manager()->find<Graph>("graph_obj").first;

  // Load edge data
  m_edge_weight_data.reset();
  auto* edge_weight_data = m_distributed_database->get_segment_manager()->find<EdgeWeightData>("graph_edge_weight_data_obj").first;
  if (edge_weight_data == nullptr) {
    m_edge_weight_data = make_unique<EdgeWeightData>(*graph);
    m_edge_weight_data->reset(1.0);
    edge_weight_data = m_edge_weight_data.get();
  }
  MPI_Barrier(MPI_COMM_WORLD); /// @todo Use lbann_comm

  // Construct random walker
  bool small_edge_weight_variance = false;
  MPI_Comm comm = MPI_COMM_WORLD; /// @todo Use lbann_comm
  m_random_walker = make_unique<RandomWalker>(
    *graph,
    *edge_weight_data,
    small_edge_weight_variance,
    m_walk_length,
    m_return_param,
    m_inout_param,
    comm);
  MPI_Barrier(MPI_COMM_WORLD); /// @todo Use lbann_comm

}

void node2vec_reader::load() {
  if (m_distributed_database == nullptr) {
    LBANN_ERROR(
      "\"",this->get_type(),"\" data reader ",
      "attempted to load data before setup");
  }
  auto& graph = *m_distributed_database->get_segment_manager()->find<Graph>("graph_obj").first;
  m_shuffled_indices.resize(graph.max_global_vertex_id());
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  resize_shuffled_indices();
  select_subset_of_data();
}

} // namespace lbann

#endif // LBANN_HAS_LARGESCALE_NODE2VEC
