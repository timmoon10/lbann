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
#include "lbann/utils/random.hpp"
#include <dist/node2vec_rw/node2vec_rw.hpp>
#include <havoqgt/distributed_db.hpp>
#include <havoqgt/delegate_partitioned_graph.hpp>

#include <algorithm>
#include <numeric>

namespace lbann {

namespace node2vec_reader_impl {

  class DistributedDatabase : public ::havoqgt::distributed_db {
  public:
    using BaseType = ::havoqgt::distributed_db;
    template <typename... Args>
    DistributedDatabase(Args&&... args)
      : BaseType(std::forward<Args>(args)...) {}
  };

  class RandomWalker : public ::node2vec_rw::node2vec_rw<> {
  public:
    using BaseType = ::node2vec_rw::node2vec_rw<>;
    template <typename... Args>
    RandomWalker(Args&&... args)
      : BaseType(std::forward<Args>(args)...) {}
  };

  class EdgeWeightData : public RandomWalker::edge_weight_data_type {
  public:
    using BaseType = RandomWalker::edge_weight_data_type;
    template <typename... Args>
    EdgeWeightData(Args&&... args)
      : BaseType(std::forward<Args>(args)...) {}
  };

} // namespace node2vec_reader_impl

namespace {
  using node2vec_reader_impl::DistributedDatabase;
  using node2vec_reader_impl::RandomWalker;
  using node2vec_reader_impl::EdgeWeightData;
  using Graph = RandomWalker::graph_type;
  using Vertex = RandomWalker::vertex_type;
} // namespace <anon>

node2vec_reader::node2vec_reader(
  std::string graph_file,
  size_t epoch_size,
  size_t walk_length,
  double return_param,
  double inout_param,
  size_t walk_context_size,
  size_t num_negative_samples)
  : generic_data_reader(true),
    m_graph_file(std::move(graph_file)),
    m_epoch_size{epoch_size},
    m_walk_length{walk_length},
    m_return_param{return_param},
    m_inout_param{inout_param},
    m_walk_context_size{walk_context_size},
    m_num_negative_samples{num_negative_samples} {
  if (m_walk_context_size > m_walk_length) {
    LBANN_ERROR("attempted to create node2vec data reader ",
                "with the walk context size (",m_walk_context_size,") ",
                "larger than the walk length (",m_walk_length,")");
  }
}

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
  dims.push_back(static_cast<int>(m_walk_context_size + m_num_negative_samples));
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

bool node2vec_reader::fetch_data_block(
  CPUMat& X,
  El::Int thread_id,
  El::Int mb_size,
  El::Matrix<El::Int>& indices_fetched) {
  if (thread_id != 0) { return true; }
  const size_t mb_size_ = mb_size;

  // Perform random walks
  const auto contexts_per_walk = m_walk_length - m_walk_context_size + 1;
  auto num_walks = (mb_size_ + contexts_per_walk - 1) / contexts_per_walk;
  if (m_walks_cache.size() < mb_size_) {
    num_walks = El::Max(num_walks, mb_size_ - m_walks_cache.size());
  }
  num_walks = El::Max(num_walks, 1);
  num_walks = El::Min(num_walks, mb_size_);
  auto walks = run_walker(num_walks);

  // Update cache of random walks
  const size_t max_cache_size = El::Max(m_walks_cache.size(), mb_size_);
  for (auto& walk : walks) {
    m_walks_cache.emplace_front(std::move(walk));
  }
  while (m_walks_cache.size() > max_cache_size) {
    m_walks_cache.pop_back();
  }

  // Recompute noise distribution if there are enough vertex visits
  if (m_total_visit_count > 2*m_noise_visit_count) {
    compute_noise_distribution();
  }

  // Populate output tensor
  /// @todo Parallelize
  for (size_t j=0; j<mb_size_; ++j) {

    // Context window in random walk
    const auto cache_pos
      = fast_rand_int(get_io_generator(),
                      contexts_per_walk*m_walks_cache.size());
    const auto& walk = m_walks_cache[cache_pos / contexts_per_walk];
    const auto offset = cache_pos % contexts_per_walk;
    const auto start_index = walk[offset];
    for (size_t i=0; i<m_walk_context_size; ++i) {
      X(i+m_num_negative_samples,j) = static_cast<float>(walk[i+offset]);
    }

    // Negative samples
    for (size_t i=0; i<m_num_negative_samples; ++i) {
      size_t global_index;
      do {
        const auto local_index = std::distance(
          m_local_vertex_noise_distribution.begin(),
          std::lower_bound(
            m_local_vertex_noise_distribution.begin(),
            m_local_vertex_noise_distribution.end(),
            random_uniform<double>(get_io_generator())));
        global_index = m_local_vertex_global_indices[local_index];
      } while (global_index == start_index);
      X(i,j) = static_cast<float>(global_index);
    }

  }

  return true;
}

bool node2vec_reader::fetch_label(CPUMat& Y, int data_id, int col) {
  return true;
}

void node2vec_reader::load() {
  auto& comm = *get_comm();

  // Load graph data
  m_distributed_database = make_unique<DistributedDatabase>(
    ::havoqgt::db_open(),
    m_graph_file.c_str());
  auto& graph = *m_distributed_database->get_segment_manager()->find<Graph>("graph_obj").first;

  // Load edge data
  m_edge_weight_data.reset();
  auto* edge_weight_data = m_distributed_database->get_segment_manager()->find<EdgeWeightData::BaseType>("graph_edge_data_obj").first;
  if (edge_weight_data == nullptr) {
    m_edge_weight_data = make_unique<EdgeWeightData>(graph);
    m_edge_weight_data->reset(1.0);
    edge_weight_data = m_edge_weight_data.get();
  }
  comm.trainer_barrier();

  // Construct random walker
  constexpr bool small_edge_weight_variance = false;
  constexpr bool verbose = false;
  m_random_walker = make_unique<RandomWalker>(
    graph,
    *edge_weight_data,
    small_edge_weight_variance,
    m_walk_length,
    m_return_param,
    m_inout_param,
    comm.get_trainer_comm().GetMPIComm(),
    verbose);
  comm.trainer_barrier();

  // Get local vertices
  // Note: Estimate frequency of vertex visits using the vertex
  // degree, plus 1 for Laplace smoothing.
  const size_t num_local_vertices = graph.num_local_vertices();
  if (num_local_vertices == 0) {
    LBANN_ERROR("node2vec data reader loaded a graph with no local vertices");
  }
  m_local_vertex_global_indices.clear();
  m_local_vertex_global_indices.reserve(num_local_vertices);
  m_local_vertex_local_indices.clear();
  m_local_vertex_local_indices.reserve(num_local_vertices);
  m_local_vertex_visit_counts.clear();
  m_local_vertex_visit_counts.reserve(num_local_vertices);
  for (auto iter = graph.vertices_begin();
       iter != graph.vertices_end();
       ++iter) {
    const auto& vertex = *iter;
    const auto& degree = graph.degree(vertex);
    const auto& global_index = graph.locator_to_label(vertex);
    const auto& local_index = m_local_vertex_global_indices.size();
    m_local_vertex_global_indices.push_back(global_index);
    m_local_vertex_local_indices[global_index] = local_index;
    m_local_vertex_visit_counts.push_back(degree+1);
  }

  // Compute noise distribution for negative sampling
  compute_noise_distribution();

  // Reset cache of random walks
  m_walks_cache.clear();

  // Construct list of indices
  m_shuffled_indices.resize(m_epoch_size);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  resize_shuffled_indices();
  select_subset_of_data();

}

std::vector<std::vector<size_t>> node2vec_reader::run_walker(size_t num_walks) {

  // HavoqGT graph
  const auto& graph = *m_distributed_database->get_segment_manager()->find<Graph>("graph_obj").first;
  const auto num_local_vertices = m_local_vertex_global_indices.size();

  // Randomly choose start vertices for random walks
  std::vector<Vertex> start_vertices;
  start_vertices.reserve(num_walks);
  for (size_t i=0; i<num_walks; ++i) {
    const auto& local_index = fast_rand_int(get_io_generator(),
                                            num_local_vertices);
    const auto& global_index = m_local_vertex_global_indices.at(local_index);
    start_vertices.push_back(graph.label_to_locator(global_index));
  }

  // Perform random walks
  const auto walks_vertices = m_random_walker->run_walker(start_vertices);

  // Convert walks to vertex indices
  std::vector<std::vector<size_t>> walks_indices;
  walks_indices.reserve(walks_vertices.size());
  for (const auto& walk_vertices : walks_vertices) {
    walks_indices.emplace_back();
    auto& walk_indices = walks_indices.back();
    walk_indices.reserve(walk_vertices.size());
    for (const auto& vertex : walk_vertices) {
      walk_indices.emplace_back(graph.locator_to_label(vertex));
    }
  }

  // Record visits to local vertices
  for (const auto& walk : walks_indices) {
    for (const auto& global_index : walk) {
      if (m_local_vertex_local_indices.count(global_index) != 0) {
        const auto& local_index = m_local_vertex_local_indices.at(global_index);
        ++m_local_vertex_visit_counts[local_index];
        ++m_total_visit_count;
      }
    }
  }

  return walks_indices;

}

/// @todo Parallelize
void node2vec_reader::compute_noise_distribution() {

  // Count number of times each local vertex has been visited
  // Note: Distribution is proportional to count^0.75
  const size_t num_local_vertices = m_local_vertex_global_indices.size();
  m_local_vertex_noise_distribution.resize(num_local_vertices);
  m_noise_visit_count = 0;
  for (size_t i=0; i<num_local_vertices; ++i) {
    const auto& count = m_local_vertex_visit_counts[i];
    m_local_vertex_noise_distribution[i] = std::pow(count, 0.75);
    m_noise_visit_count += count;
  }
  m_total_visit_count = m_noise_visit_count;

  // Compute CDF by computing cumsum and normalizing
  /// @todo If numerical error becomes a problem, use Kahan summation
  std::partial_sum(m_local_vertex_noise_distribution.begin(),
                   m_local_vertex_noise_distribution.end(),
                   m_local_vertex_noise_distribution.begin());
  const double scale = 1. / m_local_vertex_noise_distribution.back();
  std::transform(m_local_vertex_noise_distribution.begin(),
                 m_local_vertex_noise_distribution.end(),
                 m_local_vertex_noise_distribution.begin(),
                 [&scale](const double& x) -> double { return scale*x; });

}

} // namespace lbann

#endif // LBANN_HAS_LARGESCALE_NODE2VEC
