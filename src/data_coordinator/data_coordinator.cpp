////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#include <lbann/data_coordinator/data_coordinator.hpp>
#include <lbann/trainers/trainer.hpp>

namespace lbann {

void data_coordinator::setup(int max_mini_batch_size) {

  /// @todo BVE FIXME the list of execution modes should not include
  // ones will null data readers.  Fix this in next PR.
  // Setup data readers
  for(auto&& dr: m_data_readers) {
    if (!dr.second) continue;
    dr.second->setup(m_trainer->get_io_thread_pool().get_num_threads(),
                     &(m_trainer->get_io_thread_pool()));
    dr.second->set_rank(m_comm->get_rank_in_trainer());
  }

  /** Calculate how many iterations are required for training, testing,
   *  and validation given a specified mini-batch size.
   */
  for(auto&& dr: m_data_readers) {
    if (!dr.second) continue;
    calculate_num_iterations_per_epoch(max_mini_batch_size, dr.second);
  }

  options *opts = options::get();
  if (opts->get_bool("use_data_store") || opts->get_bool("preload_data_store") || opts->get_bool("data_store_cache") || opts->has_string("data_store_spill")) {
    bool master = m_comm->am_world_master();
    if (master) {
      std::cout << "\nUSING DATA STORE!\n\n";
    }
    for (auto&& r : m_data_readers) {
      if (!r.second) continue;
      r.second->setup_data_store(m_trainer->get_max_mini_batch_size());
    }
  }
}

void data_coordinator::calculate_num_iterations_per_epoch(int max_mini_batch_size, generic_data_reader *data_reader) {
  if(data_reader == nullptr) { return; }
  // If the data reader does not have any data bail out (e.g. unused validation reader)
  if(data_reader->get_num_data() == 0) { return; }

  if(max_mini_batch_size > data_reader->get_num_data()) {
    max_mini_batch_size = data_reader->get_num_data();
  }

  /// Check to make sure that there is enough data for all of the parallel readers
  int num_parallel_readers_per_model = compute_max_num_parallel_readers(data_reader->get_num_data(), max_mini_batch_size, this->m_comm->get_procs_per_trainer());
  data_reader->set_num_parallel_readers(num_parallel_readers_per_model);
  if(num_parallel_readers_per_model == 0
     || (num_parallel_readers_per_model != this->m_comm->get_procs_per_trainer() && num_parallel_readers_per_model != max_mini_batch_size)) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: generic_data_distribution: number of parallel readers is zero");
  }

  /// Set the basic parameters for stride and offset of the data reader
  int batch_stride = max_mini_batch_size;
  int base_offset  = this->m_comm->get_rank_in_trainer();
  /// Set mini-batch size and stride
  data_reader->set_mini_batch_size(max_mini_batch_size);
  data_reader->set_stride_to_next_mini_batch(batch_stride);
  data_reader->set_sample_stride(num_parallel_readers_per_model);
  data_reader->set_iteration_stride(1);
  /// Set data reader base offset and model offset
  data_reader->set_base_offset(base_offset);
  data_reader->set_model_offset(0);
  data_reader->set_initial_position();

  /// By default each data reader will plan to process the entire data set
  int num_iterations_per_epoch = ceil((float) data_reader->get_num_data() / (float) max_mini_batch_size);
  int last_mini_batch_size = data_reader->get_num_data() % max_mini_batch_size;
  if(last_mini_batch_size == 0) {
    last_mini_batch_size = max_mini_batch_size;
  }
  data_reader->set_num_iterations_per_epoch(num_iterations_per_epoch);
  data_reader->set_last_mini_batch_size(last_mini_batch_size);
  data_reader->set_stride_to_last_mini_batch(data_reader->get_stride_to_next_mini_batch());

  data_reader->set_global_mini_batch_size(max_mini_batch_size);
  data_reader->set_global_last_mini_batch_size(last_mini_batch_size);
  return;
}

void data_coordinator::calculate_num_iterations_per_epoch(int mini_batch_size) {
  for(auto&& dr: m_data_readers) {
    if (!dr.second) continue;
    calculate_num_iterations_per_epoch(mini_batch_size, dr.second);
  }
}

int data_coordinator::compute_max_num_parallel_readers(long data_set_size, int mini_batch_size, int requested_num_parallel_readers) const {
  return compute_max_num_parallel_readers(data_set_size, mini_batch_size, requested_num_parallel_readers, this->m_comm);
}

int data_coordinator::compute_max_num_parallel_readers(long data_set_size, int mini_batch_size, int requested_num_parallel_readers, const lbann_comm* comm) {
  int num_parallel_readers = requested_num_parallel_readers;

  if(comm->get_procs_per_trainer() != num_parallel_readers) {
    if (comm->am_trainer_master()) {
      std::cout << "Warning the requested number of parallel readers "
                << num_parallel_readers
                << " does not match the grid size "
                << comm->get_procs_per_trainer()
                << " OVERRIDING requested number of parallel readers."
                << std::endl;
    }
    num_parallel_readers = comm->get_procs_per_trainer();
  }

#if 0
  if(mini_batch_size < num_parallel_readers) {
    if (comm->am_trainer_master()) {
      std::cout << "Warning the requested number of parallel readers "
                << num_parallel_readers
                << " is larger than the requested mini-batch size "
                << mini_batch_size
                << " OVERRIDING requested number of parallel readers."
                << std::endl;
    }
    num_parallel_readers = mini_batch_size;
  }
#endif
  return num_parallel_readers;
}

} // namespace lbann
