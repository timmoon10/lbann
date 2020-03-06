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

#include "lbann/utils/lbann_library.hpp"

#include "lbann/proto/factories.hpp"
#include "lbann/utils/omp_diagnostics.hpp"
#include "lbann/utils/threads/thread_utils.hpp"
#include "lbann/callbacks/checkpoint.hpp"
#include "lbann/callbacks/dump_weights.hpp"
#include "lbann/callbacks/save_model.hpp"
#include "lbann/callbacks/load_model.hpp"

#include <lbann.pb.h>
#include <model.pb.h>

namespace lbann {

/// Construct a trainer that contains a lbann comm object and threadpool
std::unique_ptr<trainer> construct_trainer(lbann_comm *comm,
                                             lbann_data::Trainer* pb_trainer,
                                             options *opts) {
  try {
    int procs_per_trainer = 0;
    if(pb_trainer->procs_per_trainer() > 0) {
      procs_per_trainer = pb_trainer->procs_per_trainer();
    }
    if (procs_per_trainer == 0) {
      procs_per_trainer = comm->get_procs_in_world();
    }

    // Set up the communicator and split the grid if necessary
    comm->split_trainers(procs_per_trainer);
    if (pb_trainer->num_parallel_readers() > procs_per_trainer) {
      pb_trainer->set_num_parallel_readers(procs_per_trainer);
    }

    // Adjust the number of parallel readers; this may be adjusted
    // after calling split_trainers()
    // set_num_parallel_readers(*comm, pb);

    // Initalize a per-trainer I/O thread pool
    std::unique_ptr<thread_pool> io_thread_pool = construct_io_thread_pool(comm, opts);

    // Setup I/O threads
    auto io_threads_per_process = io_thread_pool->get_num_threads();
    auto io_threads_offset = io_thread_pool->get_threads_offset();

    // Set algorithmic blocksize in Hydrogen
    if (pb_trainer->hydrogen_block_size() > 0) {
      El::SetBlocksize(pb_trainer->hydrogen_block_size());
    }

    // Set up the communicator and get the grid based on the trainers' spec.
    // We do not currently support splitting different trainers in different ways,
    // as this implies different grids.
    if (procs_per_trainer != comm->get_procs_per_trainer()) {
      comm->split_trainers(procs_per_trainer);
    }

    // Display how the OpenMP threads are provisioned
    // if (opts->has_string("print_affinity")) {
    //   display_omp_setup();
    // }

    // User feedback
    //    print_parameters(comm, pb);

    // Initalize trainer
    std::unique_ptr<trainer> trainer = proto::construct_trainer(comm, *pb_trainer);

    trainer->setup(std::move(io_thread_pool));

    if(opts->get_bool("disable_background_io_activity")) {
      trainer->allow_background_io_activity(false);
    }

    // Report useful information
    if (comm->am_world_master()) {
      print_lbann_configuration(comm,
                                io_threads_per_process,
                                io_threads_offset);
      std::cout << "\n"
                << trainer->get_description()
                << std::endl;
    }

    return trainer;

  } catch (lbann_exception& e) {
    El::mpi::Abort(El::mpi::COMM_WORLD, 1);
  } catch (std::exception& e) {
    El::ReportException(e);  // Elemental exceptions
  }
  return nullptr;
}

/// Setup I/O thread pool that is shared across all models
std::unique_ptr<thread_pool> construct_io_thread_pool(lbann_comm *comm, options *opts) {
  int num_io_threads = num_free_cores_per_process(comm);

  if(opts->has_int("num_io_threads")) {
    int requested_io_threads = opts->get_int("num_io_threads");
    if(requested_io_threads > 0 && requested_io_threads < num_io_threads) {
      num_io_threads = requested_io_threads;
    }
  }

  auto io_threads_offset = free_core_offset(comm);

  if(comm->am_world_master()) {
    std::cout << "\tNum. I/O Threads: " << num_io_threads <<
      " (Limited to # Unused Compute Cores or 1)" << std::endl;
  }

  auto io_thread_pool = make_unique<thread_pool>();
  io_thread_pool->launch_pinned_threads(num_io_threads, io_threads_offset);

  return io_thread_pool;
}

std::unique_ptr<model> build_model_from_prototext(
  int argc, char **argv,
  const lbann_data::Trainer* pb_trainer,
  lbann_data::LbannPB &pb,
  lbann_comm *comm,
  options *opts,
  thread_pool& io_thread_pool,
  std::vector<std::shared_ptr<callback_base>>& shared_callbacks,
  bool first_model) {

  int random_seed = lbann_default_random_seed;
  bool master = comm->am_world_master();
  if (master) {
    std::cerr << "starting build_model_from_prototext" << std::endl;
  }

  std::ostringstream err;

  lbann_data::Model *pb_model = pb.mutable_model();

  // Check to see if the model wants to reduce the I/O parallelism
  if(pb_model->serialize_io() && io_thread_pool.get_num_threads() != 1) {
    if(master) {
      std::cout << "Model " << pb_model->name() << " serialized the I/O threads" << std::endl;
    }
    io_thread_pool.relaunch_pinned_threads(1);
  }

  // Get I/O thread details
  auto io_threads_per_process = io_thread_pool.get_num_threads();

  /// @todo BVE FIXME should this be in the trainer
  // Change random seed if needed.
  if (pb_model->random_seed() > 0) {
    random_seed = pb_model->random_seed();
    // Reseed here so that setup is done with this new seed.
    init_random(random_seed);
    init_data_seq_random(random_seed);
  }

  // Initialize models differently if needed.
#ifndef LBANN_DETERMINISTIC
  if (!pb_trainer->random_init_trainers_identically()) {
    hash_combine(random_seed, comm->get_trainer_rank());
    // Reseed here so that setup is done with this new seed.
    init_random(random_seed);
    init_data_seq_random(random_seed);
  }
#else
  if (!pb_trainer->random_init_trainers_identically()) {
    if (master) {
      std::cout << "WARNING: forcing 'random_init_trainers_identically' " <<
        "due to sequential consistency" << std::endl;
    }
  }
#endif

  // Save info to file; this includes the complete prototext (with any over-rides
  // from the cmd line) and various other info
  save_session(*comm, argc, argv, pb);

  // Display how the OpenMP threads are provisioned
  if (opts->has_string("print_affinity")) {
    display_omp_setup();
  }

  // Update the index lists to accomodate multi-trainer / multi-model specification
  customize_data_readers_index_list(*comm, pb);

  // Initialize data readers
  //@todo: code not in place for correctly handling image preprocessing
  std::map<execution_mode, generic_data_reader *> data_readers;
  bool is_shared_training_data_reader = pb_model->shareable_training_data_reader();
  bool is_shared_testing_data_reader = pb_model->shareable_testing_data_reader();
  if (opts->has_string("share_testing_data_readers")) {
    is_shared_testing_data_reader = opts->get_bool("share_testing_data_readers");
  }
  init_data_readers(comm, pb, data_readers, is_shared_training_data_reader, is_shared_testing_data_reader);

  // hack to prevent all data readers from loading identical data; instead,
  // share a single copy. See data_reader_jag_conduit_hdf5 for example
  if (first_model) {
    if (opts->has_string("share_data_reader_data")) {
      for (auto&& t : data_readers) {
        opts->set_ptr((void*)t.second);
      }
    }
  }

  // User feedback
  print_parameters(*comm, pb);

  // Initalize model
  std::unique_ptr<model> ret_model = proto::construct_model(comm,
                                                            data_readers,
                                                            pb.optimizer(),
                                                            pb.trainer(),
                                                            pb.model());

  // Add the trainer's callbacks to the model
  for (auto&& c : shared_callbacks) {
    ret_model->add_callback(c);
  }

  // If the checkpoint directory has been overridden reset it before
  // setting up the model
  if (opts->has_string("ckpt_dir")) {
    for (auto&& c : ret_model->get_callbacks()) {
      {
        auto* cb = dynamic_cast<callback::checkpoint*>(c);
        if(cb != nullptr) {
          cb->set_checkpoint_dir(opts->get_string("ckpt_dir"));
          std::cout << "Setting the checkpoint directory to " << cb->get_checkpoint_dir() << std::endl;
        }
      }
      {
        auto* cb = dynamic_cast<callback::dump_weights*>(c);
        if(cb != nullptr) {
          cb->set_target_dir(opts->get_string("ckpt_dir"));
          std::cout << "Setting the dump weights directory to " << cb->get_target_dir() << std::endl;
        }
      }
      {
        auto* cb = dynamic_cast<callback::save_model*>(c);
        if(cb != nullptr) {
          cb->set_target_dir(opts->get_string("ckpt_dir"));
          std::cout << "Setting the dump weights directory to " << cb->get_target_dir() << std::endl;
        }
      }
    }
  }

  // Setup data readers
  for(auto&& dr: data_readers) {
    dr.second->setup(io_threads_per_process, &io_thread_pool);
    dr.second->set_rank(comm->get_rank_in_trainer());
  }

  // Setup models
  ret_model->setup();

  if (opts->get_bool("use_data_store") || opts->get_bool("preload_data_store") || opts->get_bool("data_store_cache") || opts->has_string("data_store_spill")) {
    if (master) {
      std::cout << "\nUSING DATA STORE!\n\n";
    }
    for (auto&& r : data_readers) {
      if (!r.second) continue;
      r.second->setup_data_store(pb_model->mini_batch_size());
    }
  }

  // restart model from checkpoint if we have one
  //@todo
  //model->restartShared();

#ifndef LBANN_DETERMINISTIC
  // Under normal conditions, reinitialize the random number generator so
  // that regularization techniques (e.g. dropout) generate unique patterns
  // on different ranks.
  init_random(random_seed + comm->get_rank_in_world());
#else
  if(comm->am_world_master()) {
    std::cout <<
      "--------------------------------------------------------------------------------\n"
      "ALERT: executing in sequentially consistent mode -- performance will suffer\n"
      "--------------------------------------------------------------------------------\n";
  }
#endif

  if (opts && opts->has_string("restart_dir")) {
    bool loaded = callback::load_model::load_model_weights(
      opts->get_string("restart_dir"),
      ret_model.get(),
      opts->get_bool("restart_dir_is_fullpath"));
    if(!loaded) {
      LBANN_ERROR("Unable to reload model from given restart directory: ",
                  opts->get_string("restart_dir"));
    }
  }

  return ret_model;
}

void print_lbann_configuration(lbann_comm *comm, int io_threads_per_process, int io_threads_offset) {
  // Report hardware settings
  std::cout << "Hardware properties (for master process)" << std::endl
            << "  Processes on node          : " << comm->get_procs_per_node() << std::endl
            << "  Total number of processes  : " << comm->get_procs_in_world() << std::endl
            << "  OpenMP threads per process : " << omp_get_max_threads() << std::endl
            << "  I/O threads per process (+offset) : " << io_threads_per_process
            << " (+" << io_threads_offset << ")" << std::endl;
#ifdef HYDROGEN_HAVE_CUDA
  std::cout << "  GPUs on node               : " << El::GPUManager::NumDevices() << std::endl;
#endif // HYDROGEN_HAVE_CUDA
  std::cout << std::endl;

  // Report build settings
  std::cout << "Build settings" << std::endl;
  std::cout << "  Type     : ";
#ifdef LBANN_DEBUG
  std::cout << "Debug" << std::endl;
#else
  std::cout << "Release" << std::endl;
#endif // LBANN_DEBUG
  std::cout << "  Aluminum : ";
#ifdef LBANN_HAS_ALUMINUM
  std::cout << "detected" << std::endl;
#else
  std::cout << "NOT detected" << std::endl;
#endif // LBANN_HAS_ALUMINUM
  std::cout << "  CUDA     : ";
#ifdef LBANN_HAS_GPU
  std::cout << "detected" << std::endl;
#else
  std::cout << "NOT detected" << std::endl;
#endif // LBANN_HAS_GPU
  std::cout << "  cuDNN    : ";
#ifdef LBANN_HAS_CUDNN
  std::cout << "detected" << std::endl;
#else
  std::cout << "NOT detected" << std::endl;
#endif // LBANN_HAS_CUDNN
  std::cout << "  CUB      : ";
#ifdef HYDROGEN_HAVE_CUB
  std::cout << "detected" << std::endl;
#else
  std::cout << "NOT detected" << std::endl;
#endif // HYDROGEN_HAVE_CUB
  const auto* env = std::getenv("MV2_USE_CUDA");
  std::cout << "  MV2_USE_CUDA : " << (env != nullptr ? env : "") << std::endl;
  std::cout << std::endl;

#ifdef LBANN_HAS_ALUMINUM
  std::cout << "Aluminum Features:" << std::endl;
  std::cout << "  NCCL : ";
#ifdef AL_HAS_NCCL
  std::cout << "enabled" << std::endl;
#else
  std::cout << "disabled" << std::endl;
#endif // AL_HAS_NCCL
  std::cout << std::endl;
#endif // LBANN_HAS_ALUMINUM

  // Report model settings
  const auto& grid = comm->get_trainer_grid();
  std::cout << "Trainer settings" << std::endl
            << "  Trainers              : " << comm->get_num_trainers() << std::endl
            << "  Processes per trainer : " << comm->get_procs_per_trainer() << std::endl
            << "  Grid dimensions       : " << grid.Height() << " x " << grid.Width() << std::endl;
  std::cout << std::endl;
}

} // namespace lbann
