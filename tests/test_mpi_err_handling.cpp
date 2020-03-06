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
//
// lbann_proto.cpp - prototext application
////////////////////////////////////////////////////////////////////////////////

#include "lbann/lbann.hpp"
#include <vector>

using namespace lbann;

const int Buf_size = 10000;
const int Trainer = 0;

int main(int argc, char *argv[]) {
  int random_seed = lbann_default_random_seed;
  world_comm_ptr comm = initialize(argc, argv, random_seed);

  try {
    const int size = comm->get_procs_in_world();
    const int me = comm->get_rank_in_world();
    if (size != 2) {
      LBANN_ERROR("Please run with two ranks");
    }

    if (me == 0) {
      std::vector<int> buf(Buf_size,-1);
      comm->send(buf.data(), Buf_size, Trainer, 1);
    }

    else {
      std::vector<int> buf;
      comm->recv(buf.data(), 0, Trainer, 0); 
    }


  } catch (lbann_exception& e) {
    e.print_report();
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

