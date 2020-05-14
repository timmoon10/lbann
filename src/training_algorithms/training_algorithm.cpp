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

#include "lbann/training_algorithms/training_algorithm.hpp"
#include "lbann/models/model.hpp"
#include "lbann/callbacks/callback.hpp"
#include "lbann/callbacks/checkpoint.hpp"
#include "lbann/callbacks/save_model.hpp"
#include "lbann/callbacks/load_model.hpp"

namespace lbann {

void training_algorithm::setup_models(std::vector<observer_ptr<model>> models, size_t max_mini_batch_size, DataReaderMetaData& dr_metadata) {
  for (observer_ptr<model> m : models) {
    // Set up callbacks
    for (auto* c : m->get_callbacks()) {
      {
        auto* cb = dynamic_cast<callback::checkpoint*>(c);
        if(cb != nullptr) {
          cb->set_active_training_algorithm(this);
        }
      }
    }
    // Setup models
    m->setup(max_mini_batch_size, dr_metadata);
  }
  return;
}


}  // namespace lbann
