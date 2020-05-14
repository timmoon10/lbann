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

#include "lbann/proto/factories.hpp"
#include "lbann/proto/datatype_helpers.hpp"

#include "lbann/layers/learning/fully_connected.hpp"
#include "lbann/layers/transform/pooling.hpp"
#include "lbann/layers/transform/unpooling.hpp"

#include <model.pb.h>
#include <trainer.pb.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace lbann {
namespace proto {

namespace {

/** Setup parent/child relationships between layers. */
void setup_parents_and_children(
       lbann_comm* comm,
       std::vector<Layer*>& layers,
       std::unordered_map<std::string, Layer*>& names_to_layers,
       const lbann_data::Model& proto_model) {
  std::stringstream err;
  for (int i=0; i<proto_model.layer_size(); ++i) {
    const auto& proto_layer = proto_model.layer(i);
    const auto& parents = parse_list<std::string>(proto_layer.parents());
    const auto& children = parse_list<std::string>(proto_layer.children());
    for (const auto& parent : parents) {
      if (names_to_layers.count(parent) == 0) {
        err << "could not find parent layer \"" << parent << "\" "
            << "for layer \"" << layers[i]->get_name() << "\"";
        LBANN_ERROR(err.str());
      }
      layers[i]->add_parent_layer(names_to_layers.at(parent));
    }
    for (const auto& child : children) {
      if (names_to_layers.count(child) == 0) {
        err << "could not find child layer \"" << child << "\" "
            << "for layer \"" << layers[i]->get_name() << "\"";
        LBANN_ERROR(err.str());
      }
      layers[i]->add_child_layer(names_to_layers.at(child));
    }
  }
}

void setup_hints(
       std::vector<Layer*>& layers,
       const std::unordered_map<std::string, Layer*>& names_to_layers,
       const lbann_data::Model& proto_model) {
  std::stringstream err;
  for (int i=0; i<proto_model.layer_size(); ++i) {
    const auto& proto_layer = proto_model.layer(i);
    const auto& hint = proto_layer.hint_layer();
    if (!hint.empty()) {
      if (names_to_layers.count(hint) == 0) {
        err << "could not find hint layer \"" << hint << "\" "
            << "for layer \"" << layers[i]->get_name() << "\"";
        LBANN_ERROR(err.str());
      }
      layers[i]->set_hint_layer(names_to_layers.at(hint));
    }
  }
}

/** Setup paired pooling layers for unpooling layers. */
void setup_unpooling_pointers(lbann_comm* comm,
                              std::vector<Layer*>& layers,
                              std::unordered_map<std::string, Layer*>& names_to_layers,
                              const lbann_data::Model& proto_model) {
  std::stringstream err;
  for (int i=0; i<proto_model.layer_size(); ++i) {
    {
      unpooling_layer<DataType, data_layout::DATA_PARALLEL, El::Device::CPU>* unpool
        = dynamic_cast<unpooling_layer<DataType, data_layout::DATA_PARALLEL, El::Device::CPU>*>(layers[i]);
      if (unpool != nullptr) {
        const auto& pool_name = proto_model.layer(i).unpooling().pooling_layer();
        pooling_layer<DataType, data_layout::DATA_PARALLEL, El::Device::CPU>* pool
          = dynamic_cast<pooling_layer<DataType, data_layout::DATA_PARALLEL, El::Device::CPU>*>(names_to_layers[pool_name]);
        if (pool == nullptr) {
          err << "could not find pooling layer " << pool_name << " "
              << "to pair with unpooling layer " << unpool->get_name();
          LBANN_ERROR(err.str());
        }
        unpool->set_pooling_layer(pool);
      }
    }
#if defined(LBANN_HAS_GPU) && defined(LBANN_UNPOOLING_LAYER_SUPPORTS_GPU)
    {
      unpooling_layer<DataType, data_layout::DATA_PARALLEL, El::Device::GPU>* unpool
        = dynamic_cast<unpooling_layer<DataType, data_layout::DATA_PARALLEL, El::Device::GPU>*>(layers[i]);
      if (unpool != nullptr) {
        const auto& pool_name = proto_model.layer(i).unpooling().pooling_layer();
        pooling_layer<DataType, data_layout::DATA_PARALLEL, El::Device::GPU>* pool
          = dynamic_cast<pooling_layer<DataType, data_layout::DATA_PARALLEL, El::Device::GPU>*>(names_to_layers[pool_name]);
        if (pool == nullptr) {
          err << "could not find pooling layer " << pool_name << " "
              << "to pair with unpooling layer " << unpool->get_name();
          LBANN_ERROR(err.str());
        }
        unpool->set_pooling_layer(pool);
      }
    }
#endif // LBANN_HAS_GPU
  }
}

} // namespace

std::vector<std::unique_ptr<Layer>> construct_layer_graph(
  lbann_comm* comm,
  int training_dr_linearized_data_size,
  const lbann_data::Trainer& proto_trainer,
  const lbann_data::Model& proto_model) {
  std::stringstream err;

  // List of layers
  std::vector<std::unique_ptr<Layer>> layers;
  layers.reserve(proto_model.layer_size());

  // Map from names to layer pointers
  std::unordered_map<std::string, Layer*> names_to_layers;

  // Create each layer in prototext
  for (int i=0; i<proto_model.layer_size(); ++i) {
    const auto& proto_layer = proto_model.layer(i);

    // Check that layer name is valid
    auto name = proto_layer.name();
    const auto& parsed_name = parse_list<std::string>(name);
    if (!name.empty()) {
      if (parsed_name.empty() || parsed_name.front() != name) {
        err << "weights name \"" << name << "\" is invalid since it "
            << "contains whitespace";
        LBANN_ERROR(err.str());
      }
      if (names_to_layers.count(name) != 0) {
        err << "layer name \"" << name << "\" is not unique";
        LBANN_ERROR(err.str());
      }
    }

    // Get parameters from prototext
    const auto model_disable_gpus = proto_model.disable_cuda();

    const auto& layout_str = proto_layer.data_layout();
    data_layout layout = (layout_str.empty()
                          ? data_layout::DATA_PARALLEL
                          : data_layout_from_string(layout_str));

    const auto& num_parallel_readers = proto_trainer.num_parallel_readers();
    El::Device device = El::Device::CPU;
#ifdef LBANN_HAS_GPU
    // Input layers must be on CPU
    if (!proto_layer.has_input() && !model_disable_gpus) {
      const auto& device_str = proto_layer.device_allocation();
      device = (device_str.empty()
                ? El::Device::GPU
                : device_from_string(device_str));
    }
#else
    (void) model_disable_gpus;
#endif // LBANN_HAS_GPU

    auto proto_datatype = proto_layer.datatype();

    // Construct layer
    std::unique_ptr<Layer> l;
#define TEMPLATE_INSTANTIATION(TensorDataType, T_layout, T_device)      \
    do {                                                                \
      if (proto_datatype == TypeToProtoDataType<TensorDataType>::value  \
          && layout == T_layout                                         \
          && device == T_device) {                                      \
        l = construct_layer<TensorDataType, T_layout, T_device>(        \
          comm,                                                         \
          training_dr_linearized_data_size,                             \
          num_parallel_readers,                                         \
          proto_layer);                                                 \
      }                                                                 \
    } while (0)

#define PROTO_DEVICE(T, Device) \
    TEMPLATE_INSTANTIATION(T, data_layout::DATA_PARALLEL, Device); \
    TEMPLATE_INSTANTIATION(T, data_layout::MODEL_PARALLEL, Device)

#include "lbann/macros/instantiate_device.hpp"

#undef TEMPLATE_INSTANTIATION

    // Set up parallel strategy.
    ParallelStrategy& ps = l->get_parallel_strategy();
    ps.sample_groups = proto_layer.parallel_strategy().sample_groups();
    ps.sample_splits = proto_layer.parallel_strategy().sample_splits();
    ps.height_groups = proto_layer.parallel_strategy().height_groups();
    ps.height_splits = proto_layer.parallel_strategy().height_splits();
    ps.width_groups = proto_layer.parallel_strategy().width_groups();
    ps.width_splits = proto_layer.parallel_strategy().width_splits();
    ps.channel_groups = proto_layer.parallel_strategy().channel_groups();
    ps.channel_splits = proto_layer.parallel_strategy().channel_splits();
    ps.filter_groups = proto_layer.parallel_strategy().filter_groups();
    ps.filter_splits = proto_layer.parallel_strategy().filter_splits();
    ps.replications = proto_layer.parallel_strategy().replications();
    ps.depth_groups = proto_layer.parallel_strategy().depth_groups();
    ps.depth_splits = proto_layer.parallel_strategy().depth_splits();

    // Check that layer has been constructed
    if (l == nullptr) {
      err << "could not construct layer " << name;
      LBANN_ERROR(err.str());
    }

    // Initialize layer name and check it is unique
    if (!name.empty()) {
      l->set_name(name);
    }
    name = l->get_name();
    if (names_to_layers.count(name) != 0) {
      err << "layer name \"" << name << "\" is not unique";
      LBANN_ERROR(err.str());
    }
    names_to_layers[name] = l.get();

    if (proto_layer.freeze()) {
      #ifdef LBANN_DEBUG
      if (comm->am_world_master()) {
        std::cout << "freezing " << l->get_name() << std::endl;
      }
      #endif
      l->freeze();
    }
    // Add layer to list
    layers.emplace_back(std::move(l));

  }

  // Setup pointers between layers
  std::vector<Layer*> layer_pointers;
  layer_pointers.reserve(layers.size());
  for (auto&& ptr : layers) { layer_pointers.push_back(ptr.get()); }
  setup_parents_and_children(comm, layer_pointers, names_to_layers, proto_model);
  setup_hints(layer_pointers, names_to_layers, proto_model);
  setup_unpooling_pointers(comm, layer_pointers, names_to_layers, proto_model);

  // Return layer list
  return layers;

}

} // namespace proto
} // namespace lbann
