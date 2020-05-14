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

#include "lbann/callbacks/callback.hpp"
#include "lbann/metrics/layer_metric.hpp"
#include "lbann/models/model.hpp"
#include "lbann/models/directed_acyclic_graph.hpp"
#include "lbann/objective_functions/layer_term.hpp"
#include "lbann/objective_functions/weight_regularization/l2.hpp"
#include "lbann/utils/memory.hpp"

#include <model.pb.h>
#include <objective_functions.pb.h>

#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace lbann {
namespace proto {

namespace {

/** Instantiate a model based on prototext. */
std::unique_ptr<model>
instantiate_model(lbann_comm* comm,
                  std::unique_ptr<objective_function> obj,
                  const lbann_data::Optimizer& proto_opt,
                  const lbann_data::Model& proto_model) {

  // Construct model
  const auto& type = proto_model.type();
  if (type.empty() || type == "directed_acyclic_graph_model") {
    return make_unique<directed_acyclic_graph_model>(
      comm, obj.release(),
      make_unique<lbann_data::Optimizer>(proto_opt));
  }

  // Throw error if model type is not supported
  LBANN_ERROR("unknown model type (", type, ")");
  return nullptr;
}

/** Setup pointers from objective function to layers.
 *
 *  Layer terms require pointers to layers.
 */
void assign_layers_to_objective_function(
  const std::vector<std::unique_ptr<Layer>>& layer_list,
  objective_function& obj,
  const lbann_data::ObjectiveFunction& proto_obj) {

  // Construct map from layer names to layers
  std::unordered_map<std::string, Layer*> names_to_layers;
  for (auto&& l : layer_list) {
    const auto& name = l->get_name();
    if (names_to_layers.count(name) > 0) {
      LBANN_ERROR("layer name \"", name, "\" is not unique");
    }
    names_to_layers[name] = l.get();
  }

  // Assign layers to layer terms in objective function
  auto&& obj_terms = obj.get_terms();
  El::Int num_layer_terms = 0;
  for (size_t i = 0; i < obj_terms.size(); ++i) {
    auto&& term = dynamic_cast<layer_term*>(obj_terms[i]);
    if (term != nullptr) {
      ++num_layer_terms;
      const auto& params = proto_obj.layer_term(num_layer_terms-1);
      auto* l = names_to_layers[params.layer()];
      if (l == nullptr) {
        LBANN_ERROR("attempted to set objective function layer term ",
                    "to correspond to layer \"", params.layer(), "\", ",
                    "but no such layer exists");
      }
      term->set_layer(*l);
    }
  }

  // Check that layer terms in objective function match prototext
  if (num_layer_terms != proto_obj.layer_term_size()) {
    LBANN_ERROR("recieved ", num_layer_terms,
                " objective function layer terms, but there are ",
                proto_obj.layer_term_size(), " in the prototext");
  }
}

void assign_layers_to_metrics(
  const std::vector<std::unique_ptr<Layer>>& layer_list,
  std::vector<std::unique_ptr<metric>>& metric_list,
  const lbann_data::Model& proto_model) {

  // Construct map from layer names to layers
  std::unordered_map<std::string, Layer*> names_to_layers;
  for (auto&& l : layer_list) {
    const auto& name = l->get_name();
    if (names_to_layers.count(name) > 0) {
      LBANN_ERROR("layer name \"", name, "\" is not unique");
    }
    names_to_layers[name] = l.get();
  }

  // Assign layers to layer metrics
  for (int i=0; i<proto_model.metric_size(); ++i) {
    auto&& m = dynamic_cast<layer_metric*>(metric_list[i].get());
    if (m != nullptr) {
      const auto& params = proto_model.metric(i).layer_metric();
      auto* l = names_to_layers[params.layer()];
      if (l == nullptr) {
        LBANN_ERROR("attempted to set layer metric "
                    "\"", m->name(), "\" "
                    "to correspond to layer \"", params.layer(), "\", "
                    "but no such layer exists");
      }
      m->set_layer(*l);
    }
  }

}

/** Setup pointers from layers to weights. */
void assign_weights_to_layers(
  const std::vector<std::unique_ptr<Layer>>& layer_list,
  std::vector<std::unique_ptr<weights>>& weights_list,
  const lbann_data::Model& proto_model) {

  // Construct map from weights names to weights
  std::unordered_map<std::string, weights*> names_to_weights;
  for (auto&& w : weights_list) {
    const auto& name = w->get_name();
    if (names_to_weights.count(name) > 0) {
      LBANN_ERROR("weights name \"", name, "\" is not unique");
    }
    names_to_weights[name] = w.get();
  }

  // Find weights assigned to each layer
  for (int i=0; i<proto_model.layer_size(); ++i) {
    const auto& proto_layer = proto_model.layer(i);
    auto layer_weights = extract_weights(*layer_list[i]);
    const bool is_frozen = layer_list[i]->is_frozen();
    for (auto&& name : parse_list<std::string>(proto_layer.weights())) {
      auto&& w = names_to_weights[name];
      if (!w) {
        LBANN_ERROR("could not find weights named "
                    "\"", name, "\", "
                    "which are expected by layer ",
                    layer_list[i]->get_name());
      }
      if (is_frozen) {
        w->freeze();
      } else if (w->is_frozen()) {
        w->unfreeze();
      }
      layer_weights.push_back(w);
    }
    layer_list[i]->set_weights(layer_weights);
  }

}

/** Setup pointers from objective function to weights.
 *
 *  L2 weight regularization requires pointers to weights.
 */
void assign_weights_to_objective_function(
  const std::vector<std::unique_ptr<weights>>& weights_list,
  objective_function& obj,
  const lbann_data::ObjectiveFunction& proto_obj) {

  // Construct map from weights names to weights
  std::unordered_map<std::string, weights*> names_to_weights;
  for (auto&& w : weights_list) {
    const auto& name = w->get_name();
    if (names_to_weights.count(name) > 0) {
      LBANN_ERROR("weights name \"", name, "\" is not unique");
    }
    names_to_weights[name] = w.get();
  }

  // Setup weights with L2 regularization
  auto&& obj_terms = obj.get_terms();
  El::Int num_l2_weight_regularization_terms = 0;
  for (size_t i = 0; i < obj_terms.size(); ++i) {
    auto&& term = dynamic_cast<l2_weight_regularization*>(obj_terms[i]);
    if (term != nullptr) {
      ++num_l2_weight_regularization_terms;
      const auto& params = proto_obj.l2_weight_regularization(num_l2_weight_regularization_terms-1);
      std::vector<weights*> term_weights;
      for (auto&& weights_name : parse_list<std::string>(params.weights())) {
        auto&& w = names_to_weights[weights_name];
        if (!w) {
          LBANN_ERROR("attempted to apply L2 weight regularization to "
                      "weights \"", weights_name, "\", "
                      "but no such weights exists");
        }
        term_weights.push_back(w);
      }
      term->set_weights_pointers(term_weights);
    }
  }

}

} // namespace

std::unique_ptr<model> construct_model(
  lbann_comm* comm,
  int training_dr_linearized_data_size,
  const lbann_data::Optimizer& proto_opt,
  const lbann_data::Trainer& proto_trainer,
  const lbann_data::Model& proto_model) {

  // Construct layer graph
  auto&& layer_list = construct_layer_graph(comm,
                                            training_dr_linearized_data_size,
                                            proto_trainer,
                                            proto_model);

  // Construct objective function
  const auto& proto_obj = proto_model.objective_function();
  auto obj = construct_objective_function(proto_obj);
  assign_layers_to_objective_function(layer_list, *obj, proto_obj);

  // Construct weights
  std::vector<std::unique_ptr<weights>> weights_list;
  for (int i=0; i<proto_model.weights_size(); i++) {
    weights_list.push_back(
      construct_weights(comm,
                        proto_opt,
                        proto_model.weights(i)));
  }
  assign_weights_to_layers(layer_list, weights_list, proto_model);
  assign_weights_to_objective_function(weights_list, *obj, proto_obj);

  // Construct metrics
  std::vector<std::unique_ptr<metric>> metric_list;
  for (int i=0; i<proto_model.metric_size(); ++i) {
    const auto& params = proto_model.metric(i).layer_metric();
    metric_list.push_back(make_unique<layer_metric>(comm,
                                                    params.name(),
                                                    params.unit()));
  }
  assign_layers_to_metrics(layer_list, metric_list, proto_model);

  // Construct callbacks
  std::vector<std::unique_ptr<callback_base>> callback_list;
  auto summarizer = std::shared_ptr<lbann_summary>(construct_summarizer(comm, proto_model));
  for (int i=0; i<proto_model.callback_size(); i++) {
    callback_list.push_back(construct_callback(proto_model.callback(i),
                                               summarizer));
  }

  // Instantiate model
  auto m = instantiate_model(comm, std::move(obj), proto_opt, proto_model);
  for (auto&& l   : layer_list   ) { m->add_layer(std::move(l));    }
  for (auto&& w   : weights_list ) { m->add_weights(std::move(w));  }
  for (auto&& met : metric_list  ) { m->add_metric(met.release());  }
  for (auto&& cb  : callback_list) { m->add_callback(std::move(cb)); }
  const auto& name = proto_model.name();
  if (!name.empty()) {
    m->set_name(name);
  }
  return m;

}

} // namespace proto
} // namespace lbann
