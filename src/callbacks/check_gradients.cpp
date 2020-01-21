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

#include "lbann/callbacks/check_gradients.hpp"
#include "lbann/data_readers/data_reader.hpp"
#include "lbann/layers/io/input/generic_input_layer.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/memory.hpp"

#include "lbann/utils/h2_tmp.hpp"

#include <callbacks.pb.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>

namespace lbann {
namespace callback {

namespace {

/** @details Forward prop is applied to all layers, except input
 *  layers. It is assumed that input layers have already loaded data.
 */
EvalType compute_objective_function(model& m) {
  const auto& c = static_cast<sgd_execution_context&>(m.get_execution_context());

  // Forward prop, skipping input layers
  for (auto&& l : m.get_layers()) {
    if (dynamic_cast<generic_input_layer<DataType>*>(l) == nullptr) {
      l->forward_prop();
    }
  }

  // Get objective function value
  auto&& obj = m.get_objective_function();
  const auto mode = c.get_execution_mode();
  const auto mini_batch_size = c.get_current_mini_batch_size();
  obj->start_evaluation(mode, mini_batch_size);
  return obj->finish_evaluation(mode, mini_batch_size);

}

struct DefaultErrorReporter
{
  template <typename... Ts>
  void DispatchError(Ts&&...)
  {
    LBANN_ERROR("Unable to dispatch functor.");
  }

  template <typename... Ts>
  void DeductionError(Ts&&...)
  {
    LBANN_ERROR("Unable to deduce an argument type.");
  }
};

struct CheckWeightsFunctor : DefaultErrorReporter
{
  model &m;
  sgd_execution_context const& c;
  EvalType epsilon;
  EvalType step_size;
  EvalType expected_error;
  bool verbose;
  bool error_on_failure;

  CheckWeightsFunctor(model& arg_m, sgd_execution_context const& arg_c,
                      EvalType arg_epsilon, EvalType arg_step_size, EvalType arg_expected_error,
                      bool arg_verbose, bool arg_error_on_failure)
    : m(arg_m), c(arg_c),
      epsilon(arg_epsilon), step_size(arg_step_size), expected_error(arg_expected_error),
      verbose(arg_verbose), error_on_failure(arg_error_on_failure)
  {}

  template <typename TensorDataType>
  void operator()(data_type_weights<TensorDataType>& dtw) {
    // Get weights matrix and gradient
    const El::AbstractDistMatrix<TensorDataType>& weights_matrix = dtw.get_values();
    const El::AbstractDistMatrix<TensorDataType>& gradient = dtw.get_optimizer()->get_gradient();

    // Iterate through weights matrix entries
    for (El::Int col = 0; col < weights_matrix.Width(); ++col) {
      for (El::Int row = 0; row < weights_matrix.Height(); ++row) {
        const bool weight_is_local = weights_matrix.IsLocal(row, col);
        const El::Int local_row = (weight_is_local ?
                                   weights_matrix.LocalRow(row) :
                                   0);
        const El::Int local_col = (weight_is_local ?
                                   weights_matrix.LocalCol(col) :
                                   0);
        const TensorDataType initial_weight = (weight_is_local ?
                                         weights_matrix.GetLocal(local_row,
                                                                 local_col) :
                                         TensorDataType(0.));

        // Compute objective function values
        // Note: matrix entry is reset after computing objective
        // function values
        dtw.set_value(initial_weight + El::To<TensorDataType>(2 * step_size), row, col);
        const EvalType f_2h = compute_objective_function(m);
        dtw.set_value(initial_weight + El::To<TensorDataType>(step_size), row, col);
        const EvalType f_h = compute_objective_function(m);
        dtw.set_value(initial_weight - El::To<TensorDataType>(step_size), row, col);
        const EvalType f_nh = compute_objective_function(m);
        dtw.set_value(initial_weight - El::To<TensorDataType>(2 * step_size), row, col);
        const EvalType f_n2h = compute_objective_function(m);
        dtw.set_value(initial_weight, row, col);

        // Compute relative error in gradient.
        // Note: only weight owner participates
        if (weight_is_local && weights_matrix.RedundantRank() == 0) {
          const EvalType analytical_gradient
            = gradient.GetLocal(local_row, local_col);
          const EvalType numerical_gradient
            = (- f_2h + 8 * f_h - 8 * f_nh + f_n2h) / (12 * step_size);
          const EvalType error = std::fabs(analytical_gradient - numerical_gradient);
          auto relative_error = EvalType(0.);
          if (error != EvalType(0.)) {
            relative_error = error / std::max(std::fabs(analytical_gradient),
                                              std::fabs(numerical_gradient));
          }

          // Print warning if relative error is large
          if (error > expected_error || std::isnan(error) || std::isinf(error)) {
            std::cout << "  GRADIENT ERROR: " << dtw.get_name() << ", "
                      << "entry (" << row << "," << col << ")" << std::endl;
            std::cout << "    Weight              = " << initial_weight << std::endl
                      << "    Analytical gradient = " << analytical_gradient << std::endl
                      << "    Numerical gradient  = " << numerical_gradient << std::endl
                      << "    Error               = " << error << std::endl
                      << "    Relative error      = " << relative_error << std::endl;
            if (error_on_failure) {
              LBANN_ERROR("gradient checking found large difference between "
                          "analytical and numerical gradients");
            }
          } else if (verbose) {
            std::cout << "  " << dtw.get_name() << ", "
                      << "entry (" << row << "," << col << ")" << std::endl;
            std::cout << "    Weight              = " << initial_weight << std::endl
                      << "    Analytical gradient = " << analytical_gradient << std::endl
                      << "    Numerical gradient  = " << numerical_gradient << std::endl
                      << "    Error               = " << error << std::endl
                      << "    Relative error      = " << relative_error << std::endl;
          }
        }
      }
    }
    return;
  }
}; // struct CheckWeightsFunctor

} // namespace

check_gradients::check_gradients(std::set<execution_mode> modes,
                                 DataType step_size,
                                 bool verbose,
                                 bool error_on_failure)
  : m_modes(std::move(modes)),
    m_step_size(step_size),
    m_verbose(verbose),
    m_error_on_failure(error_on_failure) {}

void check_gradients::do_check_gradients(model& m) const {

  // Get objects from model
  const auto& c = static_cast<sgd_execution_context&>(m.get_execution_context());
  auto& comm = *m.get_comm();
  const auto mode = c.get_execution_mode();
  const auto& layers = m.get_layers();

  // Return immediately if gradient check isn't currently needed
  if (!m_modes.empty() && m_modes.count(mode) == 0) { return; }

  // Reset statistics and gradients
  m.get_objective_function()->reset_statistics(mode);
  for (auto&& met : m.get_metrics()) {
    met->reset_statistics(mode);
  }
  for (auto&& w : m.get_weights()) {
    auto&& opt = w->get_optimizer();
    if (opt != nullptr) { opt->clear_gradient(); }
  }

  // Load data in input layers
  for (auto&& l : m.get_layers()) {
    if (dynamic_cast<generic_input_layer<DataType>*>(l) != nullptr) {
      l->forward_prop();
    }
  }

  // Compute objective function
  const EvalType objective = compute_objective_function(m);

  // Choose finite difference step
  // Note: Consider a central difference scheme:
  //   f'(x) ~ ( - f(x+2h) + 8 f(x+h) - 8 f(x-h) + f(x-2h) ) / 12h
  // By Taylor's theorem, the truncation error is bounded by
  //   E_trunc <= | f'''''(xi) | / 18 * h^4
  // Assuming f can be computed to a relative accuracy of epsilon,
  //   E_fl <= epsilon * | f(chi) | / h
  // For simplicity, we assume f(chi) ~ f(x), and | f'''''(xi) | ~ 1.
  // If step size is not specified, then we choose h so that
  //   E_fl <= sqrt(epsilon)
  // For the integrity of the test, the current implementation uses an
  // epsilon based on the minimum step size of the float data type
  const EvalType epsilon = std::pow(std::numeric_limits<DataType>::epsilon(), 0.9);
  const EvalType step_size = (m_step_size > EvalType{0} ?
                              m_step_size :
                              std::fabs(objective) * El::Sqrt(epsilon));
  EvalType expected_error = (epsilon * objective / step_size
                             + std::pow(step_size, 4) / 18);
  expected_error = std::pow(expected_error, 0.9);

  // Compute gradients
  m.get_objective_function()->differentiate();
  m.get_objective_function()->compute_weight_regularization();
  for (El::Int i = layers.size()-1; i >= 0; --i) {
    layers[i]->back_prop();
  }

  // Print objective function value
  if (comm.am_world_master()) {
    std::cout << "----------------------------------------------------------------\n"
              << "Gradient checking...\n"
              << "  Objective function value = " << objective << "\n"
              << "  Step size                = " << step_size << "\n"
              << "  Expected gradient error  = " << expected_error << "\n";
  }

  for (weights *w : m.get_weights()) {
    if (!w->has_optimizer()) {
      continue;
    }
    if (comm.am_world_master()) {
      std::cout << "Checking " << w->get_name() << std::endl;
    }

    using WeightsTypes =
      h2::meta::tlist::ExpandTL<data_type_weights, supported_layer_data_type>;
    using Dispatcher = h2::multimethods::SwitchDispatcher<CheckWeightsFunctor,
                                                          void,
                                                          weights,
                                                          WeightsTypes>;
    Dispatcher::Exec(CheckWeightsFunctor(m, c, epsilon, step_size, expected_error, m_verbose, m_error_on_failure), *w);
  }
  if (comm.am_world_master()) {
    std::cout << "----------------------------------------------------------------\n";
  }

  // Clean up
  /// @todo tym: I'm not sure if data readers are properly reset
  for (auto&& l : m.get_layers()) {
    auto&& input = dynamic_cast<generic_input_layer<DataType>*>(l);
    if (input != nullptr) {
      auto&& reader = input->get_data_reader(mode);
      reader->set_initial_position();
    }
  }
  m.get_objective_function()->reset_statistics(mode);
  for (auto&& met : m.get_metrics()) {
    met->reset_statistics(mode);
  }

}

// Builder function
std::unique_ptr<callback_base>
build_check_gradients_callback_from_pbuf(
  const google::protobuf::Message& proto_msg, const std::shared_ptr<lbann_summary>&) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackCheckGradients&>(proto_msg);
  const auto& modes =
    parse_set<execution_mode>(params.execution_modes());
  return make_unique<check_gradients>(modes,
                                      params.step_size(),
                                      params.verbose(),
                                      params.error_on_failure());
}

} // namespace callback
} // namespace lbann
