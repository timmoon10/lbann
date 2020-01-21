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

#include "lbann/weights/data_type_weights.hpp"
#include "lbann/weights/initializer.hpp"
#include "lbann/weights/variance_scaling_initializers.hpp"

#include "lbann/proto/helpers.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/utils/factory.hpp"

#include <weights.pb.h>

namespace lbann {
namespace proto {
namespace {

using MessageT = google::protobuf::Message;

// Define the factory type.
using factory_type = lbann::generic_factory<
  lbann::weights_initializer,
  std::string,
  generate_builder_type<lbann::weights_initializer,
                        MessageT const&>,
  default_key_error_policy>;

/** @brief Singleton holder for a factory.
 *
 *  @note This design requires that the builder function be valid for
 *  every combination of T, L, and D. That is, layer types for which a
 *  combination is invalid must handle that error inside their builder
 *  function.
 */
template <typename T>
class factory_manager
{
public:
  factory_manager() { register_default_builders(); }
  factory_type const& get() const noexcept { return factory_; }

private:
  void register_default_builders()
  {
    factory_.register_builder("ConstantInitializer",
                              build_constant_initializer_from_pbuf<T>);
    factory_.register_builder("ValueInitializer",
                              build_value_initializer_from_pbuf<T>);
    factory_.register_builder("UniformInitializer",
                              build_uniform_initializer_from_pbuf<T>);
    factory_.register_builder("NormalInitializer",
                              build_normal_initializer_from_pbuf<T>);
    factory_.register_builder("GlorotNormalInitializer",
                              build_glorot_initializer_from_pbuf<T>);
    factory_.register_builder("GlorotUniformInitializer",
                              build_glorot_initializer_from_pbuf<T>);
    factory_.register_builder("HeNormalInitializer",
                              build_he_initializer_from_pbuf<T>);
    factory_.register_builder("HeUniformInitializer",
                              build_he_initializer_from_pbuf<T>);
    factory_.register_builder("LeCunNormalInitializer",
                              build_lecun_initializer_from_pbuf<T>);
    factory_.register_builder("LeCunUniformInitializer",
                              build_lecun_initializer_from_pbuf<T>);
  }

private:
  factory_type factory_;
};

template <typename TensorDataType>
factory_type const& get_weight_initializer_factory() noexcept
{
  static factory_manager<TensorDataType> factory_mgr_;
  return factory_mgr_.get();
}

/* Construct a weights initialization specified with prototext. */
template <typename TensorDataType>
std::unique_ptr<weights_initializer>
construct_initializer(const lbann_data::Weights& proto_weights) {
  auto const& factory = get_weight_initializer_factory<TensorDataType>();
  auto const& msg =
    helpers::get_oneof_message(proto_weights.initializer(), "initializer_type");
  return factory.create_object(msg.GetDescriptor()->name(), msg);
}

} // namespace

std::unique_ptr<weights> construct_weights(
  lbann_comm* comm,
  const lbann_data::Optimizer& proto_opt,
  const lbann_data::Weights& proto_weights) {
  std::stringstream err;

  auto proto_datatype = proto_weights.datatype();

  // Instantiate weights
  //  auto w = make_unique<data_type_weights<DataType>>(comm);
  std::unique_ptr<weights> w;
  std::unique_ptr<weights_initializer> init;
  std::unique_ptr<optimizer> opt;

#define TEMPLATE_INSTANTIATION(TensorDataType)                                \
    do {                                                                      \
      if (proto_datatype == TypeToProtoDataType<TensorDataType>::value) {     \
        w = make_unique<data_type_weights<TensorDataType>>(comm);             \
        init = (proto_weights.has_initializer()                               \
          ? construct_initializer<TensorDataType>(proto_weights)              \
          : nullptr);                                                         \
        const lbann_data::Optimizer& opt_msg = (proto_weights.has_optimizer() \
          ? proto_weights.optimizer()                                         \
          : proto_opt);                                                       \
        opt = (helpers::has_oneof(opt_msg, "optimizer_type")                  \
          ? construct_optimizer<TensorDataType>(opt_msg)                      \
          : nullptr);                                                         \
      }                                                                       \
    } while (0)

#define PROTO(T) \
    TEMPLATE_INSTANTIATION(T)

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

#undef PROTO
#undef TEMPLATE_INSTANTIATION

  // Set weights name if provided
  const auto& name = proto_weights.name();
  const auto& parsed_name = parse_list<std::string>(name);
  if (!name.empty()) {
    if (parsed_name.empty() || parsed_name.front() != name) {
      err << "weights name \"" << name << "\" is invalid since it "
          << "contains whitespace";
      LBANN_ERROR(err.str());
    }
    w->set_name(name);
  }

  // Set weights initializer and optimizer
  w->set_initializer(std::move(init));
  w->set_optimizer(std::move(opt));

  return w;
}

} // namespace proto
} // namespace lbann
