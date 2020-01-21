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

#define LBANN_POOLING_LAYER_INSTANTIATE
#include "lbann/layers/transform/pooling.hpp"

#include <lbann/proto/proto_common.hpp>
#include <lbann.pb.h>

namespace lbann {
namespace {

template <typename T, data_layout L, El::Device D>
struct Builder
{
  static std::unique_ptr<Layer> Build(...)
  {
    LBANN_ERROR("Attempted to instantiate layer \"pooling\" with "
                "Layout=", to_string(L), ".\nThis layer is only "
                "supported with DATA_PARALLEL data layout.");
    return nullptr;
  }
};

template <typename TensorDataType, El::Device Device>
struct Builder<TensorDataType, data_layout::DATA_PARALLEL, Device>
{
  template <typename... Args>
  static std::unique_ptr<Layer> Build(Args&&... args)
  {
    using LayerType = pooling_layer<TensorDataType,
                                    data_layout::DATA_PARALLEL,
                                    Device>;
    return lbann::make_unique<LayerType>(std::forward<Args>(args)...);
  }
};
}// namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer> build_pooling_layer_from_pbuf(
  lbann_comm* comm, lbann_data::Layer const& proto_layer)
{
  LBANN_ASSERT_MSG_HAS_FIELD(proto_layer, pooling);

  using BuilderType = Builder<TensorDataType, Layout, Device>;
  const auto& params = proto_layer.pooling();

  const auto& mode_str = params.pool_mode();
  pool_mode mode = pool_mode::invalid;
  if (mode_str == "max" )            { mode = pool_mode::max; }
  if (mode_str == "average" )        { mode = pool_mode::average; }
  if (mode_str == "average_no_pad" ) { mode = pool_mode::average_no_pad; }
  if (params.has_vectors()) {
    const auto& dims = parse_list<int>(params.pool_dims());
    const auto& pads = parse_list<int>(params.pool_pads());
    const auto& strides = parse_list<int>(params.pool_strides());
    return BuilderType::Build(
      comm, dims.size(), dims, pads, strides, mode);
  } else {
    const auto& num_dims = params.num_dims();
    const auto& dim = params.pool_dims_i();
    const auto& pad = params.pool_pads_i();
    const auto& stride = params.pool_strides_i();
    return BuilderType::Build(
      comm, num_dims, dim, pad, stride, mode);
  }
}

#define PROTO_DEVICE(T, Device) \
  template class pooling_layer<T, data_layout::DATA_PARALLEL, Device>; \
  LBANN_LAYER_BUILDER_ETI(pooling, T, Device)

#include "lbann/macros/instantiate_device.hpp"

}// namespace lbann
