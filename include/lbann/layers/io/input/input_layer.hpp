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

#ifndef LBANN_LAYERS_INPUT_LAYER_HPP_INCLUDED
#define LBANN_LAYERS_INPUT_LAYER_HPP_INCLUDED

#include "lbann/layers/io/input/generic_input_layer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/models/model.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace lbann {

/** @brief Interface with data reader. */
template <typename TensorDataType,
          typename T_io_buffer,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class input_layer : public generic_input_layer<TensorDataType> {
  static_assert(T_layout == data_layout::DATA_PARALLEL,
                "input layer only supports DATA_PARALLEL data layout");
 public:
  /** @name Public Types */
  ///@{

  /** @brief The local tensor type expected for IO in this object. */
  using IODataType = DataType;

  ///@}
 public:

  /// @todo make the map and vector references
  input_layer(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode,
    generic_data_reader *> data_readers, bool data_set_spans_models = true,
    data_reader_target_mode target_mode = data_reader_target_mode::CLASSIFICATION)
    : generic_input_layer<TensorDataType>(comm, num_parallel_readers, data_readers, data_set_spans_models, target_mode) {
    // Initialize two buffers
    initialize_io_buffer(comm, std::min(num_parallel_readers, data_type_layer<TensorDataType>::m_comm->get_procs_per_trainer()), data_readers);
    initialize_io_buffer(comm, std::min(num_parallel_readers, data_type_layer<TensorDataType>::m_comm->get_procs_per_trainer()), data_readers);
    for (auto io_buffer : this->m_io_buffers) {
      io_buffer->fetch_data_fn = new fetch_data_functor<IODataType>(target_mode);
      io_buffer->update_data_reader_fn = new update_data_reader_functor();
    }
  }
  input_layer(const input_layer&) = default;
  input_layer& operator=(const input_layer&) = default;
  input_layer* copy() const override {
    return new input_layer(*this);
  }

  inline void initialize_io_buffer(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers) {
    generic_input_layer<TensorDataType>::template initialize_io_buffer<T_io_buffer>(comm, num_parallel_readers, data_readers);
  }

  std::string get_type() const override { return "input"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

};

#ifndef LBANN_INPUT_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device)         \
  extern template class input_layer<    \
    T, partitioned_io_buffer<T>,        \
    data_layout::DATA_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_INPUT_LAYER_INSTANTIATE

} // namespace lbann

#endif  // LBANN_LAYERS_INPUT_LAYER_HPP_INCLUDED
