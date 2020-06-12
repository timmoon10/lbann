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

#ifndef LBANN_LAYER_EVALUATION_HPP_INCLUDED
#define LBANN_LAYER_EVALUATION_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"

namespace lbann {

/** @brief Interface with objective function and metrics. */
template <typename TensorDataType>
class abstract_evaluation_layer : public transform_layer<TensorDataType> {
public:
#ifdef LBANN_DETERMINISTIC
  using EvalDataType = EvalType;
#else
  using EvalDataType = TensorDataType;
#endif
  using CPUMatType = El::Matrix<EvalDataType, El::Device::CPU>;

public:

  /** Get scaling factor. */
  EvalType get_scale() const { return m_scale; }
  /** Set scaling factor. */
  void set_scale(EvalType scale) { m_scale = scale; }
  /** Get evaluated value. */
  EvalType get_value(bool scaled = true);

  /** Construct an evaluation layer.
   *  The caller is responsible for deallocating the layer.
   */
  static abstract_evaluation_layer* construct(lbann_comm *comm,
                                              data_layout layout,
                                              El::Device device);

protected:
  abstract_evaluation_layer(lbann_comm *comm);
  void setup_dims(DataReaderMetaData& dr_metadata) override;
  void setup_data(size_t max_mini_batch_size) override;
  void fp_compute() override;
  void bp_compute() override;

private:

  /** Scaling factor to apply to evaluated value. */
  EvalType m_scale = 0;
  /** Evaluated value.
   *  The value may be stored in pinned memory.
   */
  CPUMatType m_value;
  /** Non-blocking allreduce request. */
  Al::request m_allreduce_req;
#ifdef LBANN_HAS_GPU
  /** CUDA event after a non-blocking GPU-CPU memory copy. */
  cuda::event_wrapper m_copy_event;
#endif // LBANN_HAS_GPU

};

/** Evaluation layer.
 *  Computes the average value across a mini-batch. If the input
 *  tensor has multiple neurons, their values are added together.
 */
template <typename TensorDataType,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class evaluation_layer : public abstract_evaluation_layer<TensorDataType> {
public:
  evaluation_layer(lbann_comm *comm) : abstract_evaluation_layer<TensorDataType>(comm) {}
  evaluation_layer* copy() const override { return new evaluation_layer(*this); }
  std::string get_type() const override { return "evaluation"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }
};

LBANN_DEFINE_LAYER_BUILDER(evaluation);

#ifndef LBANN_EVALUATION_LAYER_INSTANTIATE
#define PROTO(T)                           \
  extern template class abstract_evaluation_layer<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#undef LBANN_INSTANTIATE_CPU_HALF
#undef LBANN_INSTANTIATE_GPU_HALF

#define PROTO_DEVICE(T, Device)                                         \
  extern template class evaluation_layer<T, data_layout::DATA_PARALLEL, Device>; \
  extern template class evaluation_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_EVALUATION_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_EVALUATION_HPP_INCLUDED
