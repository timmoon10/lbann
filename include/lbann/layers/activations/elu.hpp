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

#ifndef LBANN_LAYERS_ACTIVATIONS_ELU_HPP_INCLUDED
#define LBANN_LAYERS_ACTIVATIONS_ELU_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"

namespace lbann {

/** @brief Exponential linear unit.
 *
 *  @f[
 *    \text{ELU}(x; \alpha) =
 *      \begin{cases}
 *        x                & x > 0 \\
 *        \alpha (e^x - 1) & x \leq 0
 *      \end{cases}
 *  @f]
 *  @f$\alpha@f$ should be non-negative. See:
 *
 *  Djork-Arne Clevert, Thomas Unterthiner, and Sepp Hochreiter. "Fast
 *  and accurate deep network learning by exponential linear units
 *  (ELUs)." arXiv preprint arXiv:1511.07289 (2015).
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class elu_layer : public data_type_layer<TensorDataType> {
public:
  elu_layer(lbann_comm *comm, TensorDataType alpha = 1)
    : data_type_layer<TensorDataType>(comm), m_alpha(alpha) {}
  elu_layer* copy() const override { return new elu_layer(*this); }
  std::string get_type() const override { return "ELU"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

  description get_description() const override {
    auto desc = data_type_layer<TensorDataType>::get_description();
    desc.add("alpha", m_alpha);
    return desc;
  }

protected:
  void setup_dims() override {
    data_type_layer<TensorDataType>::setup_dims();
    this->set_output_dims(this->get_input_dims());
  }
  void fp_compute() override;
  void bp_compute() override;

private:
  /** Scale parameter for negative region. */
  TensorDataType m_alpha;

};

#ifndef LBANN_ELU_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device) \
  extern template class elu_layer<T, data_layout::DATA_PARALLEL, Device>; \
  extern template class elu_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_ELU_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_ACTIVATIONS_ELU_HPP_INCLUDED
