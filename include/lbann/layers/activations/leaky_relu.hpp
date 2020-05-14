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

#ifndef LBANN_LAYERS_ACTIVATIONS_LEAKY_RELU_HPP_INCLUDED
#define LBANN_LAYERS_ACTIVATIONS_LEAKY_RELU_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/distconv.hpp"

namespace lbann {

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class leaky_relu_distconv_adapter: public data_type_distconv_adapter<TensorDataType> {
 public:
  using TensorDevType = typename data_type_distconv_adapter<TensorDataType>::TensorDevType;

  leaky_relu_distconv_adapter(Layer& layer): data_type_distconv_adapter<TensorDataType>(layer) {}
  virtual ~leaky_relu_distconv_adapter() = default;

  void setup_distributions(tensor_overlap_constraints &constraints) override;
  void setup_layer(size_t workspace_capacity) override;

  std::unique_ptr<dc::LeakyReLU> m_leaky_relu;
};
#endif // LBANN_HAS_DISTCONV

/** @brief
 *
 *  @f[
 *    \text{LeakyReLU}(x; \alpha) =
 *      \begin{cases}
 *        x        & x > 0 \\
 *        \alpha x & x \leq 0
 *      \end{cases}
 *  @f]
 *  See:
 *
 *  Andrew L. Maas, Awni Y. Hannun, and Andrew Y. Ng. "Rectifier
 *  nonlinearities improve neural network acoustic models." In
 *  Proc. ICML, vol. 30, no. 1, p. 3. 2013.
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class leaky_relu_layer : public data_type_layer<TensorDataType> {
public:
  leaky_relu_layer(lbann_comm *comm, TensorDataType negative_slope = 0.01)
    : data_type_layer<TensorDataType>(comm), m_negative_slope(negative_slope) {}
  leaky_relu_layer* copy() const override { return new leaky_relu_layer(*this); }
  std::string get_type() const override { return "leaky ReLU"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

  description get_description() const override {
    auto desc = data_type_layer<TensorDataType>::get_description();
    desc.add("Negative slope", m_negative_slope);
    return desc;
  }

protected:
  void setup_dims(DataReaderMetaData& dr_metadata) override {
    data_type_layer<TensorDataType>::setup_dims(dr_metadata);
    this->set_output_dims(this->get_input_dims());
  }
  void fp_compute() override;
  void bp_compute() override;

private:
  /** Function slope in negative region. */
  TensorDataType m_negative_slope;

#ifdef LBANN_HAS_DISTCONV
 protected:
  bool is_distconv_supported() const override {
    return Device == El::Device::GPU && Layout == data_layout::DATA_PARALLEL;
  }
  void setup_distconv_adapter() override {
    this->get_distconv_adapter_ptr() = make_unique<leaky_relu_distconv_adapter<
      TensorDataType, Layout, Device>>(*this);
  }
  leaky_relu_distconv_adapter<TensorDataType, Layout, Device>& get_distconv_adapter() override;
  const leaky_relu_distconv_adapter<TensorDataType, Layout, Device>& get_distconv_adapter() const override;
#endif // LBANN_HAS_DISTCONV
};

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
leaky_relu_distconv_adapter<TensorDataType, T_layout, Dev>&
leaky_relu_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter() {
  return const_cast<leaky_relu_distconv_adapter<TensorDataType, T_layout, Dev>&>(
      static_cast<const leaky_relu_layer<TensorDataType, T_layout, Dev>&>(*this).get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
const leaky_relu_distconv_adapter<TensorDataType, T_layout, Dev>&
leaky_relu_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter() const {
  return dynamic_cast<const leaky_relu_distconv_adapter<TensorDataType, T_layout, Dev>&>(
      data_type_layer<TensorDataType>::get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void leaky_relu_distconv_adapter<TensorDataType, T_layout, Dev>::
setup_distributions(tensor_overlap_constraints &constraints) {
  data_type_distconv_adapter<TensorDataType>::setup_distributions(
      constraints);

  auto &x = this->get_prev_activations_dist();
  auto &y = this->get_activations_dist();
  auto &dx = this->get_error_signals_dist();
  auto &dy = this->get_prev_error_signals_dist();

  // x == y
  constraints.mark_equivalent(x, y);
  // x == dx
  constraints.mark_equivalent(x, dx);
  // dx == dy
  constraints.mark_equivalent(dx, dy);
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void leaky_relu_distconv_adapter<TensorDataType, T_layout, Dev>::setup_layer(
    size_t workspace_capacity) {
  m_leaky_relu = make_unique<dc::LeakyReLU>(dc::get_backend());
}
#endif // LBANN_HAS_DISTCONV

#ifndef LBANN_LEAKY_RELU_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device) \
  extern template class leaky_relu_layer<T, data_layout::DATA_PARALLEL, Device>; \
  extern template class leaky_relu_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_LEAKY_RELU_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_ACTIVATIONS_LEAKY_RELU_HPP_INCLUDED
