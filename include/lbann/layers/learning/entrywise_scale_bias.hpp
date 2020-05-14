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

#ifndef LBANN_LAYER_LEARNING_ENTRYWISE_SCALE_BIAS_HPP_INCLUDED
#define LBANN_LAYER_LEARNING_ENTRYWISE_SCALE_BIAS_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/** @brief Apply scale and bias to tensor entries.
 *
 *  Scale and bias terms are applied independently to each tensor
 *  entry. More precisely, given input, output, scale, and bias
 *  tensors @f$ X,Y,A,B\in\mathbb{R}^{d_1\times\cdots\times d_n} @f$:
 *  @f[
 *    Y = A \circ X + B
 *  @f]
 *
 *  The scale and bias terms are fused into a single weights tensor to
 *  reduce the number of gradient allreduces during backprop. In
 *  particular, the weights tensor is a
 *  @f$ \text{size} \times 2 @f$ matrix, where the first
 *  column correspond to scale terms and the second column to bias
 *  terms.
 */
template <typename TensorDataType,
          data_layout Layout = data_layout::DATA_PARALLEL,
          El::Device Device = El::Device::CPU>
class entrywise_scale_bias_layer : public data_type_layer<TensorDataType> {
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  /** @brief The concrete weights type used by this object. */
  using WeightsType = data_type_weights<TensorDataType>;

  /** @brief The concrete optimizer type used by this object. */
  using OptimizerType = data_type_optimizer<TensorDataType>;

  ///@}

public:

  entrywise_scale_bias_layer(lbann_comm *comm)
    : data_type_layer<TensorDataType>(comm) {}

  entrywise_scale_bias_layer(const entrywise_scale_bias_layer& other)
    : data_type_layer<TensorDataType>(other),
      m_weights_gradient(other.m_weights_gradient ?
                         other.m_weights_gradient->Copy() : nullptr) {}
  entrywise_scale_bias_layer& operator=(const entrywise_scale_bias_layer& other) {
    data_type_layer<TensorDataType>::operator=(other);
    m_weights_gradient.reset(other.m_weights_gradient ?
                             other.m_weights_gradient->Copy() :
                             nullptr);
    return *this;
  }

  entrywise_scale_bias_layer* copy() const override {
    return new entrywise_scale_bias_layer(*this);
  }
  std::string get_type() const override { return "entry-wise scale/bias"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

  void setup_matrices(const El::Grid& grid) override {
    data_type_layer<TensorDataType>::setup_matrices(grid);
    auto dist = this->get_prev_activations().DistData();
    dist.rowDist = El::STAR;
    m_weights_gradient.reset(AbsDistMatrixType::Instantiate(dist));
  }

  void setup_data(size_t max_mini_batch_size) override {
    data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);

    // Initialize output dimensions
    this->set_output_dims(this->get_input_dims());
    const auto output_dims = this->get_output_dims();
    const El::Int output_size = this->get_output_size();

    // Construct default weights if needed
    // Note: Scale is initialized to 1 and bias to 0
    if (!this->has_weights()) {
      auto w = make_unique<WeightsType>(this->get_comm());
      std::vector<TensorDataType> vals(2*output_size, El::TypeTraits<TensorDataType>::Zero());
      std::fill(vals.begin(), vals.begin()+output_size, El::TypeTraits<TensorDataType>::One());
      auto init = make_unique<value_initializer<TensorDataType>>(vals);
      auto opt = this->m_model->template create_optimizer<TensorDataType>();
      w->set_name(this->get_name() + "_weights");
      w->set_initializer(std::move(init));
      w->set_optimizer(std::move(opt));
      this->add_weights(w.get());
      this->m_model->add_weights(std::move(w));
    }
    if (this->num_weights() != 1) {
      LBANN_ERROR("attempted to setup ",
                  this->get_type()," layer \"",this->get_name(),"\" ",
                  "with an invalid number of weights ",
                  "(expected 1, found ",this->num_weights(),")");
    }

    // Setup weights
    auto dist = this->get_prev_activations().DistData();
    dist.rowDist = El::STAR;
    this->get_data_type_weights(0).set_dims(output_dims,
                                     {static_cast<int>(2)});
    this->get_data_type_weights(0).set_matrix_distribution(dist);

    // Setup gradient w.r.t. weights
    m_weights_gradient->AlignWith(dist);
    m_weights_gradient->Resize(output_size, 2);

  }

  void fp_setup_outputs(El::Int mini_batch_size) override {
    data_type_layer<TensorDataType>::fp_setup_outputs(mini_batch_size);

#if 0 /// @todo See https://github.com/LLNL/lbann/issues/1123

    // Check that input and weights tensors are aligned
    /// @todo Realign weights tensor if misaligned
    bool aligned = true;
    try {
      const auto& x = this->get_prev_activations();
      const auto& w = m_weights[0]->get_values();
      aligned = (x.ColAlign() == w.ColAlign()
                 && x.RowAlign() == w.RowAlign());
    }
    catch (const exception& e) {
      // An exception is thrown if you try accessing weights values
      // before they are initialized. We don't care if this case is
      // aligned, so it's safe to ignore.
    }
    if (!aligned) {
      std::ostringstream err;
      err << this->get_type() << " layer \"" << this->get_name() << "\" "
          << "has misaligned input and weights matrices";
      LBANN_ERROR(err.str());
    }

#endif // 0

  }

  void bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) override {
    data_type_layer<TensorDataType>::bp_setup_gradient_wrt_inputs(mini_batch_size);
    m_weights_gradient->Empty(false);
    m_weights_gradient->AlignWith(this->get_prev_activations());
    m_weights_gradient->Resize(this->get_input_size(), 2);
  }

protected:
  void fp_compute() override;
  void bp_compute() override;

private:

  /** Objective function gradient w.r.t. weights. */
  std::unique_ptr<AbsDistMatrixType> m_weights_gradient;

};

LBANN_DEFINE_LAYER_BUILDER(entrywise_scale_bias);

#ifndef LBANN_ENTRYWISE_SCALE_BIAS_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device)                     \
  extern template class entrywise_scale_bias_layer< \
    T, data_layout::DATA_PARALLEL, Device>;         \
  extern template class entrywise_scale_bias_layer< \
    T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_ENTRYWISE_SCALE_BIAS_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_LEARNING_ENTRYWISE_SCALE_BIAS_HPP_INCLUDED
