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

#ifndef LBANN_LAYER_GAUSSIAN_HPP_INCLUDED
#define LBANN_LAYER_GAUSSIAN_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/random.hpp"

namespace lbann {

/** @brief Random values from Gaussian/normal distribution. */
template <typename TensorDataType,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class gaussian_layer : public transform_layer<TensorDataType> {
private:
  /** @brief Gaussian distribution mean. */
  TensorDataType m_mean;
  /** @brief Gaussian distribution standard deviation. */
  TensorDataType m_stdev;
  /** @brief Whether to have deterministic output when not training.
   *
   *  Applies to execution modes other than training, e.g. validation
   *  and inference. If true, outputs are all equal to the
   *  distribution mean when not training.
   */
  bool m_training_only;

public:
  gaussian_layer(lbann_comm *comm,
                 const std::vector<int>& dims,
                 TensorDataType mean = El::TypeTraits<TensorDataType>::Zero(),
                 TensorDataType stdev = El::TypeTraits<TensorDataType>::One(),
                 bool training_only = false)
    : transform_layer<TensorDataType>(comm),
      m_mean(mean), m_stdev(stdev), m_training_only(training_only) {
    this->set_output_dims(dims);
    this->m_expected_num_parent_layers = 0;
  }
  gaussian_layer* copy() const override { return new gaussian_layer(*this); }
  std::string get_type() const override { return "Gaussian"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  description get_description() const override {
    auto desc = transform_layer<TensorDataType>::get_description();
    desc.add("Mean", m_mean);
    desc.add("Standard deviation", m_stdev);
    desc.add("Training only", m_training_only);
    return desc;
  }

protected:

  void fp_compute() override {
    auto& output = this->get_activations();
    const auto& mode = this->m_model->get_execution_context().get_execution_mode();
    if (m_training_only && (mode != execution_mode::training)) {
      El::Fill(output, m_mean);
    }
    else {
      gaussian_fill(output, output.Height(), output.Width(), m_mean, m_stdev);
    }
  }

};

#ifndef LBANN_GAUSSIAN_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device) \
  extern template class gaussian_layer<T, data_layout::DATA_PARALLEL, Device>; \
  extern template class gaussian_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_GAUSSIAN_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_GAUSSIAN_HPP_INCLUDED
