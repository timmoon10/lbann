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

#ifndef LBANN_LAYERS_MISC_VARIANCE_HPP_INCLUDED
#define LBANN_LAYERS_MISC_VARIANCE_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"

namespace lbann {

/** @brief Estimate variance.
 *
 *  Given an input @f$x@f$ with empirical mean @f$\bar{x}@f$, an
 *  unbiased estimator for the variance is given by
 *  @f[
 *    \sigma_{x}^2
 *      \approx \frac{1}{n-1} \sum\limits_{i=1}^{n} (x - \bar{x})^2
 *  @f]
 *  Scaling by @f$ 1/n @f$ instead of @f$ 1/(n-1) @f$ is a biased
 *  estimator.
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class variance_layer : public data_type_layer<TensorDataType> {
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  ///@}

public:

  variance_layer(lbann_comm *comm, bool biased)
    : data_type_layer<TensorDataType>(comm), m_biased(biased) {}
  variance_layer(const variance_layer& other)
    : data_type_layer<TensorDataType>(other),
      m_biased(other.m_biased),
      m_means(other.m_means ? other.m_means->Copy() : nullptr),
      m_workspace(other.m_workspace ?
                  other.m_workspace->Copy() : nullptr) {}
  variance_layer& operator=(const variance_layer& other) {
    data_type_layer<TensorDataType>::operator=(other);
    m_biased = other.m_biased;
    m_means.reset(other.m_means ? other.m_means->Copy() : nullptr);
    m_workspace.reset(other.m_workspace ?
                      other.m_workspace->Copy() : nullptr);
    return *this;
  }

  variance_layer* copy() const override { return new variance_layer(*this); }
  std::string get_type() const override { return "variance"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

  description get_description() const override {
    auto desc = data_type_layer<TensorDataType>::get_description();
    desc.add("Biased", m_biased);
    return desc;
  }

protected:

  void setup_matrices(const El::Grid& grid) override {
    data_type_layer<TensorDataType>::setup_matrices(grid);
    auto dist_data = this->get_prev_activations().DistData();
    dist_data.colDist = El::STAR;
    m_means.reset(AbsDistMatrixType::Instantiate(dist_data));
    m_workspace.reset(AbsDistMatrixType::Instantiate(dist_data));
  }

  void setup_dims(DataReaderMetaData& dr_metadata) override {
    data_type_layer<TensorDataType>::setup_dims(dr_metadata);
    this->set_output_dims({1});
    if (this->get_input_size() <= 1) {
      std::stringstream err;
      const auto& parents = this->get_parent_layers();
      const auto& dims = this->get_input_dims();
      err << get_type() << " layer \"" << this->get_name() << "\" "
          << "expects an input tensor with at least two entries, "
          << "but parent layer \"" << parents[0]->get_name() << "\" "
          << "outputs a tensor with dimensions ";
      for (size_t i = 0; i < dims.size(); ++i) {
        err << (i > 0 ? " x " : "") << dims[i];
      }
      LBANN_ERROR(err.str());
    }
  }

  void fp_compute() override;
  void bp_compute() override;

private:

  /** Whether to use biased variance estimator. */
  bool m_biased;

  /** Means for each mini-batch sample.  */
  std::unique_ptr<AbsDistMatrixType> m_means;
  /** Workspace. */
  std::unique_ptr<AbsDistMatrixType> m_workspace;

};

#ifndef LBANN_VARIANCE_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device) \
  extern template class variance_layer<T, data_layout::DATA_PARALLEL, Device>; \
  extern template class variance_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_VARIANCE_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_MISC_VARIANCE_HPP_INCLUDED
