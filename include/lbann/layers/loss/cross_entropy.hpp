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

#ifndef LBANN_LAYERS_LOSS_CROSS_ENTROPY_HPP_INCLUDED
#define LBANN_LAYERS_LOSS_CROSS_ENTROPY_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"

namespace lbann {

/** @brief Cross entropy loss function.
 *
 *  Given a predicted distribution @f$y@f$ and ground truth
 *  distribution @f$\hat{y}@f$,
 *  @f[ CE(y,\hat{y}) = - \sum\limits_{i} \hat{y}_i \log y_i @f]
 */
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class cross_entropy_layer : public data_type_layer<TensorDataType> {
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  ///@}

public:

  cross_entropy_layer(lbann_comm *comm) : data_type_layer<TensorDataType>(comm) {
    this->m_expected_num_parent_layers = 2;
  }

  cross_entropy_layer(const cross_entropy_layer& other)
    : data_type_layer<TensorDataType>(other) {
    m_workspace.reset(other.m_workspace ?
                      other.m_workspace->Copy() :
                      nullptr);
  }

  cross_entropy_layer& operator=(const cross_entropy_layer& other) {
    data_type_layer<TensorDataType>::operator=(other);
    m_workspace.reset(other.m_workspace ?
                      other.m_workspace->Copy() :
                      nullptr);
    return *this;
  }

  cross_entropy_layer* copy() const override { return new cross_entropy_layer(*this); }
  std::string get_type() const override { return "cross entropy"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  void setup_dims() override {
    data_type_layer<TensorDataType>::setup_dims();
    this->set_output_dims({1});

    // Check that input dimensions match
    if (this->get_input_dims(0) != this->get_input_dims(1)) {
      const auto& parents = this->get_parent_layers();
      std::stringstream err;
      err << get_type() << " layer \"" << this->get_name() << "\" "
          << "has input tensors with different dimensions (";
      for (int i = 0; i < this->get_num_parents(); ++i) {
        const auto& dims = this->get_input_dims(i);
        err << (i > 0 ? ", " : "")
            << "layer \"" << parents[i]->get_name() << "\" outputs ";
        for (size_t j = 0; j < dims.size(); ++j) {
          err << (j > 0 ? " x " : "") << dims[j];
        }
      }
      err << ")";
      LBANN_ERROR(err.str());
    }

  }

  void setup_data() override {
    data_type_layer<TensorDataType>::setup_data();

    // Initialize workspace
    const auto& prediction = this->get_prev_activations(0);
    switch (this->get_data_layout()) {
    case data_layout::DATA_PARALLEL:
      m_workspace.reset(new StarVCMatDT<TensorDataType, Dev>(
                          prediction.Grid(),
                          prediction.Root()));
      break;
    case data_layout::MODEL_PARALLEL:
      m_workspace.reset(new StarMRMatDT<TensorDataType, Dev>(
                          prediction.Grid(),
                          prediction.Root()));
      break;
    default: LBANN_ERROR("invalid data layout");
    }
#ifdef HYDROGEN_HAVE_CUB
    if (m_workspace->GetLocalDevice() == El::Device::GPU) {
      m_workspace->Matrix().SetMemoryMode(1); // CUB memory pool
    }
#endif // HYDROGEN_HAVE_CUB

  }

  void fp_compute() override {

    // Initialize workspace
    const auto& prediction = this->get_prev_activations(0);
    m_workspace->AlignWith(prediction.DistData());
    m_workspace->Resize(1, prediction.Width());

    // Compute local contributions and accumulate
    /// @todo Consider reduce rather than allreduce
    local_fp_compute();
    this->get_comm()->allreduce(*m_workspace, m_workspace->RedundantComm());
    El::Copy(*m_workspace, this->get_activations());

  }

  void bp_compute() override {

    // Initialize workspace
    const auto& prediction = this->get_prev_activations(0);
    m_workspace->AlignWith(prediction.DistData());
    El::Copy(this->get_prev_error_signals(), *m_workspace);

    // Compute local gradients
    local_bp_compute();

  }

private:

  /** Compute local contributions to cross entropy loss. */
  void local_fp_compute();
  /** Compute local gradients. */
  void local_bp_compute();

  /** Workspace matrix. */
  std::unique_ptr<AbsDistMatrixType> m_workspace;

};

#ifndef LBANN_CROSS_ENTROPY_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device)              \
  extern template class cross_entropy_layer< \
    T, data_layout::DATA_PARALLEL, Device>;  \
  extern template class cross_entropy_layer< \
    T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_CROSS_ENTROPY_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_LOSS_CROSS_ENTROPY_HPP_INCLUDED
