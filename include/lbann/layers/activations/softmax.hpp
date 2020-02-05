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

#ifndef LBANN_LAYERS_ACTIVATIONS_SOFTMAX_HPP_INCLUDED
#define LBANN_LAYERS_ACTIVATIONS_SOFTMAX_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/cudnn.hpp"

// Threshold outputs to a minimum value.

// If enabled, the minimum output value is sqrt(min), where min is the
// minimum, normalized, positive value (~1e-19 for float and ~1e-154
// for double). During backprop, gradients are computed as if
// thresholding did not occur, so there will be a discrepancy for
// values that are thresholded.
#define LBANN_ENABLE_SOFTMAX_THRESHOLD

namespace lbann {

/** @brief Which tensor dimensions to apply softmax over. */
enum class softmax_mode {
  INVALID,
  /** @brief Sample-wise softmax.
   *
   *  Slice tensor along the sample dimension (assuming data in NCHW
   *  format) and apply softmax independently to each slice (once per
   *  sample).
   */
  INSTANCE,
  /** @brief Position-wise softmax.
   *
   *  Split tensor along all but the channel dimension (assuming data
   *  in NCHW format) and apply softmax independently to each piece
   *  (once per spatial position per sample).
   *
   *  This is not to be confused with @c channelwise_softmax, which
   *  slices along the sample and channel dimensions.
   */
  CHANNEL
};

/**
 *  @f[ \text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}} @f]
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class softmax_layer : public data_type_layer<TensorDataType> {
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  ///@}

public:

  softmax_layer(lbann_comm *comm,
                softmax_mode mode)
    : data_type_layer<TensorDataType>(comm),
      m_mode(mode)
#ifdef LBANN_HAS_CUDNN
    , m_tensors_cudnn_desc(this)
#endif // LBANN_HAS_CUDNN
  {
    if(mode == softmax_mode::INVALID) {
      LBANN_ERROR("invalid softmax mode");
    }
  }

  softmax_layer(const softmax_layer& other)
    : data_type_layer<TensorDataType>(other),
      m_mode(other.m_mode),
      m_workspace(other.m_workspace ?
                  other.m_workspace->Copy() : nullptr)
#ifdef LBANN_HAS_CUDNN
    , m_tensors_cudnn_desc(other.m_tensors_cudnn_desc)
#endif // LBANN_HAS_CUDNN
  {
#ifdef LBANN_HAS_CUDNN
    m_tensors_cudnn_desc.set_layer(this);
#endif // LBANN_HAS_CUDNN
  }

  ~softmax_layer() = default;

  softmax_layer* copy() const override { return new softmax_layer(*this); }
  std::string get_type() const override { return "softmax"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

  void setup_dims() override {
    data_type_layer<TensorDataType>::setup_dims();
    this->set_output_dims(this->get_input_dims());
  }

  void setup_matrices(const El::Grid& grid) override {
    data_type_layer<TensorDataType>::setup_matrices(grid);
    auto dist = this->get_prev_activations().DistData();
    dist.colDist = El::STAR;
    m_workspace.reset(AbsDistMatrixType::Instantiate(dist));
#ifdef HYDROGEN_HAVE_CUB
    if (m_workspace->GetLocalDevice() == El::Device::GPU) {
      m_workspace->Matrix().SetMemoryMode(1); // CUB memory pool
    }
#endif // HYDROGEN_HAVE_CUB
  }

  void fp_setup_outputs(El::Int mini_batch_size) override {
    data_type_layer<TensorDataType>::fp_setup_outputs(mini_batch_size);
    const auto& dist_data = this->get_prev_activations().DistData();
    m_workspace->Empty(false);
    m_workspace->AlignWith(dist_data);
    m_workspace->Resize(1, mini_batch_size);
  }

  void fp_compute() override;
  void bp_compute() override;

  template <typename U>
  friend void fp_compute_impl(softmax_layer<U, Layout, Device>& l);
  template <typename U>
  friend void bp_compute_impl(softmax_layer<U, Layout, Device>& l);

private:

  /** Softmax mode. */
  const softmax_mode m_mode;

  /** Workspace for column-wise reductions. */
  std::unique_ptr<AbsDistMatrixType> m_workspace;

#ifdef LBANN_HAS_CUDNN
  /** Tensor cuDNN descriptors. */
  cudnn::data_parallel_layer_tensor_manager<TensorDataType> m_tensors_cudnn_desc;
#endif // LBANN_HAS_CUDNN

// Minimum output value to avoid denormalized floats
#ifdef LBANN_ENABLE_SOFTMAX_THRESHOLD
  const TensorDataType threshold_val = static_cast<TensorDataType>(El::Sqrt(std::numeric_limits<TensorDataType>::min()));
#else
  const TensorDataType threshold_val = El::TypeTraits<TensorDataType>::Zero();
#endif // LBANN_ENABLE_SOFTMAX_THRESHOLD

};

LBANN_DEFINE_LAYER_BUILDER(softmax);

#ifndef LBANN_SOFTMAX_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device) \
  extern template class softmax_layer<T, data_layout::DATA_PARALLEL, Device>; \
  extern template class softmax_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_SOFTMAX_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_ACTIVATIONS_SOFTMAX_HPP_INCLUDED
