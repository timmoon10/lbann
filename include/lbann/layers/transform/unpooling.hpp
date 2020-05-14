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

#ifndef LBANN_LAYER_UNPOOLING_HPP_INCLUDED
#define LBANN_LAYER_UNPOOLING_HPP_INCLUDED

#include <vector>
#include "lbann/layers/transform/pooling.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/im2col.hpp"

namespace lbann {

/** @brief Transpose of pooling layer.
 *  @todo GPU support.
 */
template <typename TensorDataType, data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class unpooling_layer : public transform_layer<TensorDataType> {
  static_assert(T_layout == data_layout::DATA_PARALLEL,
                "unpooling only supports DATA_PARALLEL");
  static_assert(Dev == El::Device::CPU,
                "unpooling only supports CPU");
 private:

  /** Corresponding pooling layer. */
  pooling_layer<TensorDataType, T_layout, Dev>* m_pooling_layer;

 public:

  unpooling_layer(lbann_comm *comm,
                  pooling_layer<TensorDataType, T_layout, Dev>* pool = nullptr)
    : transform_layer<TensorDataType>(comm),
      m_pooling_layer(pool) { }

  unpooling_layer* copy() const override { return new unpooling_layer(*this); }
  std::string get_type() const override { return "unpooling"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  void setup_pointers() override {
    // Check that pooling layer is valid
    if(m_pooling_layer == nullptr) {
      throw lbann_exception("unpooling_layer: no paired pooling layer");
    }
    if(m_pooling_layer->m_pool_mode != pool_mode::max) {
      throw lbann_exception("unpooling_layer: currently only max unpooling layer is implemented");
    }
    if(m_pooling_layer->using_gpus()) {
      throw lbann_exception("unpooling_layer: GPU version not yet implemented");
    }
  }

  void setup_dims(DataReaderMetaData& dr_metadata) override {
    transform_layer<TensorDataType>::setup_dims(dr_metadata);

    // Check that input tensor is valid
    const auto& input_dims = this->get_input_dims();
    const auto& pool_output_dims = m_pooling_layer->get_output_dims();
    if (input_dims != pool_output_dims) {
      std::stringstream err;
      err << get_type() << " layer \"" << this->get_name() << "\" "
          << "expects input tensors with dimensions ";
      for (size_t i = 0; i < pool_output_dims.size(); ++i) {
        err << (i > 0 ? " x " : "") << pool_output_dims[i];
      }
      err << ", but parent layer "
          << "\"" << this->get_parent_layers()[0]->get_name() << "\" "
          << "outputs with dimensions ";
      for (size_t i = 0; i < input_dims.size(); ++i) {
        err << (i > 0 ? " x " : "") << input_dims[i];
      }
      LBANN_ERROR(err.str());
    }

    // Initialize output tensor based on corresponding pooling layer
    this->set_output_dims(m_pooling_layer->get_input_dims());

  }

  void set_pooling_layer(pooling_layer<TensorDataType, T_layout, Dev>* pool) {
    m_pooling_layer = pool;
  }

  std::vector<Layer*> get_layer_pointers() override {
    std::vector<Layer*> layers = transform_layer<TensorDataType>::get_layer_pointers();
    layers.push_back((Layer*) m_pooling_layer);
    return layers;
  }

  void set_layer_pointers(std::vector<Layer*> layers) override {
    m_pooling_layer = dynamic_cast<pooling_layer<TensorDataType, T_layout, Dev>*>(layers.back());
    if (m_pooling_layer == nullptr) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__
          << " :: unpooling_layer: invalid layer pointer used to set paired pooling layer";
      throw lbann_exception(err.str());
    }
    layers.pop_back();
    transform_layer<TensorDataType>::set_layer_pointers(layers);
  }

  protected:

  void fp_compute() override {
    if(this->using_gpus()) {
      throw lbann_exception("unpooling_layer: GPU version not yet implemented");
    } else {
      fp_compute_im2col();
    }
  }

  void bp_compute() override {
    if(this->using_gpus()) {
      throw lbann_exception("unpooling_layer: GPU version not yet implemented");
    } else {
      bp_compute_im2col();
    }
  }

 private:

  /// Unpooling forward propagation with im2col
  void fp_compute_im2col() {

    using DMatDT = El::Matrix<TensorDataType, Dev>;

    // Get local matrices
    const DMatDT& prev_activations_local = this->get_local_prev_activations();
    DMatDT& activations_local = this->get_local_activations();

    // Get parameters
    const int local_width = prev_activations_local.Width();
    const auto& output_dims = this->get_output_dims();
    const int num_channels = output_dims[0];
    const int num_per_input_channel = this->get_input_size() / num_channels;
    const int pool_size = m_pooling_layer->m_pool_size;

    // Initialize im2col matrix
    DMatDT im2col_mat(pool_size * num_channels, num_per_input_channel);

    // Iterate through data samples
    for(int sample = 0; sample < local_width; ++sample) {

      // Clear im2col matrix
      El::Zero(im2col_mat);

      // Populate im2col matrix
      const TensorDataType *prev_activations_buffer
        = prev_activations_local.LockedBuffer(0, sample);
      const int *indices_buffer
        = &m_pooling_layer->m_max_pool_indices[sample * this->get_input_size()];
      LBANN_OMP_PARALLEL_FOR
      for(int channel = 0; channel < num_channels; ++channel) {
        for(int j = 0; j < num_per_input_channel; ++j) {
          const int input_index = j + channel * num_per_input_channel;
          const int max_index = indices_buffer[input_index];
          TensorDataType *im2col_buffer
            = im2col_mat.Buffer(channel * pool_size, j);
          im2col_buffer[max_index]
            = prev_activations_buffer[input_index];
        }
      }

      // Convert im2col matrix to output matrix
      DMatDT output_mat =
        El::View(activations_local, El::ALL, El::IR(sample));
      col2im<TensorDataType>(im2col_mat,
             output_mat,
             num_channels,
             output_dims.size() - 1,
             &output_dims[1],
             m_pooling_layer->m_pads.data(),
             m_pooling_layer->m_pool_dims.data(),
             m_pooling_layer->m_strides.data(),
             [](TensorDataType const& a, TensorDataType const& b) {
               return std::max(a, b);
             });
    }
  }

  /// Unpooling backward propagation with im2col
  void bp_compute_im2col() {

    using DMatDT = El::Matrix<TensorDataType, Dev>;

    // Get local matrices
    const DMatDT& prev_error_signal_local = this->get_local_prev_error_signals();
    DMatDT& error_signal_local = this->get_local_error_signals();

    // Get parameters
    const int local_width = prev_error_signal_local.Width();
    const auto& output_dims = this->get_output_dims();
    const int num_channels = output_dims[0];
    const int num_per_output_channel = this->get_input_size() / num_channels;
    const int pool_size = m_pooling_layer->m_pool_size;

    // Initialize im2col matrix
    DMatDT im2col_mat(pool_size * num_channels, num_per_output_channel);

    // Iterate through data samples
    for(int sample = 0; sample < local_width; ++sample) {

      // Construct im2col matrix from input
      const DMatDT& input_mat = El::LockedView(prev_error_signal_local,
                                               El::ALL, El::IR(sample));
      im2col<TensorDataType>(input_mat,
             im2col_mat,
             num_channels,
             output_dims.size() - 1,
             &output_dims[1],
             m_pooling_layer->m_pads.data(),
             m_pooling_layer->m_pool_dims.data(),
             m_pooling_layer->m_strides.data());

      // Propagate error signal based on pooling layer
      TensorDataType *output_buffer = error_signal_local.Buffer(0, sample);
      const int *indices_buffer
        = &m_pooling_layer->m_max_pool_indices[sample * this->get_input_size()];
      LBANN_OMP_PARALLEL_FOR
      for(int channel = 0; channel < num_channels; ++channel) {
        for(int j = 0; j < num_per_output_channel; ++j) {
          const int output_index = j + channel * num_per_output_channel;
          const int max_index = indices_buffer[output_index];
          TensorDataType *im2col_buffer
            = im2col_mat.Buffer(channel * pool_size, j);
          output_buffer[output_index] = im2col_buffer[max_index];
        }
      }

    }

  }

};

#ifndef LBANN_UNPOOLING_LAYER_INSTANTIATE
#define PROTO(T)                           \
  extern template class unpooling_layer<T, data_layout::DATA_PARALLEL, El::Device::CPU>

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#undef LBANN_INSTANTIATE_CPU_HALF
#endif // LBANN_UNPOOLING_LAYER_INSTANTIATE

}  // namespace lbann

#endif  // LBANN_LAYER_UNPOOLING_HPP_INCLUDED
