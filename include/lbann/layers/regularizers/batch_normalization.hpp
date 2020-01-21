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

#ifndef LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_HPP_INCLUDED
#define LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_HPP_INCLUDED

#include "lbann/layers/regularizers/regularizer.hpp"
#include "lbann/models/model.hpp"

namespace lbann {

enum class batch_normalization_stats_aggregation {
  /** Statistics are aggregated only within a single rank. */
  local,
  /** Statistics are aggregated among every rank in a single node. */
  node_local,
  /** Statistics are aggregated among every rank in the model. */
  global
};

/** @brief
 *
 *  Each input channel is normalized across the mini-batch to have
 *  zero mean and unit standard deviation. Learned scaling factors and
 *  biases are then applied. This uses the standard approach of
 *  maintaining the running mean and standard deviation (with
 *  exponential decay) for use at test time. See:
 *
 *  Sergey Ioffe and Christian Szegedy. "Batch Normalization:
 *  Accelerating Deep Network Training by Reducing Internal Covariate
 *  Shift." In International Conference on Machine Learning,
 *  pp. 448-456. 2015.
 */
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class batch_normalization_layer : public regularizer_layer<TensorDataType> {
  static_assert(T_layout == data_layout::DATA_PARALLEL,
                "batch normalization only supports DATA_PARALLEL");
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

private:

  /** Decay rate for the running statistics. */
  TensorDataType m_decay;
  /** Small number to avoid division by zero. */
  TensorDataType m_epsilon;
  /** @brief Size of group to aggregate statistics over.
   *
   * If this is 1, the group consists of one process and aggregation
   * is local. If it is 0, statistics are aggregated globally.
   */
  int m_statistics_group_size;
  /**
   * Cache of node-local num_per_sum results for node-local stats.
   * Indexed by effective mini-batch size.
   */
  std::unordered_map<El::Int, El::Int> m_num_per_sum_cache;

  /** @brief Current minibatch means and standard deviations.
   *
   * These are fused for performance when doing non-local batchnorm.
   */
  std::unique_ptr<AbsDistMatrixType> m_mean_and_var;
  /** View of current mini-batch means. */
  std::unique_ptr<AbsDistMatrixType> m_mean_v;
  /** View of current mini-batch standard deviations. */
  std::unique_ptr<AbsDistMatrixType> m_var_v;
  /** @brief Gradients w.r.t. means and standard deviations.
   *
   * These are fused for performance when doing non-local batchnorm.
   */
  std::unique_ptr<AbsDistMatrixType> m_mean_and_var_gradient;
  /** View of gradient w.r.t. means. */
  std::unique_ptr<AbsDistMatrixType> m_mean_gradient_v;
  /** View of gradient w.r.t. standard deviations. */
  std::unique_ptr<AbsDistMatrixType> m_var_gradient_v;
  /** Gradient w.r.t. scaling terms. */
  std::unique_ptr<AbsDistMatrixType> m_scale_gradient;
  /** Gradient w.r.t. bias terms. */
  std::unique_ptr<AbsDistMatrixType> m_bias_gradient;

public:
  /** @brief Set up batch normalization.
   *
   *  @param comm The communication context for this layer
   *  @param decay Controls the momentum of the running mean/standard
   *         deviation averages.
   *  @param epsilon A small number to avoid division by zero.
   *  @param statistics_group_size Number of processors to aggregate
   *         statistics over. Defaults to 1 (i.e. local aggregation).
   */
  batch_normalization_layer(lbann_comm *comm,
                            TensorDataType decay=0.9,
                            TensorDataType epsilon=1e-5,
                            int statistics_group_size=1)
    : regularizer_layer<TensorDataType>(comm),
      m_decay(decay),
      m_epsilon(epsilon),
      m_statistics_group_size(statistics_group_size) {
#ifdef LBANN_DETERMINISTIC
    // Force global computation.
    m_statistics_group_size = 0;
#endif
  }

  batch_normalization_layer(const batch_normalization_layer& other)
    : regularizer_layer<TensorDataType>(other),
      m_decay(other.m_decay),
      m_epsilon(other.m_epsilon),
      m_statistics_group_size(other.m_statistics_group_size),
      m_num_per_sum_cache(other.m_num_per_sum_cache),
      m_mean_and_var(other.m_mean_and_var ?
                     other.m_mean_and_var->Copy() : nullptr),
      m_mean_v(other.m_mean_v ? other.m_mean_v->Copy() : nullptr),
      m_var_v(other.m_var_v ? other.m_var_v->Copy() : nullptr),
      m_mean_and_var_gradient(other.m_mean_and_var_gradient ?
                              other.m_mean_and_var_gradient->Copy() : nullptr),
      m_mean_gradient_v(other.m_mean_gradient_v ?
                        other.m_mean_gradient_v->Copy() : nullptr),
      m_var_gradient_v(other.m_var_gradient_v ?
                       other.m_var_gradient_v->Copy() : nullptr),
      m_scale_gradient(other.m_scale_gradient ?
                       other.m_scale_gradient->Copy() : nullptr),
      m_bias_gradient(other.m_bias_gradient ?
                      other.m_bias_gradient->Copy() : nullptr) {}

  batch_normalization_layer& operator=(const batch_normalization_layer& other) {
    regularizer_layer<TensorDataType>::operator=(other);
    m_decay = other.m_decay;
    m_epsilon = other.m_epsilon;
    m_statistics_group_size = other.m_statistics_group_size;
    m_num_per_sum_cache = other.m_num_per_sum_cache;

    // Deep copy matrices
    m_mean_and_var.reset(other.m_mean_and_var ?
                         other.m_mean_and_var->Copy() : nullptr);
    m_mean_v.reset(other.m_mean_v ?
                   other.m_mean_v->Copy() : nullptr);
    m_var_v.reset(other.m_var_v ?
                  other.m_var_v->Copy() : nullptr);
    m_mean_and_var_gradient.reset(other.m_mean_and_var_gradient ?
                                  other.m_mean_and_var_gradient->Copy() : nullptr);
    m_mean_gradient_v.reset(other.m_mean_gradient_v ?
                            other.m_mean_gradient_v->Copy() : nullptr);
    m_var_gradient_v.reset(other.m_var_gradient_v ?
                           other.m_var_gradient_v->Copy() : nullptr);
    m_scale_gradient.reset(other.m_scale_gradient ?
                           other.m_scale_gradient->Copy() : nullptr);
    m_bias_gradient.reset(other.m_bias_gradient ?
                          other.m_bias_gradient->Copy() : nullptr);

    return *this;
  }

  batch_normalization_layer* copy() const override { return new batch_normalization_layer(*this); }
  std::string get_type() const override { return "batch normalization"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  description get_description() const override {
    auto desc = regularizer_layer<TensorDataType>::get_description();
    desc.add("Decay", m_decay);
    desc.add("Epsilon", m_epsilon);
    desc.add("Statistics group size", m_statistics_group_size);
    return desc;
  }

protected:

  void setup_matrices(const El::Grid& grid) override {
    regularizer_layer<TensorDataType>::setup_matrices(grid);
    m_mean_and_var.reset(new StarMatDT<TensorDataType, Dev>(grid));
    m_mean_v.reset(new StarMatDT<TensorDataType, Dev>(grid));
    m_var_v.reset(new StarMatDT<TensorDataType, Dev>(grid));
    m_mean_and_var_gradient.reset(new StarMatDT<TensorDataType, Dev>(grid));
    m_mean_gradient_v.reset(new StarMatDT<TensorDataType, Dev>(grid));
    m_var_gradient_v.reset(new StarMatDT<TensorDataType, Dev>(grid));
    m_scale_gradient.reset(new StarMatDT<TensorDataType, Dev>(grid));
    m_bias_gradient.reset(new StarMatDT<TensorDataType, Dev>(grid));
  }

  void setup_dims() override {
    regularizer_layer<TensorDataType>::setup_dims();
    this->set_output_dims(this->get_input_dims());
  }

  void setup_data() override {
    regularizer_layer<TensorDataType>::setup_data();
    const auto& output_dims = this->get_output_dims();
    const auto& num_channels = output_dims[0];

    // Display warning if mini-batch size is small
    const auto& output = this->get_activations();
    const auto& mini_batch_size = output.Width();
    const auto& local_mini_batch_size = mini_batch_size / output.DistSize();
    if (m_statistics_group_size == 0 && mini_batch_size <= 4) {
      if (output.DistRank() == 0) {
        std::stringstream err;
        err << "LBANN warning: "
            << get_type() << " layer \"" << this->get_name() << "\" "
            << "is using global statistics and "
            << "the mini-batch size (" << mini_batch_size << ") "
            << "may be too small to get good statistics";
        std::cerr << err.str() << std::endl;
      }
    } else if (m_statistics_group_size != 0 &&
               m_statistics_group_size*local_mini_batch_size <= 4) {
      // This possibly underestimates the aggregation size for processors with
      // smaller local mini-batch sizes.
      if (output.DistRank() == 0) {
        std::stringstream err;
      err << "LBANN warning: "
          << get_type() << " layer \"" << this->get_name() << "\" "
          << "is aggregating statistics over "
          << m_statistics_group_size
          << "processors and the aggregated mini-batch size ("
          << (m_statistics_group_size*local_mini_batch_size) << ") "
          << "may be too small to get good statistics";
        std::cerr << err.str() << std::endl;
      }
    }

    // Initialize default weights if none are provided
    if (this->num_weights() > 4) {
      std::stringstream err;
      err << "attempted to setup layer \"" << this->m_name << "\" "
          << "with an invalid number of weights";
      LBANN_ERROR(err.str());
    }
    this->set_num_data_type_weights(4);
    if (!this->has_data_type_weights(0)) {
      auto w = make_unique<WeightsType>(this->get_comm());
      auto init = make_unique<constant_initializer<TensorDataType>>(El::TypeTraits<TensorDataType>::One());
      auto opt = this->m_model->template create_optimizer<TensorDataType>();
      w->set_name(this->get_name() + "_scale");
      w->set_initializer(std::move(init));
      w->set_optimizer(std::move(opt));
      this->set_data_type_weights(0, w.get());
      this->m_model->add_weights(std::move(w));
    }
    if (!this->has_data_type_weights(1)) {
      auto w = make_unique<WeightsType>(this->get_comm());
      auto init = make_unique<constant_initializer<TensorDataType>>(El::TypeTraits<TensorDataType>::Zero());
      auto opt = this->m_model->template create_optimizer<TensorDataType>();
      w->set_name(this->get_name() + "_bias");
      w->set_initializer(std::move(init));
      w->set_optimizer(std::move(opt));
      this->set_data_type_weights(1, w.get());
      this->m_model->add_weights(std::move(w));
    }
    if (!this->has_data_type_weights(2)) {
      auto w = make_unique<WeightsType>(this->get_comm());
      auto init = make_unique<constant_initializer<TensorDataType>>(El::TypeTraits<TensorDataType>::Zero());
      w->set_name(this->get_name() + "_running_mean");
      w->set_initializer(std::move(init));
      this->set_data_type_weights(2, w.get());
      this->m_model->add_weights(std::move(w));
    }
    if (!this->has_data_type_weights(3)) {
      auto w = make_unique<WeightsType>(this->get_comm());
      auto init = make_unique<constant_initializer<TensorDataType>>(El::TypeTraits<TensorDataType>::One());
      w->set_name(this->get_name() + "_running_variance");
      w->set_initializer(std::move(init));
      this->set_data_type_weights(3, w.get());
      this->m_model->add_weights(std::move(w));
    }

    // Setup weights
    auto dist = this->get_prev_activations().DistData();
    dist.colDist = El::STAR;
    dist.rowDist = El::STAR;
    for (auto* w : this->get_data_type_weights()) {
      w->set_dims(num_channels);
      w->set_matrix_distribution(dist);
    }

    // Initialize matrices
    El::Zeros(*m_mean_and_var,   num_channels, 2);
    El::Zeros(*m_mean_and_var_gradient, num_channels, 2);
    El::Zeros(*m_scale_gradient, num_channels, 1);
    El::Zeros(*m_bias_gradient,  num_channels, 1);

    // Initialize views.
    El::View(*m_mean_v, *m_mean_and_var, El::ALL, El::IR(0, 1));
    El::View(*m_var_v, *m_mean_and_var, El::ALL, El::IR(1, 2));
    El::View(*m_mean_gradient_v, *m_mean_and_var_gradient,
             El::ALL, El::IR(0, 1));
    El::View(*m_var_gradient_v, *m_mean_and_var_gradient,
             El::ALL, El::IR(1, 2));

    // Initialize freeze state
    for (auto&& w : this->get_data_type_weights()) {
      if (this->m_frozen) {
        w->freeze();
      } else {
        w->unfreeze();
      }
    }
    for (auto&& w : this->get_data_type_weights()) {
      if (w->is_frozen() != this->m_frozen) {
        std::stringstream err;
        err << (this->m_frozen ? "" : "un") << "frozen "
            << "layer \"" << this->get_name() << "\" has "
            << (w->is_frozen() ? "" : "un") << "frozen "
            << "weights \"" << w->get_name() << "\"";
        LBANN_ERROR(err.str());
      }
    }

  }

  void fp_compute() override;
  void bp_compute() override;

};

#ifndef LBANN_BATCH_NORMALIZATION_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device) \
  extern template class batch_normalization_layer<T, data_layout::DATA_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_BATCH_NORMALIZATION_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_HPP_INCLUDED
