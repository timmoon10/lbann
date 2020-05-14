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
#include "lbann/utils/distconv.hpp"

namespace lbann {

enum class batch_normalization_stats_aggregation {
  /** Statistics are aggregated only within a single rank. */
  local,
  /** Statistics are aggregated among every rank in a single node. */
  node_local,
  /** Statistics are aggregated among every rank in the model. */
  global
};

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class batch_normalization_distconv_adapter: public data_type_distconv_adapter<TensorDataType> {
 public:
  using TensorDevType = typename data_type_distconv_adapter<TensorDataType>::TensorDevType;
  batch_normalization_distconv_adapter(Layer& layer):
      data_type_distconv_adapter<TensorDataType>(layer) {}
  virtual ~batch_normalization_distconv_adapter() = default;
  void setup_fp_tensors() override;
  void setup_bp_tensors() override;
  dc::Shape get_per_channel_stat_shape() const;
  dc::Dist get_per_channel_stat_dist(const dc::Dist &input_dist) const;
  void setup_layer(size_t workspace_capacity) override;
  void fp_compute();
  void bp_compute();

  TensorDevType m_mean;
  TensorDevType m_var;
  TensorDevType m_scale;
  TensorDevType m_bias;
  TensorDevType m_running_mean;
  TensorDevType m_running_var;
  TensorDevType m_mean_gradient;
  TensorDevType m_var_gradient;
  TensorDevType m_scale_gradient;
  TensorDevType m_bias_gradient;
  std::unique_ptr<dc::BatchNormalization<TensorDataType>> m_bn;
};
#endif // LBANN_HAS_DISTCONV

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

  void setup_dims(DataReaderMetaData& dr_metadata) override {
    regularizer_layer<TensorDataType>::setup_dims(dr_metadata);
    this->set_output_dims(this->get_input_dims());
  }

  void setup_data(size_t max_mini_batch_size) override {
    regularizer_layer<TensorDataType>::setup_data(max_mini_batch_size);
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

#ifdef LBANN_HAS_DISTCONV
  friend class batch_normalization_distconv_adapter<TensorDataType, T_layout, Dev>;
 protected:
  bool is_distconv_supported() const override {
    return Dev == El::Device::GPU && T_layout == data_layout::DATA_PARALLEL;
  }
  void setup_distconv_adapter() override {
    this->get_distconv_adapter_ptr() = make_unique<
      batch_normalization_distconv_adapter<TensorDataType, T_layout, Dev>>(*this);
  }
  batch_normalization_distconv_adapter<TensorDataType, T_layout, Dev>& get_distconv_adapter() override;
  const batch_normalization_distconv_adapter<TensorDataType, T_layout, Dev>& get_distconv_adapter() const override;
#endif // LBANN_HAS_DISTCONV
};

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
const batch_normalization_distconv_adapter<TensorDataType, T_layout, Dev>&
batch_normalization_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter() const {
  return dynamic_cast<const batch_normalization_distconv_adapter<
    TensorDataType, T_layout, Dev>&>(data_type_layer<TensorDataType>::get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
batch_normalization_distconv_adapter<TensorDataType, T_layout, Dev>&
batch_normalization_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter() {
  return const_cast<batch_normalization_distconv_adapter<TensorDataType, T_layout, Dev>&>(
      static_cast<const batch_normalization_layer<TensorDataType, T_layout, Dev>&>(*this).get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
dc::Shape batch_normalization_distconv_adapter<TensorDataType, T_layout, Dev>::
get_per_channel_stat_shape() const {
  auto &l = dynamic_cast<const batch_normalization_layer<
    TensorDataType, T_layout, Dev>&>(this->layer());
  const int num_channels = this->get_activations_shape()[dc::get_channel_dim()];
  // Sanity check that the shared tensors have the correct shape
  assert_ne(num_channels, 0);
  assert_eq(l.m_mean_and_var->Matrix().Width() *
            l.m_mean_and_var->Matrix().Height(),
            num_channels * 2);
  dc::Shape per_channel_stat_shape(dc::get_num_dims(l), 1);
  per_channel_stat_shape[dc::get_channel_dim()] = num_channels;
  return per_channel_stat_shape;
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
dc::Dist batch_normalization_distconv_adapter<TensorDataType, T_layout, Dev>::
get_per_channel_stat_dist(const dc::Dist &input_dist) const {
  auto shared_dist = dc::Dist::make_distribution(
      input_dist.get_locale_shape());
  auto split_shape = input_dist.get_split_shape();
  // set all dimensions to be 1 except for the channel dimension
  auto pc = split_shape[-2];
  // set all elements to 1
  split_shape = 1;
  split_shape[-2] = pc;
  shared_dist.set_split_shape(split_shape);

  return shared_dist;
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void batch_normalization_distconv_adapter<TensorDataType, T_layout, Dev>::
setup_fp_tensors() {
  data_type_distconv_adapter<TensorDataType>::setup_fp_tensors();

  auto &l = static_cast<batch_normalization_layer<
    TensorDataType, T_layout, Dev>&>(this->layer());
  const auto &input_dist = this->get_prev_activations_dist();

  const auto per_channel_stat_shape = get_per_channel_stat_shape();
  const auto shared_dist = get_per_channel_stat_dist(input_dist);

  const dc::LocaleMPI loc(dc::get_mpi_comm(), false);

  // mean
  m_mean = TensorDevType(per_channel_stat_shape, loc, shared_dist);
  assert0(dc::tensor::View(m_mean, l.m_mean_v->Buffer()));
  // var
  m_var = TensorDevType(per_channel_stat_shape, loc, shared_dist);
  assert0(dc::tensor::View(m_var, l.m_var_v->Buffer()));
  // scale: view to weights[0]
  m_scale = TensorDevType(per_channel_stat_shape, loc, shared_dist);
  // bias: view to weights[1]
  m_bias = TensorDevType(per_channel_stat_shape, loc, shared_dist);
  // running_mean: view to weights[2]
  m_running_mean = TensorDevType(per_channel_stat_shape, loc, shared_dist);
  // running_var: view to weights[3]
  m_running_var = TensorDevType(per_channel_stat_shape, loc, shared_dist);
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void batch_normalization_distconv_adapter<TensorDataType, T_layout, Dev>::
setup_bp_tensors() {
  data_type_distconv_adapter<TensorDataType>::setup_bp_tensors();

  const auto &prev_error_signal_dist = this->get_prev_error_signals_dist();
  auto &l = static_cast<batch_normalization_layer<
    TensorDataType, T_layout, Dev>&>(this->layer());

  const auto per_channel_stat_shape = get_per_channel_stat_shape();
  const auto shared_dist = get_per_channel_stat_dist(
      prev_error_signal_dist);

  const dc::LocaleMPI loc(dc::get_mpi_comm(), false);

  // scale_gradient
  m_scale_gradient = TensorDevType(per_channel_stat_shape, loc, shared_dist);
  assert0(dc::tensor::View(
      m_scale_gradient, l.m_scale_gradient->Buffer()));
  // bias_gradient
  m_bias_gradient = TensorDevType(per_channel_stat_shape, loc, shared_dist);
  assert0(dc::tensor::View(
      m_bias_gradient, l.m_bias_gradient->Buffer()));
  // mean_gradient
  m_mean_gradient = TensorDevType(per_channel_stat_shape, loc, shared_dist);
  assert0(dc::tensor::View(
      m_mean_gradient, l.m_mean_gradient_v->Buffer()));
  // var_gradient
  m_var_gradient = TensorDevType(per_channel_stat_shape, loc, shared_dist);
  assert0(dc::tensor::View(
      m_var_gradient, l.m_var_gradient_v->Buffer()));
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void batch_normalization_distconv_adapter<TensorDataType, T_layout, Dev>::setup_layer(
    size_t workspace_capacity) {
  auto &l = dynamic_cast<batch_normalization_layer<
    TensorDataType, T_layout, Dev>&>(this->layer());
  bool global_stats;
  if (l.m_statistics_group_size  == 0) {
    global_stats = true;
  } else if (l.m_statistics_group_size == 1) {
    global_stats = false;
  } else {
    LBANN_ERROR("statistics_group_size must be either 0 or 1 for now.");
  }

  m_bn = make_unique<dc::BatchNormalization<TensorDataType>>(
      dc::get_backend(), dc::get_num_dims(l),
      l.m_decay, l.m_epsilon, global_stats);
}
#endif // LBANN_HAS_DISTCONV

#ifndef LBANN_BATCH_NORMALIZATION_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device) \
  extern template class batch_normalization_layer<T, data_layout::DATA_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_BATCH_NORMALIZATION_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_HPP_INCLUDED
