////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
//
// lbann_optimizer_adam .hpp .cpp - SGD with Adam
// Reference:
// Kingma, D. and Ba, J. 2014. Adam: A Method for Stochastic Optimization.
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_OPTIMIZER_ADAM_HPP
#define LBANN_OPTIMIZER_ADAM_HPP

#include "lbann/optimizers/lbann_optimizer.hpp"

namespace lbann {

/// Adam optimizer
class adam : public optimizer {
 public:
  /// Constructor
  adam
  (lbann_comm *comm,
   DataType learning_rate,
   DataType beta1 = DataType(0.9),
   DataType beta2 = DataType(0.99),
   DataType eps = DataType(1e-8));
  adam(const adam& other);
  /// Destructor
  ~adam();
  /// Set parameters to optimize and initialize optimizer
  void setup(AbsDistMat *parameters);
  /// Update parameters using objective function gradient
  void update(const AbsDistMat *gradient);
 private:
  /// Update factor for first moment estimate
  DataType m_beta1;
  /// Update factor for second moment estimate
  DataType m_beta2;
  /// Small factor to avoid division by zero
  DataType m_eps;
  /// beta1 ^ iteration
  DataType m_current_beta1;
  /// beta2 ^ iteration
  DataType m_current_beta2;
  /// First moment estimates
  AbsDistMat *m_moment1;
  /// Second moment estimates
  AbsDistMat *m_moment2;
};

/// Factory for Adam optimizer
class adam_factory : public optimizer_factory {
 public:
  /// Constructor
  adam_factory
  (lbann_comm *comm,
   DataType learning_rate,
   DataType beta1 = DataType(0.9),
   DataType beta2 = DataType(0.99),
   DataType eps = DataType(1e-8));
  /// Destructor
  virtual ~adam_factory();
  /// Create Adam optimizer
  optimizer *create_optimizer();
 private:
  /// Learning rate
  DataType m_learning_rate;
  /// Update factor for first moment estimate
  DataType m_beta1;
  /// Update factor for second moment estimate
  DataType m_beta2;
  /// Small factor to avoid division by zero
  DataType m_eps;
};

} // namespace lbann

#endif  // LBANN_OPTIMIZER_ADAM_HPP
