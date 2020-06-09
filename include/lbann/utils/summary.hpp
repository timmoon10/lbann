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

// lbann_summary - Write summary statistics to Tensorboard

#ifndef LBANN_SUMMARY_HPP_INCLUDED
#define LBANN_SUMMARY_HPP_INCLUDED

#include <string>
#include <vector>
#include "lbann/base.hpp"
#include "lbann/comm.hpp"

#ifdef LBANN_HAS_TBINF
#include "TBinf.hpp"
#endif

namespace lbann {

template <typename T, typename U>
using BiggerOf = typename std::conditional<(sizeof(T) > sizeof(U)), T, U>::type;

#ifdef LBANN_HAS_TBINF

/**
 * Interface for computing summary statistics within and among models and
 * outputting them to Tensorboard.
 * All methods take a step parameter that gives the global step to use when
 * reporting the output.
 * All methods buffer data locally. The flush method must be called to write
 * summaries out, and must be called by every LBANN process.
 * Be sure to add summaries in the same order on every process, or confusion
 * will result.
 * Distributed matrices should be distributed by model.
 * This class automatically prepends "modelN/" to each tag. The tag is only
 * relevant at the world master process.
 *
 * @note WHEN YOU UPDATE THE PUBLIC API HERE, REMEMBER TO UPDATE THE KLUDGE FOR
 * NON-TENSORBOARD BUILDS BELOW!
 */
class lbann_summary {
public:

  /**
   * Create a new summary manager.
   * @param logdir The directory to output events to.
   * @param comm Communicator to use.
   */
  lbann_summary(std::string logdir, lbann_comm *comm);
  ~lbann_summary();

  /** Report the mean of mat. */
  template <typename TensorDataType>
  void reduce_mean(const std::string tag, const El::AbstractDistMatrix<TensorDataType>& mat, int step);
  /** Report the minimum value of mat. */
  template <typename TensorDataType>
  void reduce_min(const std::string tag, const El::AbstractDistMatrix<TensorDataType>& mat, int step);
  /** Report the maximum value of mat. */
  template <typename TensorDataType>
  void reduce_max(const std::string tag, const El::AbstractDistMatrix<TensorDataType>& mat, int step);
  /** Report the standard deviation of mat. */
  template <typename TensorDataType>
  void reduce_stdev(const std::string tag, const El::AbstractDistMatrix<TensorDataType>& mat, int step);
  /** Report a scalar from each model (only meaningful on model masters). */
  template <typename TensorDataType>
  void reduce_scalar(const std::string tag, TensorDataType s, int step);
  /** Do a trainer_reduce (sum) on a scalar, then report that sum. */
  template <typename TensorDataType>
  void sum_reduce_scalar(const std::string tag, TensorDataType s, int step);
  /** Report a scalar from every rank. */
  template <typename TensorDataType>
  void reduce_scalar_all(const std::string tag, TensorDataType s, int step);
  /** Report a histogram of the values in mat. */
  template <typename TensorDataType>
  void reduce_histogram(const std::string tag, const El::AbstractDistMatrix<TensorDataType>& mat, int step);
  /** Report the (squared) 2-norm of mat. */
  template <typename TensorDataType>
  void reduce_2norm(const std::string tag, const El::AbstractDistMatrix<TensorDataType>& mat, int step);
  void report_image(std::string const& /*tag*/,
                    std::string const& /*img_format*/,
                    CPUMat const& /*image*/,
                    std::vector<int> const& /*dims*/,
                    int /*step*/);
  /**
   * Write all summaries out.
   * @todo This can be made faster by combining collective operations element-
   * wise for each type of summary.
   */
  void flush();

 private:
  lbann_comm *m_comm;
  TBinf::SummaryWriter *m_sw;

  /** Represent a pending summary operation.
    * Note that TensorBoard takes scalars as floats
    */
  struct pending_op {
    pending_op(const std::string tag_, int step_, float local_,
               float local2_ = 0.0f, int num_ = 0) :
      tag(tag_), step(step_), local(local_), local2(local2_), num(num_) {}
    /** Associated tag. */
    const std::string tag;
    /** Global step. */
    int step;
    /** Locally-computed data. */
    float local;
    /** More locally-computed data (for stdev). */
    float local2;
    /** Size of matrix (needed for mean/stdev). */
    int num;
  };
  /** Represent a pending histogram operation.
    * Note that TensorBoard takes histograms as double
    */
  struct pending_histogram {
    pending_histogram(const std::string tag_, int step_,
                      std::vector<double> buckets_,
                      double min_, double max_, double num_,
                      double sum_, double sqsum_) :
      tag(tag_), step(step_), buckets(buckets_), min(min_), max(max_),
      num(num_), sum(sum_), sqsum(sqsum_) {}
    /** Associated tag. */
    const std::string tag;
    /** Global step. */
    int step;
    /** Histogram buckets, using histogram_buckets as the limits. */
    std::vector<double> buckets;
    /** Minimum value in the data. */
    double min;
    /** Maximum value in the data. */
    double max;
    /** Number of values in the data. */
    double num;
    /** Sum of the values in the data. */
    double sum;
    /** Sum of the squares of the values in the data. */
    double sqsum;
  };

  /** Currently-pending reduce_means. */
  std::vector<pending_op> m_pending_means;
  /** Currently-pending reduce_mins. */
  std::vector<pending_op> m_pending_mins;
  /** Currently-pending reduce_maxes. */
  std::vector<pending_op> m_pending_maxes;
  /** Currently-pending reduce_stdevs. */
  std::vector<pending_op> m_pending_stdevs;
  /** Currently-pending reduce_scalars. */
  std::vector<pending_op> m_pending_scalars;
  /** Currently-pending sum_reduce_scalars. */
  std::vector<pending_op> m_pending_sum_scalars;
  /** Currently-pending reduce_scalar_alls. */
  std::vector<pending_op> m_pending_scalar_alls;
  /** Buckets for histograms. */
  std::vector<double> m_histogram_buckets;
  /** Currently-pending reduce_histograms. */
  std::vector<pending_histogram> m_pending_histograms;

  /** Execute all pending mean operations. */
  void flush_means();
  /** Execute all pending min operations. */
  void flush_mins();
  /** Execute all pending max operations. */
  void flush_maxes();
  /** Execute all pending stdev operations. */
  void flush_stdevs();
  /** Execute all pending scalar operations. */
  void flush_scalars();
  /** Execute all pending sum-scalar operations. */
  void flush_sum_scalars();
  /** Execute all pending scalar-all operations. */
  void flush_scalar_alls();
  /** Execute all pending histogram operations. */
  void flush_histograms();

  /** Compute the sum of elements in mat. */
  template <typename TensorDataType>
  auto local_sum(const El::AbstractMatrix<TensorDataType>& mat) const -> BiggerOf<TensorDataType, float>;
  /** Compute the sum of square of elements in mat. */
  template <typename TensorDataType, typename AccumT>
  void local_sum_sqsum(const El::AbstractMatrix<TensorDataType>& mat, AccumT& sum, AccumT& sqsum) const;
  /** Compute the minimum element in mat. */
  template <typename TensorDataType>
  auto local_min(const El::AbstractMatrix<TensorDataType>& mat) const -> BiggerOf<TensorDataType, float>;
  /** Compute the maximum element in mat. */
  template <typename TensorDataType>
  auto local_max(const El::AbstractMatrix<TensorDataType>& mat) const -> BiggerOf<TensorDataType, float>;
  /** Compute the element-wise L2 norm of mat. */
  template <typename TensorDataType>
  auto local_2norm(const El::AbstractMatrix<TensorDataType>& mat) const -> BiggerOf<TensorDataType, float>;
  /** Prepend "model<model>/" to tag. */
  std::string prepend_model(const std::string tag, int model) const;
  /** Gather and write out a scalar summary for each model. */
  void gather_scalar_summary(const std::string tag, float s, int step);
  /** Gather and write out a scalar summary for each entry in a vector. */
  void gather_scalar_summary(const std::vector<pending_op>& ops,
                             std::vector<float>& scalars);
};

#include "lbann/utils/summary_impl.hpp"

#else

/** Dummy class when TBinf is not present. */
class lbann_summary {
 public:
  lbann_summary(std::string logdir, lbann_comm *comm) {}

  void report_image(std::string const& tag,
                    std::string const& img_format,
                    CPUMat const& image,
                    std::vector<int> const& dims,
                    int step)
  {}

  template <typename TensorDataType>
  void reduce_mean(const std::string tag, const El::AbstractDistMatrix<TensorDataType>& mat, int step) {}
  template <typename TensorDataType>
  void reduce_min(const std::string tag, const El::AbstractDistMatrix<TensorDataType>& mat, int step) {}
  template <typename TensorDataType>
  void reduce_max(const std::string tag, const El::AbstractDistMatrix<TensorDataType>& mat, int step) {}
  template <typename TensorDataType>
  void reduce_stdev(const std::string tag, const El::AbstractDistMatrix<TensorDataType>& mat, int step) {}
  template <typename TensorDataType>
  void reduce_scalar(const std::string tag, TensorDataType s, int step) {}
  template <typename TensorDataType>
  void sum_reduce_scalar(const std::string tag, TensorDataType s, int step) {}
  template <typename TensorDataType>
  void reduce_scalar_all(const std::string tag, TensorDataType s, int step) {}
  template <typename TensorDataType>
  void reduce_histogram(const std::string tag, const El::AbstractDistMatrix<TensorDataType>& mat, int step) {}
  template <typename TensorDataType>
  void reduce_2norm(const std::string tag, const El::AbstractDistMatrix<TensorDataType>& mat, int step) {}
  void flush() {}
};

#endif  // LBANN_HAS_TBINF

}  // namespace lbann

#endif  // LBANN_SUMMARY_HPP_INCLUDED
