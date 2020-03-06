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
//
// lbann_file_io .hpp .cpp - Input / output utilities
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_PERSIST_H
#define LBANN_PERSIST_H

#include "lbann/base.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/enum_iterator.hpp"
#include "El.hpp"
#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <sstream>

namespace lbann {

enum class persist_type {
  train, // data should be saved in file with train data
  model, // data should be saved in file with model data
  metrics,
  validate,
  testing,
  prediction_context,
  training_context,
  testing_context,
  validation_context,
};

using persist_type_iterator = enum_iterator<persist_type, persist_type::train, persist_type::validation_context>;

inline persist_type execution_mode_to_persist_type(execution_mode m) {
  switch(m) {
  case execution_mode::training:
    return persist_type::training_context;
  case execution_mode::validation:
    return persist_type::validation_context;
  case execution_mode::testing:
    return persist_type::testing_context;
  case execution_mode::prediction:
    return persist_type::prediction_context;
  // case execution_mode::tournament:
  //   return persist_type::tournament;
  case execution_mode::invalid:
  default:
    LBANN_ERROR("Invalid execution mode specified");
  }
}

inline std::string to_string(persist_type pt) {
  switch(pt) {
  case persist_type::model:
    return "model";
  case persist_type::metrics:
    return "metrics";
  case persist_type::train:
    return "train";
  case persist_type::validate:
    return "validate";
  case persist_type::testing:
    return "test";
  case persist_type::prediction_context:
    return "prediction";
  case persist_type::training_context:
    return "training";
  case persist_type::validation_context:
    return "validation";
  case persist_type::testing_context:
    return "testing";
  default:
      LBANN_ERROR("Invalid persist type specified");
  }
}

/// @todo Fix the callback types to properly track execution phases
enum class callback_type {
  model_only,
  execution_context_only,
  full_checkpoint,
  invalid
};

class persist {
 private:
  std::map<persist_type, uint64_t> m_bytes;
  std::map<persist_type, int> m_FDs;
  std::map<persist_type, std::string> m_filenames;
  callback_type ckpt_type;
 public:
  std::string m_checkpoint_dir;

 public:
  persist();
  ~persist() {};

  callback_type get_cb_type() const {
    return ckpt_type;
  }

  void set_cb_type(callback_type type){
    ckpt_type = type;
  }

  void open_checkpoint(const std::string& dir);
  void close_checkpoint();

  void open_restart(const std::string& dir);
  void close_restart();

  uint64_t get_bytes() const {
    uint64_t bytes = 0;
    for(auto& pt : m_bytes) {
      bytes += pt.second;
    }
    return bytes;
  }

  void reset_bytes() {
    for(auto& pt : m_bytes) {
      pt.second = 0;
    }
  }

  template <typename TensorDataType>
  bool write_rank_distmat(persist_type type, const char *name, const El::AbstractDistMatrix<TensorDataType>& M);
  template <typename TensorDataType>
  bool read_rank_distmat(persist_type type, const char *name, El::AbstractDistMatrix<TensorDataType>& M);

  template <typename TensorDataType>
  bool write_distmat(persist_type type, const char *name, El::AbstractDistMatrix<TensorDataType> *M);
  template <typename TensorDataType>
  bool read_distmat (persist_type type, const char *name, El::AbstractDistMatrix<TensorDataType> *M);

  bool write_bytes(persist_type type, const char *name, const void *buf, size_t size);
  bool read_bytes(persist_type type, const char *name, void *buf, size_t size);

  bool write_uint32(persist_type type, const char *name, uint32_t  val);
  bool read_uint32 (persist_type type, const char *name, uint32_t *val);

  bool write_uint64(persist_type type, const char *name, uint64_t  val);
  bool read_uint64 (persist_type type, const char *name, uint64_t *val);

  bool write_int32_contig(persist_type type, const char *name, const int32_t *buf, uint64_t count);
  bool read_int32_contig (persist_type type, const char *name, int32_t *buf, uint64_t count);

  bool write_float(persist_type type, const char *name, float  val);
  bool read_float (persist_type type, const char *name, float *val);

  bool write_string(persist_type type, const char *name, const char *val, int str_length);
  bool read_string (persist_type type, const char *name, char *val, int str_length);

  bool write_double(persist_type type, const char *name, double  val);
  bool read_double (persist_type type, const char *name, double *val);

  template <typename TensorDataType>
  bool write_datatype(persist_type type, const char *name, TensorDataType  val);
  template <typename TensorDataType>
  bool read_datatype (persist_type type, const char *name, TensorDataType *val);

  std::string get_filename(persist_type type) const;
 private:
  int get_fd(persist_type type) const;
};

template <typename TensorDataType>
bool write_distmat(int fd, const char *name, DistMatDT<TensorDataType> *M, uint64_t *bytes);
template <typename TensorDataType>
bool read_distmat (int fd, const char *name, DistMatDT<TensorDataType> *M, uint64_t *bytes);

bool write_bytes(int fd, const char *name, const void *buf, size_t size);
bool read_bytes(int fd, const char *name, void *buf, size_t size);

bool write_uint32(int fd, const char *name, uint32_t  val);
bool read_uint32 (int fd, const char *name, uint32_t *val);

bool write_uint64(int fd, const char *name, uint64_t  val);
bool read_uint64 (int fd, const char *name, uint64_t *val);

bool write_int32_contig(int fd, const char *name, const int32_t *buf, uint64_t count);
bool read_int32_contig (int fd, const char *name, int32_t *buf, uint64_t count);

bool write_float(int fd, const char *name, float  val);
bool read_float (int fd, const char *name, float *val);

bool write_double(int fd, const char *name, double  val);
bool read_double (int fd, const char *name, double *val);

bool write_string(int fd, const char *name, const char *buf, size_t size);
bool read_string(int fd, const char *name, char *buf, size_t size);

class NonexistentArchiveFile : public std::runtime_error {
public:
  NonexistentArchiveFile(std::string const& filename) : std::runtime_error(std::string("Archive file not found: ") + filename) {}
};

template <typename C>
void write_cereal_archive(C& obj, const std::string& filename) {
  std::ofstream os(filename);
  if(!os.is_open()) {
    throw NonexistentArchiveFile(filename);
  }
  cereal::XMLOutputArchive archive(os);
  archive(obj);
}

template <typename C>
void write_cereal_archive(C& obj, persist& p, persist_type pt, const std::string& suffix) {
  write_cereal_archive<C>(obj, p.get_filename(pt) + suffix);
}

template <typename C>
void write_cereal_archive(C& obj, persist& p, execution_mode mode, const std::string& suffix) {
  const persist_type pt = execution_mode_to_persist_type(mode);
  write_cereal_archive<C>(obj, p, pt, suffix);
}

template <typename C>
void read_cereal_archive(C& obj, const std::string& filename) {
  std::ifstream is(filename);
  if(!is.is_open()) {
    throw NonexistentArchiveFile(filename);
  }
  cereal::XMLInputArchive archive(is);
  archive(obj);
}

template <typename C>
void read_cereal_archive(C& obj, persist& p, persist_type pt, const std::string& suffix) {
  read_cereal_archive(obj, p.get_filename(pt) + suffix);
}

template <typename C>
void read_cereal_archive(C& obj, persist& p, execution_mode mode, const std::string& suffix) {
  const persist_type pt = execution_mode_to_persist_type(mode);
  read_cereal_archive<C>(obj, p, pt, suffix);
}

template <typename C>
std::string create_cereal_archive_binary_string(C& obj) {
  std::ostringstream ss;
  {
    cereal::BinaryOutputArchive archive(ss);
    archive(obj);
  } // archive goes out of scope, ensuring all contents are flushed
  return ss.str();
}

template <typename C>
void unpack_cereal_archive_binary_string(C& obj, const std::string& buf) {
  std::istringstream ss(buf);
  {
    cereal::BinaryInputArchive archive(ss);
    archive(obj);
  } // archive goes out of scope, ensuring all contents are flushed
}

template <typename C>
void load_from_shared_cereal_archive(C& obj, persist& p, persist_type pt,
                                     lbann_comm& comm,
                                     const std::string& suffix) {
  std::string buf;
  if (comm.am_trainer_master()) {
    read_cereal_archive<C>(obj, p, pt, suffix);
    buf = create_cereal_archive_binary_string<C>(obj);
  }else {
    // If you are not the trainer master, still check to see if the file exists
    std::ifstream is(p.get_filename(pt) + suffix);
    if(!is.is_open()) {
      throw NonexistentArchiveFile(p.get_filename(pt) + suffix);
    }
  }

  // TODO: this assumes homogeneous processors
  // broadcast state from rank 0
  comm.trainer_broadcast(0, buf);

  if (!comm.am_trainer_master()) {
    unpack_cereal_archive_binary_string<C>(obj, buf);
  }
}

template <typename C>
void load_from_shared_cereal_archive(C& obj, persist& p, execution_mode mode,
                                     lbann_comm& comm,
                                     const std::string& suffix) {
  const persist_type pt = execution_mode_to_persist_type(mode);
  load_from_shared_cereal_archive<C>(obj, p, pt, comm, suffix);
}

#ifndef LBANN_PERSIST_INSTANTIATE
#define PROTO(T)                                                            \
  extern template bool persist::write_rank_distmat<T>(                      \
  persist_type type, const char *name, const El::AbstractDistMatrix<T>& M); \
  extern template bool persist::read_rank_distmat<T>(                       \
  persist_type type, const char *name, El::AbstractDistMatrix<T>& M);       \
  extern template bool persist::write_distmat<T>(                           \
  persist_type type, const char *name, El::AbstractDistMatrix<T> *M);       \
  extern template bool persist::read_distmat<T>(                            \
  persist_type type, const char *name, El::AbstractDistMatrix<T> *M);       \
  extern template bool persist::write_datatype<T>(                          \
    persist_type type, const char *name, T val);                            \
  extern template bool persist::read_datatype<T>(                           \
    persist_type type, const char *name, T *val);                           \
  extern template bool write_distmat<T>(                                    \
    int fd, const char *name, DistMatDT<T> *M, uint64_t *bytes);            \
  extern template bool read_distmat<T>(                                     \
    int fd, const char *name, DistMatDT<T> *M, uint64_t *bytes)

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#undef LBANN_INSTANTIATE_CPU_HALF
#undef LBANN_INSTANTIATE_GPU_HALF
#endif // LBANN_PERSIST_INSTANTIATE

} // namespace lbann

#endif // LBANN_PERSIST_H
