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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_jag_conduit.hpp"
#include "lbann/io/data_buffers/partitioned_io_buffer.hpp"
#include "lbann/data_store/data_store_conduit.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/execution_contexts/sgd_execution_context.hpp"
#include "lbann/utils/lbann_library.hpp"
#include "lbann/utils/image.hpp"
#include "lbann/utils/opencv.hpp"
#include "lbann/transforms/repack_HWC_to_CHW_layout.hpp"
#include "lbann/transforms/scale_and_translate.hpp"

#include "lbann/utils/file_utils.hpp" // for add_delimiter() in load()
#include <limits>     // numeric_limits
#include <algorithm>  // max_element
#include <numeric>    // accumulate
#include <functional> // multiplies
#include <type_traits>// is_same
#include <set>
#include <map>
#include <omp.h>
#include "lbann/utils/timer.hpp"
#include "lbann/utils/glob.hpp"
#include "lbann/utils/peek_map.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_io_hdf5.hpp"


#include <cereal/archives/binary.hpp>
#include <sstream>

// This comes after all the headers, and is only visible within the current implementation file.
// To make sure, we put '#undef _CN_' at the end of this file
#define _CN_ std::string("data_reader_jag_conduit")

namespace lbann {

std::unordered_map<std::string, int> data_reader_jag_conduit::m_num_local_readers;

const std::set<std::string> data_reader_jag_conduit::non_numeric_vars = {
  "fusion_reaction",
  "fusion_model_reaction",
  "radial_profile",
  "postp_timeseries_vars",
  "name",
  "solver",
  "mesh_def",
  "hs_volume_integral",
  "fusion_model_sv",
  "shell_model",
  "shape_model",
  "ablation_cv_model",
  "infalling_model",
  "radiation_model",
  "hotspot_model",
  "shape_model_initial_velocity_amplitude",
  "stopping_model",
  "energy_balance_model_ablation_cv_model",
  "solver_method",
  "conduction_model_conductivity",
  "solver_mode"
};

void data_reader_jag_conduit::set_io_buffer_type(const std::string io_buffer) {
  m_io_buffer_type = io_buffer;
}

void data_reader_jag_conduit::set_local_id(const std::string role) {
  m_local_reader_id = m_num_local_readers[role]++;
}

int data_reader_jag_conduit::get_local_id(const std::string role) const {
  return m_local_reader_id;
}

void data_reader_jag_conduit::set_leading_reader(data_reader_jag_conduit* r) {
  m_leading_reader = r;
}

data_reader_jag_conduit* data_reader_jag_conduit::get_leading_reader() {
  return m_leading_reader;
}

void data_reader_jag_conduit::shuffle_indices(rng_gen& gen) {
  if ((m_leading_reader != this) && (m_leading_reader != nullptr)) {
    m_shuffled_indices = m_leading_reader->get_shuffled_indices();
    return;
  }
  generic_data_reader::shuffle_indices(gen);
  m_sample_list.compute_epochs_file_usage(get_shuffled_indices(), get_mini_batch_size(), *m_comm);
}

int data_reader_jag_conduit::compute_max_num_parallel_readers() {
  if (m_io_buffer_type == "partitioned") {
#if 0
    // @todo BVE FIXME - Why are we doing this here
    set_num_parallel_readers(partitioned_io_buffer<DataType>::compute_max_num_parallel_readers(
                             0, get_mini_batch_size(),
                             get_num_parallel_readers(), get_comm()));
#endif
    set_sample_stride(get_num_parallel_readers());
    set_iteration_stride(1);
  } else {
    LBANN_ERROR(get_type() + ":: unknown io_buffer type: " + m_io_buffer_type);
  }
  return get_num_parallel_readers();
}

bool data_reader_jag_conduit::check_num_parallel_readers(long data_set_size) {
  return true;
}

data_reader_jag_conduit::data_reader_jag_conduit(bool shuffle)
  : generic_data_reader(shuffle) {
  set_defaults();
}

void data_reader_jag_conduit::copy_members(const data_reader_jag_conduit& rhs) {
  m_independent = rhs.m_independent;
  m_independent_groups = rhs.m_independent_groups;
  m_dependent = rhs.m_dependent;
  m_dependent_groups = rhs.m_dependent_groups;
  m_image_width = rhs.m_image_width;
  m_image_height = rhs.m_image_height;
  m_image_num_channels = rhs.m_image_num_channels;
  m_num_img_srcs = rhs.m_num_img_srcs;
  m_split_channels = rhs.m_split_channels;
  set_linearized_image_size();
  m_is_data_loaded = rhs.m_is_data_loaded;
  m_emi_image_keys = rhs.m_emi_image_keys;
  m_scalar_keys = rhs.m_scalar_keys;
  m_input_keys = rhs.m_input_keys;

  m_uniform_input_type = rhs.m_uniform_input_type;

  m_output_scalar_prefix = rhs.m_output_scalar_prefix;
  m_output_image_prefix = rhs.m_output_image_prefix;
  m_input_prefix = rhs.m_input_prefix;

  m_scalar_filter = rhs.m_scalar_filter;
  m_scalar_prefix_filter = rhs.m_scalar_prefix_filter;
  m_input_filter = rhs.m_input_filter;
  m_input_prefix_filter = rhs.m_input_prefix_filter;
  m_io_buffer_type = rhs.m_io_buffer_type;
  m_local_reader_id = rhs.m_local_reader_id;
  //TODO: need  to make sure this is what we want
  m_leading_reader = rhs.m_leading_reader;

  El::Copy(rhs.m_data_cache, m_data_cache);
  El::Copy(rhs.m_response_cache, m_response_cache);
  El::Copy(rhs.m_label_cache, m_label_cache);
  m_cached_data_mb_size = rhs.m_cached_data_mb_size;
  m_cached_response_mb_size = rhs.m_cached_response_mb_size;
  m_cached_label_mb_size = rhs.m_cached_label_mb_size;

  m_image_normalization_params = rhs.m_image_normalization_params;
  m_scalar_normalization_params = rhs.m_scalar_normalization_params;
  m_input_normalization_params = rhs.m_input_normalization_params;

  m_sample_list.copy(rhs.m_sample_list);
  m_list_per_trainer = rhs.m_list_per_trainer;
  m_list_per_model = rhs.m_list_per_model;

  if(rhs.m_data_store != nullptr) {
    m_data_store = new data_store_conduit(rhs.get_data_store());
    m_data_store->set_data_reader_ptr(this);
  }
}

data_reader_jag_conduit::data_reader_jag_conduit(const data_reader_jag_conduit& rhs)
  : generic_data_reader(rhs) {
  copy_members(rhs);
}

data_reader_jag_conduit& data_reader_jag_conduit::operator=(const data_reader_jag_conduit& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }

  generic_data_reader::operator=(rhs);

  copy_members(rhs);

  return (*this);
}

data_reader_jag_conduit::~data_reader_jag_conduit() {
  // if (m_data_store != nullptr) {
  //   delete m_data_store;
  // }
}

void data_reader_jag_conduit::set_defaults() {
  m_data_store = nullptr;
  m_independent.clear();
  m_independent_groups.clear();
  m_dependent.clear();
  m_dependent_groups.clear();
  m_image_width = 0;
  m_image_height = 0;
  m_image_num_channels = 1;
  set_linearized_image_size();
  m_num_img_srcs = 1u;
  m_split_channels = false;
  m_is_data_loaded = false;
  m_num_labels = 0;
  m_emi_image_keys.clear();
  m_scalar_keys.clear();
  m_input_keys.clear();
  m_uniform_input_type = false;
  m_output_scalar_prefix = "";
  m_output_image_prefix = "";
  m_input_prefix = "";
  m_scalar_filter.clear();
  m_scalar_prefix_filter.clear();
  m_input_filter.clear();
  m_input_prefix_filter.clear();
  m_io_buffer_type = "";
  m_local_reader_id = 0;
  m_leading_reader = this;
  m_cached_data_mb_size = 0;
  m_cached_response_mb_size = 0;
  m_cached_label_mb_size = 0;

  m_image_normalization_params.clear();
  m_scalar_normalization_params.clear();
  m_input_normalization_params.clear();

  //m_sample_list.clear();
  m_list_per_trainer = false;
  m_list_per_model = false;
}

void data_reader_jag_conduit::setup(int num_io_threads, observer_ptr<thread_pool> io_thread_pool) {
  generic_data_reader::setup(num_io_threads, io_thread_pool);
}

#ifdef _USE_IO_HANDLE_
bool data_reader_jag_conduit::has_path(const data_reader_jag_conduit::file_handle_t& h,
                                       const std::string& path) const {
  return m_sample_list.is_file_handle_valid(h) && h->has_path(path);
}

void data_reader_jag_conduit::read_node(const data_reader_jag_conduit::file_handle_t& h,
                                        const std::string& path,
                                        conduit::Node& n) const {
  if (!h) {
    return;
  }
  h->read(path, n);
}
#else
bool data_reader_jag_conduit::has_path(const hid_t& h, const std::string& path) const {
  return (m_sample_list.is_file_handle_valid(h) &&
          conduit::relay::io::hdf5_has_path(h, path));
}

void data_reader_jag_conduit::read_node(const hid_t& h, const std::string& path, conduit::Node& n) const {
  conduit::relay::io::hdf5_read(h, path, n);
}
#endif

const conduit::Node& data_reader_jag_conduit::get_conduit_node(const conduit::Node& n_base, const std::string key) {
  return n_base[key];
}

bool data_reader_jag_conduit::load_conduit_node(const size_t i, const std::string& key, conduit::Node& node) const {
  if (m_io_thread_pool != nullptr && m_using_random_node.count(m_io_thread_pool->get_local_thread_id())) {
    LBANN_ERROR("previously retrieved a random conduit node from data_store, so shouldn't be here");
  }

  const sample_t& s = m_sample_list[i];
  const std::string& sample_name = s.second;
  const std::string path = sample_name + key;

  sample_file_id_t id = s.first;
  auto h = m_sample_list.get_samples_file_handle(id);
  if (!has_path(h, path)) {
    const std::string& file_name = m_sample_list.get_samples_filename(id);
    if (m_data_store != nullptr) {
      if (! m_data_store->is_fully_loaded()) {
        const conduit::Node obj = m_data_store->get_random_node();
        node = obj["data"];
        const std::vector<std::string>& child_names = node.child_names();
        const std::string cur_child = child_names[0];
        const std::string new_child = LBANN_DATA_ID_STR(i);
        node.rename_child(cur_child, new_child);
        m_using_random_node.emplace(m_io_thread_pool->get_local_thread_id());
        std::cout << get_type() + ":: replacing with random node, since failed to open file "
                  << file_name << " for sample " << sample_name
                  <<" and key: " << key << "\n";
        return false;
      } else {
        if (!m_sample_list.is_file_handle_valid(h)) {
          LBANN_ERROR("failed to get file handle for file " + file_name);
        } else {
          LBANN_ERROR("got file handle for file " + file_name + \
                      " but the path doesn't exist in the file: " + path);
        }
      }
    }

    // this block fires if we cannot load a conduit node, either from file
    // or from the data_store
    else {
      if (!m_sample_list.is_file_handle_valid(h)) {
        LBANN_ERROR(get_type() + ":: Cannot open file " + file_name + \
                    " in dir: " + m_sample_list.get_samples_dirname() +
                    " for sample "+ sample_name + " ran_in_trainer: " \
                    + std::to_string(m_comm->get_rank_in_trainer()) \
                    + " because we could not get a file handle");
        return false;
      } else {
        LBANN_ERROR(get_type() + ":: Cannot open file " + file_name + \
                    " in dir: " + m_sample_list.get_samples_dirname() +
                    " for sample "+ sample_name + " ran_in_trainer: " \
                    + std::to_string(m_comm->get_rank_in_trainer()) \
                    + " because we could not get a sample from the data_store");
          return false;
      }
    }
  }

  read_node(h, path, node);
  return true;
}

bool data_reader_jag_conduit::has_conduit_path(const size_t i, const std::string& key) const {
  const sample_t& s = m_sample_list[i];
  sample_file_id_t id = s.first;
  const std::string& sample_name = s.second;
  const auto h = m_sample_list.get_samples_file_handle(id);
  const std::string path = sample_name + key;
  if (!has_path(h, path)) {
    const std::string& file_name = m_sample_list.get_samples_filename(id);
    LBANN_ERROR(get_type() +  ":: Cannot open file " + file_name + \
                " for sample "+ sample_name);
    return false;
  }

  return true;
}


void data_reader_jag_conduit::set_independent_variable_type(
  const std::vector< std::vector<data_reader_jag_conduit::variable_t> >& independent) {
  m_independent_groups = independent;
  m_independent.clear();

  for (const auto& group: independent) {
    for (const auto type: group) {
      add_independent_variable_type(type);
    }
  }
}

void data_reader_jag_conduit::add_independent_variable_type(
  const data_reader_jag_conduit::variable_t independent) {
  if (!(independent == JAG_Image || independent == JAG_Scalar || independent == JAG_Input)) {
    LBANN_ERROR(_CN_ + ":: unrecognized independent variable type ");
  }
  m_independent.push_back(independent);
}

void data_reader_jag_conduit::set_dependent_variable_type(
  const std::vector< std::vector<data_reader_jag_conduit::variable_t> >& dependent) {
  m_dependent_groups = dependent;
  m_dependent.clear();

  for (const auto& group: dependent) {
    for (const auto type: group) {
      add_dependent_variable_type(type);
    }
  }
}

void data_reader_jag_conduit::add_dependent_variable_type(
  const data_reader_jag_conduit::variable_t dependent) {
  if (!(dependent == JAG_Image || dependent == JAG_Scalar || dependent == JAG_Input)) {
    LBANN_ERROR(_CN_ + ":: unrecognized dependent variable type ");
  }
  m_dependent.push_back(dependent);
}

std::vector<data_reader_jag_conduit::variable_t>
data_reader_jag_conduit::get_independent_variable_type() const {
  return m_independent;
}

std::vector<data_reader_jag_conduit::variable_t>
data_reader_jag_conduit::get_dependent_variable_type() const {
  return m_dependent;
}

/**
 * Note: this method is called by init_image_data_reader in
 *       src/proto/init_image_data_readers.cpp
 */
void data_reader_jag_conduit::set_image_dims(const int width, const int height, const int ch) {
  if ((width > 0) && (height > 0) && (ch > 0)) { // set and valid
    m_image_width = width;
    m_image_height = height;
    m_image_num_channels = ch;
  } else if (!((width == 0) && (height == 0) && (ch == 1))) { // set but not valid
    LBANN_ERROR(_CN_ + ":: set_image_dims() : invalid image dims");
  }
  set_linearized_image_size();
}

/**
 * Note: this method is called by init_image_data_reader in
 *       src/proto/init_image_data_readers.cpp
 */
void data_reader_jag_conduit::set_image_choices(const std::vector<std::string> image_keys) {
  m_emi_image_keys = image_keys;
  // For example, in the data reader prototext file, have a line similar to the one below
  // image_keys: ["(0.0, 0.0)/0.0","(90.0, 0.0)/0.0","(90.0, 78.0)/0.0"];

  m_num_img_srcs = m_emi_image_keys.size();
}

const std::vector<std::string>& data_reader_jag_conduit::get_image_choices() const {
  return m_emi_image_keys;
}


void data_reader_jag_conduit::add_scalar_filter(const std::string& key) {
  m_scalar_filter.insert(key);
}

void data_reader_jag_conduit::add_scalar_prefix_filter(const prefix_t& p) {
  m_scalar_prefix_filter.push_back((p.first.length() > p.second)? prefix_t(p.first, p.first.length()) : p);
}

void data_reader_jag_conduit::add_input_filter(const std::string& key) {
  m_input_filter.insert(key);
}

void data_reader_jag_conduit::add_input_prefix_filter(const prefix_t& p) {
  m_input_prefix_filter.push_back((p.first.length() > p.second)? prefix_t(p.first, p.first.length()) : p);
}

/**
 * First, it checks if the key is in the list of keys to filter.
 * Then, it checks if the key contains any prefix string to filter
 * while sayisfying the mininum length requirement.
 */
bool data_reader_jag_conduit::filter(const std::set<std::string>& key_filter,
  const std::vector<data_reader_jag_conduit::prefix_t>& prefix_filter, const std::string& key) const {
  if (key_filter.find(key) != key_filter.cend()) {
    return true;
  }
  for (const auto& pf: prefix_filter) {
    if (key.length() < pf.second) { // minimum length requirement
      continue;
    }
    if (key.compare(0, pf.first.length(), pf.first) == 0) { // match
      return true;
    }
  }
  return false;
}

/**
 * Note: this method is called by init_image_data_reader in
 *       src/proto/init_image_data_readers.cpp
 */
void data_reader_jag_conduit::set_scalar_choices(const std::vector<std::string>& keys) {
  m_scalar_keys = keys;
  check_scalar_keys();
}

void data_reader_jag_conduit::set_all_scalar_choices() {
  if (m_sample_list.empty()) {
    return;
  }
  conduit::Node n_scalar;
  load_conduit_node(0, m_output_scalar_prefix, n_scalar);
  m_scalar_keys.reserve(n_scalar.number_of_children());
  const std::vector<std::string>& child_names = n_scalar.child_names();
  for (const auto& key: child_names) {
    if (filter(m_scalar_filter, m_scalar_prefix_filter, key)) {
      continue;
    }
    m_scalar_keys.push_back(key);
  }
}

const std::vector<std::string>& data_reader_jag_conduit::get_scalar_choices() const {
  return m_scalar_keys;
}


/**
 * To use no key, set 'Undefined' to the corresponding variable type,
 * or call this with an empty vector argument after loading data.
 * Note: this method is called by init_image_data_reader in
 *       src/proto/init_image_data_readers.cpp
 */
void data_reader_jag_conduit::set_input_choices(const std::vector<std::string>& keys) {
  m_input_keys = keys;
  check_input_keys();
}

void data_reader_jag_conduit::set_all_input_choices() {
  //@TODO revisit later -- don't know how to handle this yet
  if (m_data_store != nullptr) {
    return;
  }

  if (m_sample_list.empty()) {
    return;
  }

  conduit::Node n_input;
  load_conduit_node(0, "/inputs", n_input);
  m_input_keys.reserve(n_input.number_of_children());
  const std::vector<std::string>& child_names = n_input.child_names();
  for (const auto& key: child_names) {
    if (filter(m_input_filter, m_input_prefix_filter, key)) {
      continue;
    }
    m_input_keys.push_back(key);
  }
}

const std::vector<std::string>& data_reader_jag_conduit::get_input_choices() const {
  return m_input_keys;
}


void data_reader_jag_conduit::set_linearized_image_size() {
  m_image_linearized_size = m_image_width * m_image_height * m_image_num_channels;
  m_1ch_image_linearized_size = m_image_width * m_image_height;
}

void data_reader_jag_conduit::check_image_data() {
  //@TODO revisit later -- don't know how to handle this yet
  if (m_data_store != nullptr) {
    return;
  }

  if (m_sample_list.empty()) {
    return;
  }

  size_t first_idx = (m_sample_list[0]).first;
  if (!has_conduit_path(first_idx, "")) {
    LBANN_ERROR(_CN_ + ":: check_image_data() : no sample by " + m_sample_list[first_idx].second);
    return;
  }
  conduit::Node n_imageset;
  load_conduit_node(first_idx, m_output_image_prefix, n_imageset);
  if (static_cast<size_t>(n_imageset.number_of_children()) == 0u) {
    return;
  }
  if (m_emi_image_keys.size() == 0u) {
    return;
  }
  for (const auto& emi_tag: m_emi_image_keys) {
    if (!has_conduit_path(first_idx, m_output_image_prefix + emi_tag)) {
      LBANN_ERROR(_CN_ + ":: check_image_data() : no emi image by " + emi_tag);
      return;
    }
  }
  conduit::Node n_image;
  load_conduit_node(first_idx, m_output_image_prefix + m_emi_image_keys[0], n_image);
  conduit_ch_t emi = n_image.value();

  if (m_image_linearized_size != static_cast<size_t>(emi.number_of_elements())) {
    if ((m_image_width == 0) && (m_image_height == 0)) {
      m_image_height = 1;
      m_image_width = static_cast<int>(emi.number_of_elements());
      m_image_num_channels = 1;
      set_linearized_image_size();
    } else {
      std::string msg = ":: expected linearized emi image size: "
                      + std::to_string(emi.number_of_elements()) + '\n';
      LBANN_ERROR(_CN_ +  msg + get_description());
    }
  }

  if (m_image_normalization_params.empty()) {
    m_image_normalization_params.assign(m_emi_image_keys.size()*m_image_num_channels, linear_transform_t(1.0, 0.0));
  } else if (m_image_normalization_params.size() != static_cast<size_t>(m_image_num_channels)) {
    LBANN_ERROR(_CN_ + ":: Incorrect number of image normalization parameter sets!" \
                + std::to_string(m_image_normalization_params.size()) + " != " \
                + std::to_string(m_image_num_channels));
  }
#if defined(LBANN_DEBUG)
  std::cout << "image normalization parameters: " << std::endl;
  for (size_t i = 0u, s = 0u; s < m_emi_image_keys.size(); ++s) {
    for (int c = 0; c < m_image_num_channels; ++c) {
      const auto& param = m_image_normalization_params[i*m_image_num_channels + c];
      std::cout << " scale: \t" << param.first << " \tbias: \t" << param.second
                << " \t" << m_emi_image_keys[s] << ":C" << c << std::endl;
    }
  }
#endif
}

void data_reader_jag_conduit::check_scalar_keys() {
  //@TODO revisit later -- don't know how to handle this yet
  if (m_data_store != nullptr) {
    return;
  }

  if (m_scalar_keys.empty()) {
    return;
  }
  if (!m_is_data_loaded) {
    return;
  }
  if (m_sample_list.empty()) {
    //m_scalar_keys.clear();
    return;
  }

  // If this call is made after loading data, check if the keys are in data

  size_t num_found = 0u;
  std::vector<bool> found(m_scalar_keys.size(), false);
  std::set<std::string> keys_conduit;

  conduit::Node n_scalar;
  size_t first_idx = (m_sample_list[0]).first;
  load_conduit_node(first_idx, m_output_scalar_prefix, n_scalar);
  const std::vector<std::string>& child_names = n_scalar.child_names();
  for (const auto& key: child_names) {
    keys_conduit.insert(key);
  }

  for (size_t i=0u; i < m_scalar_keys.size(); ++i) {
    std::set<std::string>::const_iterator it = keys_conduit.find(m_scalar_keys[i]);
    if (it != keys_conduit.cend()) {
      num_found ++;
      found[i] = true;
    }
  }

  if (num_found != m_scalar_keys.size()) {
    std::string msg = "keys not found:";
    for (size_t i=0u; i < m_scalar_keys.size(); ++i) {
      if (!found[i]) {
        msg += ' ' + m_scalar_keys[i];
      }
    }
    LBANN_ERROR(_CN_ + ":: check_scalar_keys() : " + msg);
  }

  if (m_scalar_normalization_params.empty()) {
    m_scalar_normalization_params.assign(m_scalar_keys.size(), linear_transform_t(1.0, 0.0));
  } else if (m_scalar_normalization_params.size() != m_scalar_keys.size()) {
     LBANN_ERROR(_CN_ + ":: Incorrect number of scalar normalization parameter sets! " \
                 + std::to_string(m_scalar_normalization_params.size()) + " != " \
                 + std::to_string(m_scalar_keys.size()));
  }
#if defined(LBANN_DEBUG)
  std::cout << "scalar normalization parameters: " << std::endl;
  for (size_t i = 0u; i < m_scalar_normalization_params.size(); ++i) {
    const auto& param = m_scalar_normalization_params[i];
    std::cout << " scale: \t" << param.first << " \tbias: \t" << param.second << "\t " << m_scalar_keys[i] << std::endl;
  }
#endif
}


void data_reader_jag_conduit::check_input_keys() {
  //@TODO revisit later -- don't know how to handle this yet
  if (m_data_store != nullptr) {
    return;
  }

  if (m_input_keys.empty()) {
    return;
  }
  if (!m_is_data_loaded) {
    return;
  }
  if (m_sample_list.empty()) {
    //m_input_keys.clear();
    return;
  }

  // If this call is made after loading data, check if the keys

  size_t num_found = 0u;
  std::vector<bool> found(m_input_keys.size(), false);
  std::map<std::string, TypeID> keys_conduit;

  conduit::Node n_input;
  size_t first_idx = (m_sample_list[0]).first;
  load_conduit_node(first_idx, "/inputs", n_input);
  conduit::NodeConstIterator itr = n_input.children();

  while (itr.has_next()) {
    const conduit::Node & n = itr.next();
    keys_conduit.insert(std::pair<std::string, TypeID>(itr.name(), static_cast<TypeID>(n.dtype().id())));
  }

  bool is_input_t = true;

  for (size_t i=0u; i < m_input_keys.size(); ++i) {
    std::map<std::string, TypeID>::const_iterator it = keys_conduit.find(m_input_keys[i]);
    if (it != keys_conduit.cend()) {
      num_found ++;
      found[i] = true;
      is_input_t = is_input_t && is_same_type<input_t>(it->second);
    }
  }

  if (num_found != m_input_keys.size()) {
    std::string msg = "keys not found:";
    for (size_t i=0u; i < m_input_keys.size(); ++i) {
      if (!found[i]) {
        msg += ' ' + m_input_keys[i];
      }
    }
    LBANN_ERROR(_CN_ + ":: check_input_keys() : " + msg);
  }

  m_uniform_input_type = (m_input_keys.size() == 0u)? false : is_input_t;

  if (m_input_normalization_params.empty()) {
    m_input_normalization_params.assign(m_input_keys.size(), linear_transform_t(1.0, 0.0));
  } else if (m_input_normalization_params.size() != m_input_keys.size()) {
     LBANN_ERROR(_CN_ + ":: Incorrect number of input normalization parameter sets! " \
                 + std::to_string(m_input_normalization_params.size()) + " != " \
                 + std::to_string(m_input_keys.size()));
  }
#if defined(LBANN_DEBUG)
  std::cout << "input normalization parameters: " << std::endl;
  for (size_t i = 0u; i < m_input_normalization_params.size(); ++i) {
    const auto& param = m_input_normalization_params[i];
    std::cout << " scale: \t" << param.first << " \tbias: \t" << param.second << " \t" << m_input_keys[i] << std::endl;
  }
#endif
}


void data_reader_jag_conduit::load() {
  if(m_gan_labelling) {
    m_num_labels=2;
  }

  if (is_master()) {
    std::cout << "JAG load GAN m_gan_labelling : label_value "
              << m_gan_labelling <<" : " << m_gan_label_value << std::endl;
  }

  if ((m_leading_reader != this) && (m_leading_reader != nullptr)) {
    // The following member variables of the leadering reader should have been
    // copied when this was copy-constructed: m_sample_list, and m_open_hdf5_files
    return;
  }

  m_shuffled_indices.clear();

  if(is_master()) {
    std::cout << "data_reader_jag_conduit - starting load" << std::endl;
  }
  const std::string data_dir = add_delimiter(get_file_dir());
  const std::string sample_list_file = data_dir + get_data_index_list();

  options *opts = options::get();
  bool check_data = opts->get_bool("check_data");

  /// The use of these flags need to be updated to properly separate
  /// how index lists are used between trainers and models
  /// @todo m_list_per_trainer || m_list_per_model
  double tm2 = get_time();
  load_list_of_samples(sample_list_file, m_comm->get_procs_per_trainer(), m_comm->get_rank_in_trainer());
  if(is_master()) {
      std::cout << "Finished loading sample list; time: " << get_time() - tm2 << std::endl;
    if (!check_data) {
      std::cout << "Skipping check data" << std::endl;
    }
  }

  /// Check the data that each rank loaded
  if (!m_is_data_loaded && !m_sample_list.empty()) {
    m_is_data_loaded = true;

    /// Open the first sample to make sure that all of the fields are correct
    m_sample_list.open_samples_file_handle(0, true);

    if (m_scalar_keys.size() == 0u) {
      set_all_scalar_choices(); // use all by default if none is specified
    }
    if (check_data) {
      check_scalar_keys();
    }

    if (m_input_keys.size() == 0u) {
      set_all_input_choices(); // use all by default if none is specified
    }
    if (check_data) {
      check_input_keys();
    }

    if (check_data) {
      check_image_data();
    }

    m_sample_list.close_if_done_samples_file_handle(0);
  }
  if(is_master()) {
    std::cout << "Done with data checking" << std::endl;
  }

  /// Merge all of the sample lists
  tm2 = get_time();
  m_sample_list.all_gather_packed_lists(*m_comm);
  if (opts->has_string("write_sample_list") && m_comm->am_trainer_master()) {
    {
      const std::string msg = " writing sample list " + sample_list_file;
      LBANN_WARNING(msg);
    }
    std::stringstream s;
    std::string basename = get_basename_without_ext(sample_list_file);
    std::string ext = get_ext_name(sample_list_file);
    s << basename << "." << ext;
    m_sample_list.write(s.str());
  }
  if (is_master()) {
    std::cout << "time for all_gather_packed_lists: " << get_time() - tm2 << std::endl;
  }

  m_shuffled_indices.resize(m_sample_list.size());
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  resize_shuffled_indices();

  if(is_master()) {
    std::cout << "Lists have been gathered" << std::endl;
  }

  instantiate_data_store();
  select_subset_of_data();
}

void data_reader_jag_conduit::preload_helper(const hid_t& h, const std::string &sample_name, const std::string &field_name, int data_id, conduit::Node &node) {
  const std::string path = sample_name + field_name;
  const std::string key2 = '/' + LBANN_DATA_ID_STR(data_id) + field_name;
  read_node(h, path, node[key2]);
}

void data_reader_jag_conduit::do_preload_data_store() {
  conduit::Node work;
  const std::string key; // key = "" is intentional

  /// @todo BVE FIXME this
  m_rank_in_model = get_comm()->get_rank_in_trainer();

  options *opts = options::get();
  double tm1 = get_time();
  if (get_comm()->am_world_master() ||
      (opts->get_bool("ltfb_verbose") && get_comm()->am_trainer_master())) {
    LBANN_WARNING("starting preload for role: ", get_role());
  }

  for (size_t idx=0; idx < m_shuffled_indices.size(); idx++) {
    int index = m_shuffled_indices[idx];
    if(m_data_store->get_index_owner(index) != m_rank_in_model) {
      continue;
    }
    try {
      const sample_t& s = m_sample_list[index];
      const std::string& sample_name = s.second;
      sample_file_id_t id = s.first;
      m_sample_list.open_samples_file_handle(index, true);
      auto h = m_sample_list.get_samples_file_handle(id);
      conduit::Node & node = m_data_store->get_empty_node(index);

      preload_helper(h, sample_name, m_output_scalar_prefix, index, node);
      preload_helper(h, sample_name, m_input_prefix, index, node);
      for (auto t : m_emi_image_keys) {
        const std::string field_name = m_output_image_prefix + t;
        preload_helper(h, sample_name, field_name, index, node);
      }
      m_data_store->set_preloaded_conduit_node(index, node);
    } catch (conduit::Error const& e) {
      LBANN_ERROR(" :: trying to load the node " + std::to_string(index) + " with key " + key + " and got " + e.what());
    }
  }
  /// Once all of the data has been preloaded, close all of the file handles
  for (size_t idx=0; idx < m_shuffled_indices.size(); idx++) {
    int index = m_shuffled_indices[idx];
    if(m_data_store->get_index_owner(index) != m_rank_in_model) {
      continue;
    }
    m_sample_list.close_if_done_samples_file_handle(index);
  }

  if (get_comm()->am_world_master() ||
      (opts->get_bool("ltfb_verbose") && get_comm()->am_trainer_master())) {
    std::stringstream msg;
    msg << " loading data for role: " << get_role() << " took " << get_time() - tm1 << "s";
    LBANN_WARNING(msg.str());
  }
}

void data_reader_jag_conduit::load_list_of_samples(const std::string sample_list_file, size_t stride, size_t offset) {
  // load the sample list
  double tm1 = get_time();
  m_sample_list.load(sample_list_file, stride, offset);
  double tm2 = get_time();

  if (is_master()) {
    std::cout << "Time to load sample list: " << tm2 - tm1 << std::endl;
  }
}

void data_reader_jag_conduit::load_list_of_samples_from_archive(const std::string& sample_list_archive) {
  // load the sample list
  double tm1 = get_time();
  std::stringstream ss(sample_list_archive); // any stream can be used

  cereal::BinaryInputArchive iarchive(ss); // Create an input archive

  iarchive(m_sample_list); // Read the data from the archive
  double tm2 = get_time();

  if (is_master()) {
    std::cout << "Time to load sample list from archive: " << tm2 - tm1 << std::endl;
  }
}

unsigned int data_reader_jag_conduit::get_num_img_srcs() const {
  return m_num_img_srcs;
}

size_t data_reader_jag_conduit::get_linearized_image_size() const {
  return m_image_linearized_size;
}

size_t data_reader_jag_conduit::get_linearized_1ch_image_size() const {
  return m_1ch_image_linearized_size;
}

size_t data_reader_jag_conduit::get_linearized_scalar_size() const {
  return m_scalar_keys.size();
}

size_t data_reader_jag_conduit::get_linearized_input_size() const {
  return m_input_keys.size();
}


size_t data_reader_jag_conduit::get_linearized_size(const data_reader_jag_conduit::variable_t t) const {
  switch (t) {
    case JAG_Image:
      return get_linearized_image_size() * get_num_img_srcs();
    case JAG_Scalar:
      return get_linearized_scalar_size();
    case JAG_Input:
      return get_linearized_input_size();
    default: { // includes Unefined case
      LBANN_ERROR(_CN_ + ":: get_linearized_size() : " \
                  + "unknown or undefined variable type");
    }
  }
  return 0u;
}

int data_reader_jag_conduit::get_linearized_data_size() const {
  size_t sz = 0u;
  for (const auto t: m_independent) {
    sz += get_linearized_size(t);
  }
  return static_cast<int>(sz);
}

int data_reader_jag_conduit::get_linearized_response_size() const {
  size_t sz = 0u;
  for (const auto t: m_dependent) {
    sz += get_linearized_size(t);
  }
  return static_cast<int>(sz);
}

std::vector<size_t> data_reader_jag_conduit::get_linearized_data_sizes() const {
  std::vector<size_t> all_dim;
  all_dim.reserve(m_independent.size());
  for (const auto t: m_independent) {
    all_dim.push_back(get_linearized_size(t));
  }
  if (all_dim.empty()) {
    return {0u};
  }
  return all_dim;
}

std::vector<size_t> data_reader_jag_conduit::get_linearized_response_sizes() const {
  std::vector<size_t> all_dim;
  all_dim.reserve(m_dependent.size());
  for (const auto t: m_dependent) {
    all_dim.push_back(get_linearized_size(t));
  }
  if (all_dim.empty()) {
    return {0u};
  }
  return all_dim;
}

const std::vector<int> data_reader_jag_conduit::get_dims(const data_reader_jag_conduit::variable_t t) const {
  switch (t) {
    case JAG_Image:
      return {static_cast<int>(get_num_img_srcs()), m_image_height, m_image_width};
      //return {static_cast<int>(get_linearized_image_size())};
    case JAG_Scalar:
      return {static_cast<int>(get_linearized_scalar_size())};
    case JAG_Input:
      return {static_cast<int>(get_linearized_input_size())};
    default: { // includes Undefined case
      LBANN_ERROR(_CN_ + ":: get_dims() : " \
                  + "unknown or undefined variable type");
    }
  }
  return {};
}

const std::vector<int> data_reader_jag_conduit::get_data_dims() const {
#if 1
  return {get_linearized_data_size()};
#else
  std::vector<int> all_dim;
  for (const auto t: m_independent) {
    const std::vector<int> ld = get_dims(t);
    all_dim.insert(all_dim.end(), ld.begin(), ld.end());
  }
  if (all_dim.empty()) {
    return {0u};
  }
  return all_dim;
#endif
}

std::vector<El::Int> data_reader_jag_conduit::get_slice_points(
  const slice_points_mode var_category,
  bool& is_supported) {
  std::vector<El::Int> slice_points;
  is_supported = true;
  if (var_category == slice_points_mode::INDEPENDENT) {
    slice_points = get_slice_points_independent();
  } else if (var_category == slice_points_mode::DEPENDENT) {
    slice_points = get_slice_points_dependent();
  } else if (var_category == slice_points_mode::NA) {
    is_supported = false;
  } else {
    LBANN_ERROR("Unknown variable category \"" + lbann::to_string(var_category) \
                + "\". Must be either \"independent\" or \"dependent\".");
  }
  return slice_points;
}

std::vector<El::Int> data_reader_jag_conduit::get_slice_points_impl(const std::vector< std::vector<data_reader_jag_conduit::variable_t> >& var) const {
  std::vector<El::Int> points(var.size()+1u, static_cast<El::Int>(0));
  for (size_t i = 0u; i < var.size(); ++i) {
    const auto& group = var[i];
    size_t size = 0u;
    for (const auto type: group) {
      size += get_linearized_size(type);
    }
    points[i+1] = points[i] + static_cast<El::Int>(size);
  }
  return points;
}

std::vector<El::Int> data_reader_jag_conduit::get_slice_points_independent() const {
  return get_slice_points_impl(m_independent_groups);
}

std::vector<El::Int> data_reader_jag_conduit::get_slice_points_dependent() const {
  return get_slice_points_impl(m_dependent_groups);
}

int data_reader_jag_conduit::get_num_data() const {

  return (int)m_shuffled_indices.size();
}

int data_reader_jag_conduit::get_num_labels() const {
  return m_num_labels;
}

int data_reader_jag_conduit::get_linearized_label_size() const {
  return m_num_labels;
}

int data_reader_jag_conduit::get_linearized_size(const std::string& desc) const {
  if (desc == "JAG_Image") {
    return get_linearized_size(JAG_Image);
  } else if (desc == "JAG_Scalar") {
    return get_linearized_size(JAG_Scalar);
  } else if (desc == "JAG_Input") {
    return get_linearized_size(JAG_Input);
  } else {
    LBANN_ERROR(_CN_ + ":: get_linearized_size() : unknown key " + desc);
  }
  return generic_data_reader::get_linearized_size(desc);
}

void data_reader_jag_conduit::set_split_image_channels() {
  m_split_channels = true;
}

void data_reader_jag_conduit::unset_split_image_channels() {
  m_split_channels = false;
}

bool data_reader_jag_conduit::check_split_image_channels() const {
  return m_split_channels;
}


std::string data_reader_jag_conduit::to_string(const variable_t t) {
  switch (t) {
    case Undefined:  return "Undefined";
    case JAG_Image:  return "JAG_Image";
    case JAG_Scalar: return "JAG_Scalar";
    case JAG_Input:  return "JAG_Input";
  }
  return "Undefined";
}

std::string data_reader_jag_conduit::to_string(const std::vector<data_reader_jag_conduit::variable_t>& vec) {
  std::string str("[");
  for (const auto& el: vec) {
    str += ' ' + data_reader_jag_conduit::to_string(el);
  }
  str += " ]";
  return str;
}

std::string data_reader_jag_conduit::to_string(const std::vector< std::vector<data_reader_jag_conduit::variable_t> >& vec) {
  std::string str("[");
  for (const auto& el: vec) {
    str += ' ' + data_reader_jag_conduit::to_string(el);
  }
  str += " ]";
  return str;
}

std::string data_reader_jag_conduit::get_description() const {
  std::stringstream leading_reader;
  leading_reader << m_leading_reader;
  std::string ret = std::string("data_reader_jag_conduit:\n")
    + " - independent: " + data_reader_jag_conduit::to_string(m_independent_groups) + "\n"
    + " - dependent: " + data_reader_jag_conduit::to_string(m_dependent_groups) + "\n"
    + " - images: "   + std::to_string(m_num_img_srcs) + " of "
                      + std::to_string(m_image_num_channels) + 'x'
                      + std::to_string(m_image_width) + 'x'
                      + std::to_string(m_image_height) + "\n"
    + " - scalars: "  + std::to_string(get_linearized_scalar_size()) + "\n"
    + " - inputs: "   + std::to_string(get_linearized_input_size()) + "\n"
    + " - linearized data size: "   + std::to_string(get_linearized_data_size()) + "\n"
    + " - uniform_input_type: " + (m_uniform_input_type? "true" : "false") + "\n"
    + " - leading DR: " + (m_leading_reader == this ? "true" : "false")
    + " (ptr=" + leading_reader.str() + ")\n";
  if (!m_scalar_filter.empty()) {
    ret += " - scalar filter:";
    for (const auto& f: m_scalar_filter) {
      ret += " \"" + f + '"';
    }
    ret += '\n';
  }
  if (!m_scalar_prefix_filter.empty()) {
    ret += " - scalar prefix filter:";
    for (const auto& f: m_scalar_prefix_filter) {
      ret += " [\"" + f.first + "\" " + std::to_string(f.second) + ']';
    }
    ret += '\n';
  }
  if (!m_input_filter.empty()) {
    ret += " - input filter:";
    for (const auto& f: m_input_filter) {
      ret += " \"" + f + '"';
    }
    ret += '\n';
  }
  if (!m_input_prefix_filter.empty()) {
    ret += " - input prefix filter:";
    for (const auto& f: m_input_prefix_filter) {
      ret += " [\"" + f.first + "\" " + std::to_string(f.second) + ']';
    }
    ret += '\n';
  }
  return ret;
}


bool data_reader_jag_conduit::check_non_numeric(const std::string key) {
  std::set<std::string>::const_iterator kit = non_numeric_vars.find(key);
  if (kit != non_numeric_vars.cend()) {
    std::string err = "data_reader_jag_conduit::add_val() : non-numeric '" + key
                    + "' requires a conversion method.";
   #if 1
    std::cout << err << " Skipping for now." << std::endl;
   #else
    throw lbann_exception(err);
   #endif
    return true;
  }
  return false;
}


std::vector< std::vector<DataType> >
data_reader_jag_conduit::get_image_data(const size_t sample_id, conduit::Node& sample) const {
  std::vector< std::vector<DataType> > image_ptrs;
  image_ptrs.reserve(m_emi_image_keys.size());

  for (const auto& emi_tag : m_emi_image_keys) {
    const std::string conduit_field = m_output_image_prefix + emi_tag;
    const std::string conduit_obj = LBANN_DATA_ID_STR(sample_id) + conduit_field;
    if(sample[conduit_obj].schema().dtype().is_empty()) {
      if (data_store_active()) {
        LBANN_ERROR("Unable to find field ", conduit_obj,
                    " in conduit node: ", std::to_string(sample_id),
                    ": num nodes successfully loaded by this rank: ",
                    m_data_store->get_data_size(), " num successful calls to get_image_data on this rank: ");
      }
      conduit::Node n_image;
      bool from_file = load_conduit_node(sample_id, conduit_field, n_image);
      if (from_file) {
        sample[conduit_obj].set(n_image);
      } else {
        sample = n_image;
      }
    }
    conduit_ch_t emi = sample[conduit_obj].value();
    const size_t num_vals = emi.number_of_elements();
    const ch_t* emi_data = sample[conduit_obj].value();
    // Note that data will be cast from ch_t to DataType format
    image_ptrs.emplace_back(emi_data, emi_data + num_vals);
  }

  return image_ptrs;
}


std::vector<data_reader_jag_conduit::scalar_t> data_reader_jag_conduit::get_scalars(const size_t sample_id, conduit::Node& sample) const {
  std::vector<scalar_t> scalars;
  scalars.reserve(m_scalar_keys.size());

  auto tr = m_scalar_normalization_params.cbegin();

  for(const auto key: m_scalar_keys) {
    std::string conduit_field = m_output_scalar_prefix + key;
    std::string conduit_obj = '/' + LBANN_DATA_ID_STR(sample_id) + '/' + conduit_field;
    if(sample[conduit_obj].schema().dtype().is_empty()) {
      if (data_store_active()) {
        LBANN_ERROR("Unable to find field " + conduit_obj
                    + " in conduit node: " + std::to_string(sample_id));
      }
      conduit::Node n_scalar;
      bool from_file = load_conduit_node(sample_id, conduit_field, n_scalar);
      if (from_file) {
        sample[conduit_obj].set(n_scalar);
      } else {
        sample = n_scalar;
      }
    }
    const scalar_t val_raw = static_cast<scalar_t>(sample[conduit_obj].to_value());
    const scalar_t val = static_cast<scalar_t>(val_raw * tr->first + tr->second);
    scalars.push_back(val);
    tr ++;
  }
  return scalars;
}

std::vector<data_reader_jag_conduit::input_t> data_reader_jag_conduit::get_inputs(const size_t sample_id, conduit::Node& sample) const {
  std::vector<input_t> inputs;
  inputs.reserve(m_input_keys.size());

  // The sequence of normalization parameters should follow the same order as
  // that of the variable keys.
  auto tr = m_input_normalization_params.cbegin();

  // automatically determine which method to use based on if all the variables are of input_t
  if (m_uniform_input_type) {
    // avoid some overhead by taking advantage of the fact that all the variables are of the same type
    for(const auto key: m_input_keys) {
      const std::string conduit_field = m_input_prefix + key;
      const std::string conduit_obj = '/' + LBANN_DATA_ID_STR(sample_id) + '/' + conduit_field;
      if(sample[conduit_obj].schema().dtype().is_empty()) {
        if (data_store_active()) {
          LBANN_ERROR("Unable to find field " + conduit_obj
                      + " in conduit node: " + std::to_string(sample_id));
        }
        conduit::Node n_input;
        bool from_file = load_conduit_node(sample_id, conduit_field, n_input);
        if (from_file) {
          sample[conduit_obj].set(n_input);
        } else {
          sample = n_input;
        }
      }
      const input_t val_raw = static_cast<input_t>(sample[conduit_obj].value());
      const input_t val = static_cast<input_t>(val_raw * tr->first + tr->second);
      inputs.push_back(val);
      tr ++;
    }
  } else {
    for(const auto key: m_input_keys) {
      const std::string conduit_field = m_input_prefix + key;
      const std::string conduit_obj = '/' + LBANN_DATA_ID_STR(sample_id) + '/' + conduit_field;
      if(sample[conduit_obj].schema().dtype().is_empty()) {
        if (data_store_active()) {
          LBANN_ERROR("Unable to find field " + conduit_obj
                      + " in conduit node: " + std::to_string(sample_id));
        }
        conduit::Node n_input;
        bool from_file = load_conduit_node(sample_id, conduit_field, n_input);
        if (from_file) {
          sample[conduit_obj].set(n_input);
        } else {
          sample = n_input;
        }
      }
      add_val(key, sample[conduit_obj], inputs); // more overhead but general
      input_t& val = inputs.back();
      val = static_cast<input_t>(val * tr->first + tr->second);
      tr ++;
    }
  }

  return inputs;
}

std::vector<CPUMat>
data_reader_jag_conduit::create_datum_views(CPUMat& X, const std::vector<size_t>& sizes, const int mb_idx) const {
  std::vector<CPUMat> X_v(sizes.size());
  El::Int h = 0;

  for(size_t i=0u; i < sizes.size(); ++i) {
    const El::Int h_end =  h + static_cast<El::Int>(sizes[i]);
    El::View(X_v[i], X, El::IR(h, h_end), El::IR(mb_idx, mb_idx + 1));
    h = h_end;
  }
  return X_v;
}

bool data_reader_jag_conduit::fetch(CPUMat& X, int data_id, conduit::Node& sample, int mb_idx, int tid,
  const data_reader_jag_conduit::variable_t vt, const std::string tag) {
  switch (vt) {
    case JAG_Image: {
      const size_t num_images = get_num_img_srcs();
      const size_t num_channels = m_image_num_channels;
      const size_t image_size = get_linearized_image_size();
      const std::vector<size_t> sizes(num_images, image_size);
      std::vector<CPUMat> X_v = create_datum_views(X, sizes, mb_idx);
      std::vector< std::vector<DataType> > img_data(get_image_data(data_id, sample));

      if (img_data.size() != num_images) {
        LBANN_ERROR(_CN_ + ":: fetch() : the number of images is not as expected " \
                    + std::to_string(img_data.size()) + "!=" + std::to_string(num_images));
      }
      if (!m_split_channels && m_image_num_channels != 1) {
        LBANN_ERROR(_CN_ + ":: fetch() : transform pipeline now requires single channel images: num_channels=" \
                    + std::to_string(m_image_num_channels) + " split_channel=" + std::to_string(m_split_channels));
      }

      std::vector<size_t> dims = {num_channels, static_cast<size_t>(m_image_height), static_cast<size_t>(m_image_width)};
      std::vector<size_t> ch_dims = {static_cast<size_t>(m_image_height), static_cast<size_t>(m_image_width)};
      auto tll = lbann::transform::repack_HWC_to_CHW_layout();

      for(size_t i=0u; i < num_images; ++i) {
        CPUMat img_mat = CPUMat(utils::get_linearized_size(dims), 1, img_data[i].data(), utils::get_linearized_size(dims));
        utils::type_erased_matrix te_img(std::move(img_mat));
        CPUMat tgt_mat = CPUMat(utils::get_linearized_size(dims), 1);
        tll.apply(te_img, X_v[i], dims);
        const std::vector<size_t> ch_sizes(num_channels, m_image_height * m_image_width);
        std::vector<CPUMat> X_ch_v = create_datum_views(X_v[i], ch_sizes, mb_idx);
        for(size_t ch = 0; ch < num_channels; ch++) {
          const auto& tr = m_image_normalization_params.at(ch);
          auto s = lbann::transform::scale_and_translate(tr.first, tr.second);
          utils::type_erased_matrix te_img_plane(std::move(X_ch_v[ch]));
          s.apply(te_img_plane, ch_dims);
        }
      }
      break;
    }
    case JAG_Scalar: {
      const std::vector<scalar_t> scalars(get_scalars(data_id, sample));
      set_minibatch_item<scalar_t>(X, mb_idx, scalars.data(), get_linearized_scalar_size());
      break;
    }
    case JAG_Input: {
      const std::vector<input_t> inputs(get_inputs(data_id, sample));
      set_minibatch_item<input_t>(X, mb_idx, inputs.data(), get_linearized_input_size());
      break;
    }
    default: { // includes Undefined case
      LBANN_ERROR(_CN_ + ":: fetch_" + tag + "() : unknown or undefined variable type");
    }
  }
  return true;
}

int data_reader_jag_conduit::reuse_data(CPUMat& X) {
  El::Copy(m_data_cache, X);
  return m_cached_data_mb_size;
}

int data_reader_jag_conduit::reuse_responses(CPUMat& Y) {
  El::Copy(m_response_cache, Y);
  return m_cached_response_mb_size;
}

int data_reader_jag_conduit::reuse_labels(CPUMat& Y) {
  El::Copy(m_label_cache, Y);
  return m_cached_label_mb_size;
}

int data_reader_jag_conduit::fetch_data(CPUMat& X, El::Matrix<El::Int>& indices_fetched) {
  if ((m_leading_reader != this) && (m_leading_reader != nullptr)) {
    return m_leading_reader->reuse_data(X);
  }
  m_cached_data_mb_size = generic_data_reader::fetch_data(X, indices_fetched);
  El::Copy(X, m_data_cache);

  return m_cached_data_mb_size;
}

int data_reader_jag_conduit::fetch_responses(CPUMat& Y) {
  if ((m_leading_reader != this) && (m_leading_reader != nullptr)) {
    return m_leading_reader->reuse_responses(Y);
  }
  m_cached_response_mb_size = generic_data_reader::fetch_responses(Y);
  El::Copy(Y, m_response_cache);

  return m_cached_response_mb_size;
}

int data_reader_jag_conduit::fetch_labels(CPUMat& Y) {
  if ((m_leading_reader != this) && (m_leading_reader != nullptr)) {
    return m_leading_reader->reuse_labels(Y);
  }
  m_cached_label_mb_size = generic_data_reader::fetch_labels(Y);
  El::Copy(Y, m_label_cache);

  return m_cached_label_mb_size;
}


bool data_reader_jag_conduit::fetch_datum(CPUMat& X, int data_id, int mb_idx) {
  int tid = m_io_thread_pool->get_local_thread_id();
  std::vector<size_t> sizes = get_linearized_data_sizes();
  std::vector<CPUMat> X_v = create_datum_views(X, sizes, mb_idx);
  bool ok = true;
  // Create a node to hold all of the data
  conduit::Node node;
  if (data_store_active()) {
    const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
    node.set_external(ds_node);
  }else {
    m_sample_list.open_samples_file_handle(data_id);
  }

  for(size_t i = 0u; ok && (i < X_v.size()); ++i) {
    // The third argument mb_idx below is 0 because it is for the view of X not X itself
    ok = fetch(X_v[i], data_id, node, 0, tid, m_independent[i], "datum");
  }

  if (priming_data_store()) {
    // Once the node has been populated save it in the data store
    m_data_store->set_conduit_node(data_id, node);
  }

  m_sample_list.close_if_done_samples_file_handle(data_id);
  m_using_random_node.erase(m_io_thread_pool->get_local_thread_id());
  return ok;
}

bool data_reader_jag_conduit::fetch_response(CPUMat& X, int data_id, int mb_idx) {
  const auto& c = static_cast<const sgd_execution_context&>(m_trainer->get_data_coordinator().get_execution_context());
  int tid = m_io_thread_pool->get_local_thread_id();
  std::vector<size_t> sizes = get_linearized_response_sizes();
  std::vector<CPUMat> X_v = create_datum_views(X, sizes, mb_idx);
  bool ok = true;
  // Create a node to hold all of the data
  conduit::Node node;
  if (m_data_store != nullptr && c.get_epoch() > 0) {
    const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
    node.set_external(ds_node);
  }
  for(size_t i = 0u; ok && (i < X_v.size()); ++i) {
    ok = fetch(X_v[i], data_id, node, 0, tid, m_dependent[i], "response");
  }
  if (m_data_store != nullptr && c.get_epoch() == 0) {
    // Once the node has been populated save it in the data store
    if (m_data_store != nullptr) {
      m_data_store->set_conduit_node(data_id, node);
    }
  }
  return ok;
}

bool data_reader_jag_conduit::fetch_label(CPUMat& Y, int data_id, int mb_idx) {
  if(m_gan_label_value) Y.Set(m_gan_label_value,mb_idx,1); //fake sample is set to 1; adversarial model
  else { //fake sample (second half of minibatch is set to 0;discriminator model
    //mb_idx < (m_mb_size/2) ? Y.Set(1,mb_idx,1) : Y.Set(m_gan_label_value,mb_idx,1);
    mb_idx < (get_current_mini_batch_size()/2) ? Y.Set(1,mb_idx,1) : Y.Set(m_gan_label_value,mb_idx,1);
  }
  //Y.Set(m_gan_label_value, mb_idx, 1);
  return true;
}

void data_reader_jag_conduit::setup_data_store(int mini_batch_size) {
   if (m_data_store != nullptr) {
     m_data_store->setup(mini_batch_size);
   }
}

void data_reader_jag_conduit::print_schema(const size_t sample_id) const {
  //@TODO revisit later -- don't know how to handle this yet
  if (m_data_store != nullptr) {
    return;
  }

  conduit::Node n;
  load_conduit_node(sample_id, "", n);
  n.schema().print();
}

void data_reader_jag_conduit::clear_image_normalization_params() {
  m_image_normalization_params.clear();
}

void data_reader_jag_conduit::clear_scalar_normalization_params() {
  m_scalar_normalization_params.clear();
}

void data_reader_jag_conduit::clear_input_normalization_params() {
  m_input_normalization_params.clear();
}

void data_reader_jag_conduit::add_image_normalization_param(const data_reader_jag_conduit::linear_transform_t& t) {
  m_image_normalization_params.push_back(t);
}

void data_reader_jag_conduit::add_scalar_normalization_param(const data_reader_jag_conduit::linear_transform_t& t) {
  m_scalar_normalization_params.push_back(t);
}

void data_reader_jag_conduit::add_input_normalization_param(const data_reader_jag_conduit::linear_transform_t& t) {
  m_input_normalization_params.push_back(t);
}

} // end of namespace lbann

#undef _CN_
