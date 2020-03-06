//////////////////////////////////////////////////////////////////////////////
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
// checkpoint .hpp .cpp - Callback hooks to checkpoint model
////////////////////////////////////////////////////////////////////////////////
#ifndef LBANN_CALLBACKS_CALLBACK_CHECKPOINT_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_CHECKPOINT_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include "lbann/io/persist.hpp"

namespace lbann {
namespace callback {

enum class callback_phase {
  batch,
  epoch,
  validation,
  inference,
  invalid
};

/** @brief Checkpoint at given interval in given directory */
class checkpoint : public callback_base {
 public:

  /** @brief Construct the checkpoint callback
   *
   *  It may be beneficial to the distributed checkpoints at a higher
   *  tempo than the shared checkpoints because they are less
   *  expensive.
   *
   *  @param checkpoint_dir directory to save checkpoint files
   *  @param checkpoint_epochs interval to checkpoint
   *  @param checkpoint_steps interval to checkpoint
   *  @param checkpoint_secs interval to checkpoint
   *  @param per_rank_dir The directory into which to dump distributed checkpoints
   *  @param ckpt_dist_epochs The frequency of distributed checkpoints in epochs
   *  @param ckpt_dist_steps The frequence of distributed checkpoints in steps
   */
  checkpoint(std::string checkpoint_dir,
             int checkpoint_epochs,
             int checkpoint_steps,
             int checkpoint_secs,
             std::string per_rank_dir,
             int ckpt_dist_epochs,
             int ckpt_dist_steps) :
    callback_base(),
    m_active_trainer(nullptr),
    m_checkpoint_dir(checkpoint_dir),
    m_checkpoint_epochs(checkpoint_epochs),
    m_checkpoint_steps(checkpoint_steps),
    m_checkpoint_secs(checkpoint_secs),
    m_per_rank_dir(per_rank_dir),
    m_ckpt_dist_epochs(ckpt_dist_epochs),
    m_ckpt_dist_steps(ckpt_dist_steps) {}
  checkpoint(const checkpoint&) = default;
  checkpoint& operator=(const checkpoint&) = default;
  checkpoint* copy() const override { return new checkpoint(*this); }
  void setup(model *m) override;
  void setup(trainer *t) override;
  void on_train_begin(model *m) override;
  void on_epoch_end(model *m) override;
  void on_batch_end(model *m) override;
  void on_validation_end(model *m) override;

  inline void set_checkpoint_dir(const std::string& dir){
    m_checkpoint_dir = dir;
  }

  inline const std::string& get_checkpoint_dir(){
    return m_checkpoint_dir;
  }

  inline void set_active_trainer(trainer* t){
    m_active_trainer = t;
  }

  inline trainer& get_active_trainer(){
    if(m_active_trainer == nullptr) {
      LBANN_ERROR("No active trainer for the checkpoint callback");
    }
    return *m_active_trainer;
  }

  inline void set_checkpoint_epochs(int epochs){
    m_checkpoint_epochs= epochs;
  }

  inline void set_checkpoint_steps(int steps){
    m_checkpoint_steps= steps;
  }

  inline void set_checkpoint_secs(EvalType secs){
    m_checkpoint_secs= secs;
  }

  inline void set_per_rank_dir(std::string dir){
    m_per_rank_dir = dir;
  }

  inline const std::string& get_per_rank_dir(){
    return m_per_rank_dir;
  }

  inline void set_ckpt_dist_epochs(int ckpt_dist_epochs){
    m_ckpt_dist_epochs = ckpt_dist_epochs;
  }

  inline void set_ckpt_dist_steps(int ckpt_dist_steps){
    m_ckpt_dist_steps = ckpt_dist_steps;
  }

  inline std::string get_shared_checkpoint_rootdir() {
    return get_checkpoint_dir();
  }

  inline std::string get_distributed_checkpoint_rootdir() {
    if(m_per_rank_dir.length()) {
      return get_per_rank_dir() + "/" + get_checkpoint_dir();
    }else {
      return get_checkpoint_dir();
    }
  }

  bool need_checkpoint(model *m, callback_phase phase);
  std::string find_latest_checkpoint(model *m, std::string& latest_file, execution_mode& mode, size_t &epoch, size_t& step, int& shared);
  bool open_latest_checkpoint(model *m,
                              const std::string& task_label,
                              std::function<void(/*const */persist&)> reload_shared_ckpt,
                              std::function<void(/*const */persist&)> reload_distributed_ckpt);
  bool reload_model(model *m);
  bool reload_trainer(trainer *t);
  bool restart(model *m);
  std::string name() const override { return "checkpoint"; }
 protected:
  bool do_checkpoint(model *m);
 private:
  trainer* m_active_trainer;
  std::string m_checkpoint_dir;
  int m_checkpoint_epochs;
  int m_checkpoint_steps;
  EvalType m_checkpoint_secs;
  std::string m_per_rank_dir;
  int m_ckpt_dist_epochs;
  int m_ckpt_dist_steps;
  EvalType m_checkpoint_last;
  persist p;
  bool m_checkpoint_dist;
  bool m_checkpoint_shared;

  template<size_t _max_dir_len>
  struct header_t {
    execution_mode mode;
    int epoch;
    int step;
    int shared;
    char dirname[_max_dir_len];
  };
};

inline std::string get_trainer_checkpoint_dirname(const std::string& trainer_name, const std::string& dir) {
  return build_string(dir, '/', trainer_name, '/');
}

inline std::string get_last_shared_checkpoint_filename(model *m, const std::string& dir) {
  return build_string(dir, '/', m->get_name(), ".last.shared.checkpoint");
}

inline std::string get_last_shared_checkpoint_filename(const std::string& trainer_name, model *m, const std::string& dir) {
  return get_last_shared_checkpoint_filename(m, get_trainer_checkpoint_dirname(trainer_name, dir));
}

inline std::string get_shared_checkpoint_dirname(model *m, const std::string& dir, execution_mode mode, size_t epoch, size_t step) {
  return build_string(dir, '/', m->get_name(), ".shared.", to_string(mode), ".epoch.", epoch, ".step.", step, '/');
}

inline std::string get_shared_checkpoint_dirname(const std::string& trainer_name, model *m, const std::string& dir, execution_mode mode, size_t epoch, size_t step) {
  return get_shared_checkpoint_dirname(m, get_trainer_checkpoint_dirname(trainer_name, dir), mode, epoch, step);
}

inline std::string get_last_distributed_checkpoint_filename(model *m, const std::string& dir) {
  return build_string(dir, '/', m->get_name(), ".last.distributed.checkpoint");
}

inline std::string get_last_distributed_checkpoint_filename(const std::string& trainer_name, model *m, const std::string& dir) {
  return get_last_distributed_checkpoint_filename(m, get_trainer_checkpoint_dirname(trainer_name, dir));
}

inline std::string get_distributed_checkpoint_dirname(model *m, const std::string& dir, execution_mode mode, size_t epoch, size_t step) {
  lbann_comm *comm = m->get_comm();
  return build_string(dir, '/',
    m->get_name(),
    ".rank.", comm->get_rank_in_trainer(),
    ".distributed.", to_string(mode),
    ".epoch.", epoch,
    ".step.", step, '/');
}

inline std::string get_distributed_checkpoint_dirname(const std::string& trainer_name, model *m, const std::string& dir, execution_mode mode, size_t epoch, size_t step) {
  return get_distributed_checkpoint_dirname(m, get_trainer_checkpoint_dirname(trainer_name, dir), mode, epoch, step);
}

// Print last checkpoint to file, used to determine which checkpoint to load from.
inline bool write_latest(std::string filename, execution_mode mode, size_t epoch, size_t train) {
  // open the file for writing
  int fd = openwrite(filename.c_str());
  if (fd != -1) {
    char field[256];
    sprintf(field, "mode=%s epoch=%ld step=%ld\n", to_string(mode).c_str(), epoch, train);
    write_string(fd, filename.c_str(), field, strlen(field));
    // close our file
    closewrite(fd, filename.c_str());
  }
  return true;
}

/** \brief Reads the "latest" file and returns the epoch number and
 *        sample offset for most recent checkpoint
 */
inline bool read_latest(std::string filename, execution_mode *mode, size_t *epochLast, size_t *trainLast) {
  // assume we don't have a file, we'll return -1 in that case
  *epochLast = -1;
  *trainLast = -1;
  *mode = execution_mode::invalid;
  // open the file for reading
  int fd = openread(filename.c_str());
  if (fd != -1) {
    // read epoch from file
    char field[256];
    read_string(fd, filename.c_str(), field, sizeof(field));
    char modeStr[64];
    int ret = sscanf(field, "mode=%s epoch=%ld step=%ld\n", modeStr, epochLast, trainLast);
    *mode = exec_mode_from_string(modeStr);
    // close our file
    closeread(fd, filename.c_str());
    if(ret != 3) { return false; }
    return true;
  }
  return false;
}

// Builder function
std::unique_ptr<callback_base>
build_checkpoint_callback_from_pbuf(
  const google::protobuf::Message&);

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_CHECKPOINT_HPP_INCLUDED
