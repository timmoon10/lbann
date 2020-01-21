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

#define LBANN_UTILS_IM2COL_INSTANTIATE
#include "lbann/utils/im2col.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

template <typename TensorDataType>
void im2col(const CPUMatDT<TensorDataType>& im,
            CPUMatDT<TensorDataType>& col,
            const int num_channels,
            const int im_num_dims,
            const int * im_dims,
            const int * im_pads,
            const int * window_dims,
            const int * window_strides) {

  // Input and output parameters
  const int col_height = col.Height();
  const int col_width = col.Width();
  const TensorDataType *__restrict__ im_buffer = im.LockedBuffer();
  TensorDataType *__restrict__ col_buffer = col.Buffer();

  // im2col parameters
  std::vector<int> offset_start(im_num_dims);
  std::vector<int> offset_end(im_num_dims);
  std::vector<int> offset_stride(im_num_dims);
  std::vector<int> offset_num(im_num_dims);
  for(int d = 0; d < im_num_dims; ++d) {
    offset_start[d] = -im_pads[d];
    offset_end[d] = im_dims[d] + im_pads[d] - window_dims[d] + 1;
    offset_stride[d] = window_strides[d];
    offset_num[d] = (offset_end[d] - offset_start[d] + offset_stride[d] - 1) / offset_stride[d];
  }

  #ifdef LBANN_DEBUG
  const int im_size = im.Height();
  // Check matrix dimensions
  const int expected_im_size = std::accumulate(im_dims,
                                               im_dims + im_num_dims,
                                               num_channels,
                                               std::multiplies<int>());
  const int expected_col_height = std::accumulate(window_dims,
                                                  window_dims + im_num_dims,
                                                  num_channels,
                                                  std::multiplies<int>());
  const int expected_col_width = std::accumulate(offset_num.begin(),
                                                 offset_num.end(),
                                                 1,
                                                 std::multiplies<int>());
  if(im_size != expected_im_size || im.Width() != 1) {
    std::stringstream ss;
    ss << "im2col: im matrix has invalid dimensions "
       << "(expected " << expected_im_size << " x " << 1 << ", "
       << "found " << im_size << " x " << im.Width() << ")";
    throw lbann_exception(ss.str());
  }
  if(col_height != expected_col_height
     || col_width != expected_col_width) {
    std::stringstream ss;
    ss << "im2col: col matrix has invalid dimensions "
       << "(expected " << expected_col_height << " x " << expected_col_width << ", "
       << "found " << col_height << " x " << col_width << ")";
    throw lbann_exception(ss.str());
  }
  #endif // LBANN_DEBUG

  // Call optimized routine for 1x1 im2col
  std::vector<int> zeros(im_num_dims, 0), ones(im_num_dims, 1);
  if(std::equal(im_pads, im_pads + im_num_dims, zeros.begin())
     && std::equal(window_dims, window_dims + im_num_dims, ones.begin())
     && std::equal(window_strides, window_strides + im_num_dims, ones.begin())) {
    im2col_1x1(im_buffer, col_buffer, num_channels, im_num_dims, im_dims);
    return;
  }

  // Call optimized routine for 2D data
  if(im_num_dims == 2) {
    im2col_2d(im_buffer, col_buffer,
              im_dims[1], im_dims[0], im_pads[1], im_pads[0], num_channels,
              window_dims[1], window_dims[0],
              window_strides[1], window_strides[0]);
    return;
  }

  // Iterate through col matrix columns
  LBANN_OMP_PARALLEL_FOR
  for(int col_col = 0; col_col < col_width; ++col_col) {

    // Initialize arrays
    std::vector<int> offset_pos(im_num_dims);
    std::vector<int> window_pos(im_num_dims);

    // Get position of current offset
    int col_col_remainder = col_col;
    for(int d = im_num_dims-1; d >= 0; --d) {
      const int offset = col_col_remainder % offset_num[d];
      offset_pos[d] = offset_start[d] + offset * offset_stride[d];
      col_col_remainder /= offset_num[d];
    }

    // Iterate through col matrix entries
    for(int col_row = 0; col_row < col_height; ++col_row) {
      const int col_index = col_row + col_col * col_height;

      // Get position in window and channel
      int col_row_remainder = col_row;
      for(int d = im_num_dims-1; d >= 0; --d) {
        window_pos[d] = col_row_remainder % window_dims[d];
        col_row_remainder /= window_dims[d];
      }
      const int channel = col_row_remainder;

      // Get im matrix entry
      bool im_pos_valid = true;
      int im_index = channel;
      for(int d = 0; d < im_num_dims; ++d) {
        const int im_pos = offset_pos[d] + window_pos[d];
        im_pos_valid = im_pos_valid && 0 <= im_pos && im_pos < im_dims[d];
        im_index = im_pos + im_index * im_dims[d];
      }

      // Copy im matrix entry to col matrix if valid
      col_buffer[col_index] = (im_pos_valid ?
                               im_buffer[im_index] : TensorDataType(0.));

    }
  }

}

namespace {
template <typename TensorDataType, typename ReductionOpType>
void col2im_impl(const CPUMatDT<TensorDataType>& col,
                 CPUMatDT<TensorDataType>& im,
                 const int num_channels,
                 const int im_num_dims,
                 const int * im_dims,
                 const int * im_pads,
                 const int * window_dims,
                 const int * window_strides,
                 ReductionOpType reduction_op) {

  // Input and output parameters
  const int col_height = col.Height();
  const int im_size = im.Height();
  const TensorDataType *__restrict__ col_buffer = col.LockedBuffer();
  TensorDataType *__restrict__ im_buffer = im.Buffer();

  // im2col parameters
  std::vector<int> offset_start(im_num_dims);
  std::vector<int> offset_end(im_num_dims);
  std::vector<int> offset_stride(im_num_dims);
  std::vector<int> offset_num(im_num_dims);
  for(int d = 0; d < im_num_dims; ++d) {
    offset_start[d] = -im_pads[d];
    offset_end[d] = im_dims[d] + im_pads[d] - window_dims[d] + 1;
    offset_stride[d] = window_strides[d];
    offset_num[d] = (offset_end[d] - offset_start[d] + offset_stride[d] - 1) / offset_stride[d];
  }

  // Call optimized routine for 1x1 col2im
  std::vector<int> zeros(im_num_dims, 0), ones(im_num_dims, 1);
  if(std::equal(im_pads, im_pads + im_num_dims, zeros.begin())
     && std::equal(window_dims, window_dims + im_num_dims, ones.begin())
     && std::equal(window_strides, window_strides + im_num_dims, ones.begin())) {
    col2im_1x1(col_buffer, im_buffer, num_channels, im_num_dims, im_dims);
    return;
  }

  // Iterate through im matrix entries
  LBANN_OMP_PARALLEL_FOR
  for(int im_index = 0; im_index < im_size; ++im_index) {

    // Initialize arrays
    std::vector<int> im_pos(im_num_dims);
    std::vector<int> first_offset(im_num_dims);
    std::vector<int> last_offset(im_num_dims);
    std::vector<int> offset(im_num_dims);

    // Get position of im matrix entry
    int im_index_remainder = im_index;
    for(int d = im_num_dims-1; d >= 0; --d) {
      im_pos[d] = im_index_remainder % im_dims[d];
      im_index_remainder /= im_dims[d];
    }
    const int channel = im_index_remainder;

    // Initialize im matrix entry
    TensorDataType im_entry = El::TypeTraits<TensorDataType>::Zero();
    bool im_entry_initialized = false;
    bool offsets_finished = false;

    // Get window offsets containing im matrix entry
    for(int d = 0; d < im_num_dims; ++d) {
      first_offset[d] = (im_pos[d] - offset_start[d] - window_dims[d] + offset_stride[d]) / offset_stride[d];
      first_offset[d] = std::max(first_offset[d], 0);
      last_offset[d] = (im_pos[d] - offset_start[d]) / offset_stride[d];
      last_offset[d] = std::min(last_offset[d], offset_num[d] - 1);
      offset[d] = first_offset[d];
      if(first_offset[d] > last_offset[d]) {
        offsets_finished = true;
      }
    }

    // Iterate through window offsets containing im matrix entry
    while(!offsets_finished) {

      // Get col matrix entry corresponding to im matrix entry
      int col_row = channel;
      int col_col = 0;
      for(int d = 0; d < im_num_dims; ++d) {
        const int window_pos = im_pos[d] - (offset_start[d] + offset[d] * offset_stride[d]);
        col_row = window_pos + col_row * window_dims[d];
        col_col = offset[d] + col_col * offset_num[d];
      }
      const int col_index = col_row + col_col * col_height;

      // Add col matrix entry to im matrix entry
      const TensorDataType col_entry = col_buffer[col_index];
      im_entry = (im_entry_initialized ?
                  reduction_op(im_entry, col_entry) :
                  col_entry);
      im_entry_initialized = true;

      // Move to next window offset
      ++offset[im_num_dims-1];
      for(int d = im_num_dims-1; d >= 1; --d) {
        if(offset[d] > last_offset[d]) {
          offset[d] = first_offset[d];
          ++offset[d-1];
        }
      }
      offsets_finished = offset[0] > last_offset[0];

    }

    // Update output entry
    im_buffer[im_index] = im_entry;

  }

}
}// namespace <anon>

template <typename TensorDataType>
void col2im(const CPUMatDT<TensorDataType>& col,
            CPUMatDT<TensorDataType>& im,
            int num_channels,
            int im_num_dims,
            const int * im_dims,
            const int * im_pads,
            const int * window_dims,
            const int * window_strides,
            std::function<TensorDataType(const TensorDataType&, const TensorDataType&)> reduction_op) {
  col2im_impl(col, im,
              num_channels, im_num_dims,
              im_dims, im_pads, window_dims, window_strides, reduction_op);
}

template <typename TensorDataType>
void col2im(const CPUMatDT<TensorDataType>& col,
            CPUMatDT<TensorDataType>& im,
            const int num_channels,
            const int im_num_dims,
            const int * im_dims,
            const int * im_pads,
            const int * window_dims,
            const int * window_strides) {

  // Input and output parameters
  const TensorDataType *__restrict__ col_buffer = col.LockedBuffer();
  TensorDataType *__restrict__ im_buffer = im.Buffer();

  // col2im parameters
  std::vector<int> offset_start(im_num_dims);
  std::vector<int> offset_end(im_num_dims);
  std::vector<int> offset_stride(im_num_dims);
  std::vector<int> offset_num(im_num_dims);
  for(int d = 0; d < im_num_dims; ++d) {
    offset_start[d] = -im_pads[d];
    offset_end[d] = im_dims[d] + im_pads[d] - window_dims[d] + 1;
    offset_stride[d] = window_strides[d];
    offset_num[d] = (offset_end[d] - offset_start[d] + offset_stride[d] - 1) / offset_stride[d];
  }

  #ifdef LBANN_DEBUG
  const int im_size = im.Height();
  const int col_height = col.Height();
  const int col_width = col.Width();
  // Check matrix dimensions
  const int expected_im_size = std::accumulate(im_dims,
                                               im_dims + im_num_dims,
                                               num_channels,
                                               std::multiplies<int>());
  const int expected_col_height = std::accumulate(window_dims,
                                                  window_dims + im_num_dims,
                                                  num_channels,
                                                  std::multiplies<int>());
  const int expected_col_width = std::accumulate(offset_num.begin(),
                                                 offset_num.end(),
                                                 1,
                                                 std::multiplies<int>());
  if(im_size != expected_im_size || im.Width() != 1) {
    std::stringstream ss;
    ss << "im2col: im matrix has invalid dimensions "
       << "(expected " << expected_im_size << " x " << 1 << ", "
       << "found " << im_size << " x " << im.Width() << ")";
    throw lbann_exception(ss.str());
  }
  if(col_height != expected_col_height
     || col_width != expected_col_width) {
    std::stringstream ss;
    ss << "im2col: col matrix has invalid dimensions "
       << "(expected " << expected_col_height << " x " << expected_col_width << ", "
       << "found " << col_height << " x " << col_width << ")";
    throw lbann_exception(ss.str());
  }
  #endif // LBANN_DEBUG

  // Call optimized routine for 1x1 col2im
  std::vector<int> zeros(im_num_dims, 0), ones(im_num_dims, 1);
  if(std::equal(im_pads, im_pads + im_num_dims, zeros.begin())
     && std::equal(window_dims, window_dims + im_num_dims, ones.begin())
     && std::equal(window_strides, window_strides + im_num_dims, ones.begin())) {
    col2im_1x1(col_buffer, im_buffer, num_channels, im_num_dims, im_dims);
    return;
  }

  // Call optimized routine for 2D data
  if(im_num_dims == 2) {
    col2im_2d(col_buffer, im_buffer,
              im_dims[1], im_dims[0], im_pads[1], im_pads[0], num_channels,
              window_dims[1], window_dims[0],
              window_strides[1], window_strides[0]);
    return;
  }

  // Default algorithm
  col2im_impl(col, im, num_channels, im_num_dims,
              im_dims, im_pads, window_dims, window_strides,
              std::plus<TensorDataType>());

}


template <typename TensorDataType>
void im2col_1x1(const TensorDataType * __restrict__ input_buffer,
                TensorDataType * __restrict__ output_buffer,
                const int num_channels,
                const int num_input_dims,
                const int * input_dims) {
  const int spatial_size = std::accumulate(input_dims,
                                           input_dims + num_input_dims,
                                           1,
                                           std::multiplies<int>());
  const CPUMatDT<TensorDataType> input_matrix(spatial_size, num_channels,
                                              input_buffer, spatial_size);
  CPUMatDT<TensorDataType> output_matrix(num_channels, spatial_size,
                                         output_buffer, num_channels);
  El::Transpose(input_matrix, output_matrix);
}

template <typename TensorDataType>
void im2col_2d(const TensorDataType *__restrict__ input_buffer,
               TensorDataType *__restrict__ output_buffer,
               const int input_dim_x,
               const int input_dim_y,
               const int input_pad_x,
               const int input_pad_y,
               const int num_channels,
               const int window_dim_x,
               const int window_dim_y,
               const int offset_stride_x,
               const int offset_stride_y) {

  // im2col parameters
  const int offset_start_x = -input_pad_x;
  const int offset_start_y = -input_pad_y;
  const int offset_end_x = input_dim_x + input_pad_x - window_dim_x + 1;
  const int offset_end_y = input_dim_y + input_pad_y - window_dim_y + 1;
  const int offset_num_x = (offset_end_x - offset_start_x + offset_stride_x - 1) / offset_stride_x;
  const int offset_num_y = (offset_end_y - offset_start_y + offset_stride_y - 1) / offset_stride_y;
  const int output_height = num_channels * window_dim_x * window_dim_y;

  // Iterate through output matrix entries
  LBANN_OMP_PARALLEL_FOR_COLLAPSE5
  for(int offset_y = 0; offset_y < offset_num_y; ++offset_y) {
    for(int offset_x = 0; offset_x < offset_num_x; ++offset_x) {
      for(int channel = 0; channel < num_channels; ++channel) {
        for(int window_pos_y = 0;
            window_pos_y < window_dim_y;
            ++window_pos_y) {
          for(int window_pos_x = 0;
              window_pos_x < window_dim_x;
              ++window_pos_x) {

            // Get input entry
            const int offset_pos_y = offset_start_y + offset_y * offset_stride_y;
            const int offset_pos_x = offset_start_x + offset_x * offset_stride_x;
            const int input_pos_y = offset_pos_y + window_pos_y;
            const int input_pos_x = offset_pos_x + window_pos_x;
            const int input_index = (input_pos_x
                                     + input_pos_y * input_dim_x
                                     + channel * input_dim_x * input_dim_y);
            const bool input_pos_valid = (0 <= input_pos_y
                                          && input_pos_y < input_dim_y
                                          && 0 <= input_pos_x
                                          && input_pos_x < input_dim_x);

            // Get output entry
            const int output_row = (window_pos_x
                                    + window_pos_y * window_dim_x
                                    + channel * window_dim_x * window_dim_y);
            const int output_col = offset_x + offset_y * offset_num_x;
            const int output_index = output_row + output_col * output_height;

            // Copy input entry to output entry if valid
            output_buffer[output_index]
              = input_pos_valid ? input_buffer[input_index] : TensorDataType(0.);

          }
        }
      }
    }
  }

}

template <typename TensorDataType>
void col2im_1x1(const TensorDataType * input_buffer,
                TensorDataType * output_buffer,
                const int num_channels,
                const int num_output_dims,
                const int * output_dims) {
  const int spatial_size = std::accumulate(output_dims,
                                           output_dims + num_output_dims,
                                           1,
                                           std::multiplies<int>());
  const CPUMatDT<TensorDataType> input_matrix(num_channels, spatial_size,
                                              input_buffer, num_channels);
  CPUMatDT<TensorDataType> output_matrix(spatial_size, num_channels,
                                         output_buffer, spatial_size);
  El::Transpose(input_matrix, output_matrix);
}

template <typename TensorDataType>
void col2im_2d(const TensorDataType *__restrict__ input_buffer,
               TensorDataType *__restrict__ output_buffer,
               const int output_dim_x,
               const int output_dim_y,
               const int output_pad_x,
               const int output_pad_y,
               const int num_channels,
               const int window_dim_x,
               const int window_dim_y,
               const int offset_stride_x,
               const int offset_stride_y) {

  // col2im parameters
  const int offset_start_x = -output_pad_x;
  const int offset_start_y = -output_pad_y;
  const int offset_end_x = output_dim_x + output_pad_x - window_dim_x + 1;
  const int offset_end_y = output_dim_y + output_pad_y - window_dim_y + 1;
  const int offset_num_x = (offset_end_x - offset_start_x + offset_stride_x - 1) / offset_stride_x;
  const int offset_num_y = (offset_end_y - offset_start_y + offset_stride_y - 1) / offset_stride_y;
  const int input_height = num_channels * window_dim_x * window_dim_y;

  // Iterate through output entries
  LBANN_OMP_PARALLEL_FOR_COLLAPSE3
  for(int channel = 0; channel < num_channels; ++channel) {
    for(int output_pos_y = 0;
        output_pos_y < output_dim_y;
        ++output_pos_y) {
      for(int output_pos_x = 0;
          output_pos_x < output_dim_x;
          ++output_pos_x) {

        // Get output entry
        const int output_index = (output_pos_x
                                  + output_pos_y * output_dim_x
                                  + channel * output_dim_x * output_dim_y);
        TensorDataType output_entry = El::TypeTraits<TensorDataType>::Zero();

        // Get window offsets containing output entry
        const int offset_x_lower = (output_pos_x - offset_start_x - window_dim_x + offset_stride_x) / offset_stride_x;
        const int offset_y_lower = (output_pos_y - offset_start_y - window_dim_y + offset_stride_y) / offset_stride_y;
        const int offset_x_upper = (output_pos_x - offset_start_x) / offset_stride_x;
        const int offset_y_upper = (output_pos_y - offset_start_y) / offset_stride_y;
        const int first_offset_x = std::max(offset_x_lower, 0);
        const int first_offset_y = std::max(offset_y_lower, 0);
        const int last_offset_x = std::min(offset_x_upper, offset_num_x - 1);
        const int last_offset_y = std::min(offset_y_upper, offset_num_y - 1);

        // Iterate through window offsets
        for(int offset_y = first_offset_y;
            offset_y <= last_offset_y;
            ++offset_y) {
          const int window_pos_y = output_pos_y - (offset_start_y + offset_y * offset_stride_y);
          for(int offset_x = first_offset_x;
              offset_x <= last_offset_x;
              ++offset_x) {
            const int window_pos_x = output_pos_x - (offset_start_x + offset_x * offset_stride_x);

            // Get input entry
            const int input_row = (window_pos_x
                                   + window_pos_y * window_dim_x
                                   + channel * window_dim_x * window_dim_y);
            const int input_col = offset_x + offset_y * offset_num_x;
            const int input_index = input_row + input_col * input_height;

            // Add input entry to output entry
            output_entry += input_buffer[input_index];

          }
        }

        // Update output entry
        output_buffer[output_index] = output_entry;

      }
    }
  }

}

#define PROTO(T)                                                    \
  template void im2col<T>(                                          \
    const CPUMatDT<T>&, CPUMatDT<T>&,                               \
    int, int,                                                       \
    const int*, const int*,                                         \
    const int*, const int*);                                        \
  template void col2im<T>(                                          \
    const CPUMatDT<T>&,                                             \
    CPUMatDT<T>&,                                                   \
    int, int,                                                       \
    const int*, const int*,                                         \
    const int*, const int*);                                        \
  template void col2im<T>(                                          \
    const CPUMatDT<T>&,                                             \
    CPUMatDT<T>&,                                                   \
    int, int,                                                       \
    const int*, const int*,                                         \
    const int*, const int*,                                         \
    std::function<T(T const&, T const&)>);                          \
  template void im2col_1x1<T>(                                      \
    const T*, T*, int, int, const int*);                            \
  template void im2col_2d(                                          \
    const T*, T*, int, int, int, int, int, int, int, int, int);     \
  template void col2im_1x1(                                         \
    const T*, T*, int, int, const int*);                            \
  template void col2im_2d(                                          \
    const T*, T*, int, int, int, int, int, int, int, int, int)

#define LBANN_INSTANTIATE_CPU_HALF
// FIXME -- these should never be called in GPU code.
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

}  // namespace lbann
