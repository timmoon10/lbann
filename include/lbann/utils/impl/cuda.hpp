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

#include <thrust/system/cuda/execution_policy.h>

// Headers for NVCC
#ifdef __CUDACC__
#ifdef HYDROGEN_HAVE_CUB
#include "cub/block/block_reduce.cuh"
#endif // HYDROGEN_HAVE_CUB
#include <math_constants.h>
#include <cuda_fp16.h>
#endif // __CUDACC__

namespace lbann {
namespace cuda {

// -------------------------------------------------------------
// Device functions
// -------------------------------------------------------------
#ifdef __CUDACC__

// Atomic add function
#if __CUDA_ARCH__ >= 530
template <> __device__ __forceinline__
__half atomic_add<__half>(__half* address, __half val) {
#if __CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__)
  return atomicAdd(address, val);
#else
  unsigned int* address_as_uint = (unsigned int*) address;
  unsigned int old = *address_as_uint;
  __half* old_as_half = (__half*) &old;
  unsigned int assumed;
  unsigned int updated;
  __half* updated_as_half = (__half*) &updated;
  do {
    assumed = old;
    updated = old;
    *updated_as_half += val;
    old = atomicCAS(address_as_uint, assumed, updated);
  } while (assumed != old);
  return *old_as_half;
#endif // __CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__)
}
#endif // __CUDA_ARCH__ >= 530
template <> __device__ __forceinline__
float atomic_add<float>(float* address, float val) {
  return atomicAdd(address, val);
}
template <> __device__ __forceinline__
double atomic_add<double>(double* address, double val) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, val);
#else
  unsigned long long int* address_as_ull =
    (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                                         __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
#endif // __CUDA_ARCH__ < 600
}

// Block reduction
template <size_t bdimx, size_t bdimy, size_t bdimz, class T>
__device__ __forceinline__
T block_reduce(T val) {
#ifdef HYDROGEN_HAVE_CUB
  constexpr auto reduce_algo = cub::BLOCK_REDUCE_WARP_REDUCTIONS;
  using BlockReduce = cub::BlockReduce<T, bdimx, reduce_algo, bdimy, bdimz>;
  __shared__ typename BlockReduce::TempStorage workspace;
  val = BlockReduce(workspace).Sum(val);
#else
  const size_t tid = threadIdx.x + threadIdx.y*bdimx + threadIdx.z*bdimx*bdimy;
  constexpr size_t bsize = bdimx * bdimy * bdimz;
  __shared__ DataType shared_max_vals[bsize];
  shared_vals[tid] = val;
  for (size_t stride = bsize/2; stride > 0; stride /= 2) {
    __syncthreads();
    if (tid < stride) {
      shared_vals[tid] = shared_vals[tid] + shared_vals[tid+stride];
    }
  }
  if (tid == 0) {
    val = shared_vals[0];
  }
#endif // HYDROGEN_HAVE_CUB
  return val;
}
template <size_t bdimx, size_t bdimy, size_t bdimz, class T, class Op>
__device__ __forceinline__
T block_reduce(T val) {
#ifdef HYDROGEN_HAVE_CUB
  constexpr auto reduce_algo = cub::BLOCK_REDUCE_WARP_REDUCTIONS;
  using BlockReduce = cub::BlockReduce<T, bdimx, reduce_algo, bdimy, bdimz>;
  __shared__ typename BlockReduce::TempStorage workspace;
  val = BlockReduce(workspace).Reduce(val, Op());
#else
  Op op;
  const size_t tid = threadIdx.x + threadIdx.y*bdimx + threadIdx.z*bdimx*bdimy;
  constexpr size_t bsize = bdimx * bdimy * bdimz;
  __shared__ DataType shared_max_vals[bsize];
  shared_vals[tid] = val;
  for (size_t stride = bsize/2; stride > 0; stride /= 2) {
    __syncthreads();
    if (tid < stride) {
      shared_vals[tid] = op(shared_vals[tid], shared_vals[tid+stride]);
    }
  }
  if (tid == 0) {
    val = shared_vals[0];
  }
#endif // HYDROGEN_HAVE_CUB
  return val;
}

// Unary math functions
#define WRAP_UNARY_CUDA_MATH_FUNCTION(func)                     \
  template <> __device__ __forceinline__                        \
  float func<float>(const float& x) { return ::func##f(x); }    \
  template <> __device__ __forceinline__                        \
  double func<double>(const double& x) { return ::func(x); }
template <typename T> __device__ __forceinline__
T abs(const T& x) { return x >= static_cast<T>(0) ? x : -x; }
template <> __device__ __forceinline__
float abs<float>(const float& x) { return ::fabsf(x); }
template <> __device__ __forceinline__
double abs<double>(const double& x) { return ::fabs(x); }
WRAP_UNARY_CUDA_MATH_FUNCTION(round)
WRAP_UNARY_CUDA_MATH_FUNCTION(ceil)
WRAP_UNARY_CUDA_MATH_FUNCTION(floor)
WRAP_UNARY_CUDA_MATH_FUNCTION(sqrt)
WRAP_UNARY_CUDA_MATH_FUNCTION(rsqrt)
WRAP_UNARY_CUDA_MATH_FUNCTION(exp)
WRAP_UNARY_CUDA_MATH_FUNCTION(expm1)
WRAP_UNARY_CUDA_MATH_FUNCTION(log)
WRAP_UNARY_CUDA_MATH_FUNCTION(log1p)
WRAP_UNARY_CUDA_MATH_FUNCTION(cos)
WRAP_UNARY_CUDA_MATH_FUNCTION(sin)
WRAP_UNARY_CUDA_MATH_FUNCTION(tan)
WRAP_UNARY_CUDA_MATH_FUNCTION(acos)
WRAP_UNARY_CUDA_MATH_FUNCTION(asin)
WRAP_UNARY_CUDA_MATH_FUNCTION(atan)
WRAP_UNARY_CUDA_MATH_FUNCTION(cosh)
WRAP_UNARY_CUDA_MATH_FUNCTION(sinh)
WRAP_UNARY_CUDA_MATH_FUNCTION(tanh)
WRAP_UNARY_CUDA_MATH_FUNCTION(acosh)
WRAP_UNARY_CUDA_MATH_FUNCTION(asinh)
WRAP_UNARY_CUDA_MATH_FUNCTION(atanh)
#undef WRAP_UNARY_CUDA_MATH_FUNCTION

template <typename T> __device__ __forceinline__
bool isfinite(T const& x) { return ::isfinite(x); }

template <typename T> __device__ __forceinline__
bool isnan(T const& x) { return ::isnan(x); }

#if __CUDA_ARCH__ >= 530
template <> __device__ __forceinline__
bool isfinite(__half const& x) { return !(::__isnan(x) || ::__hisinf(x)); }

template <> __device__ __forceinline__
bool isnan(__half const& x) { return ::__hisnan(x); }

// This support is far from complete!
#define WRAP_UNARY_CUDA_HALF_MATH_FUNCTION(func)              \
  template <> __device__ __forceinline__                      \
  __half func<__half>(__half const& x) { return ::h##func(x); }

// FIXME (trb): This is maybe not the best long-term solution, but it
// might be the best we can do without really digging into
// half-precision implementation.
#define WRAP_UNARY_CUDA_HALF_CAST_TO_FLOAT_MATH_FUNCTION(func) \
  template <> __device__ __forceinline__                       \
  __half func<__half>(__half const& x) { return func(float(x)); }

WRAP_UNARY_CUDA_HALF_CAST_TO_FLOAT_MATH_FUNCTION(round)
WRAP_UNARY_CUDA_HALF_MATH_FUNCTION(ceil)
WRAP_UNARY_CUDA_HALF_MATH_FUNCTION(floor)
WRAP_UNARY_CUDA_HALF_MATH_FUNCTION(sqrt)
WRAP_UNARY_CUDA_HALF_MATH_FUNCTION(rsqrt)
WRAP_UNARY_CUDA_HALF_MATH_FUNCTION(exp)
//WRAP_UNARY_CUDA_HALF_MATH_FUNCTION(expm1)
//
// FIXME (trb): This is not going to be as accurate as a native expm1
// implementation could be:
template <> __device__ __forceinline__
__half expm1<__half>(__half const& x) {
    return ::__hsub(::hexp(x), ::__float2half(1.f));
}

WRAP_UNARY_CUDA_HALF_MATH_FUNCTION(log)
WRAP_UNARY_CUDA_HALF_CAST_TO_FLOAT_MATH_FUNCTION(log1p)
WRAP_UNARY_CUDA_HALF_MATH_FUNCTION(cos)
WRAP_UNARY_CUDA_HALF_MATH_FUNCTION(sin)

//WRAP_UNARY_CUDA_HALF_MATH_FUNCTION(tan)
//
// FIXME (trb): This just uses the trig identity. Probably less
// accurate than a native implementation.
template <> __device__ __forceinline__
__half tan<__half>(__half const& x) { return ::__hdiv(::hsin(x), ::hcos(x)); }

WRAP_UNARY_CUDA_HALF_CAST_TO_FLOAT_MATH_FUNCTION(acos)
WRAP_UNARY_CUDA_HALF_CAST_TO_FLOAT_MATH_FUNCTION(asin)
WRAP_UNARY_CUDA_HALF_CAST_TO_FLOAT_MATH_FUNCTION(atan)
WRAP_UNARY_CUDA_HALF_CAST_TO_FLOAT_MATH_FUNCTION(cosh)
WRAP_UNARY_CUDA_HALF_CAST_TO_FLOAT_MATH_FUNCTION(sinh)
WRAP_UNARY_CUDA_HALF_CAST_TO_FLOAT_MATH_FUNCTION(tanh)
WRAP_UNARY_CUDA_HALF_CAST_TO_FLOAT_MATH_FUNCTION(acosh)
WRAP_UNARY_CUDA_HALF_CAST_TO_FLOAT_MATH_FUNCTION(asinh)
WRAP_UNARY_CUDA_HALF_CAST_TO_FLOAT_MATH_FUNCTION(atanh)

#undef WRAP_UNARY_CUDA_HALF_MATH_FUNCTION
#endif // __CUDA_ARCH__ >= 530

// Binary math functions
#define WRAP_BINARY_CUDA_MATH_FUNCTION(func)                    \
  template <> __device__ __forceinline__                        \
  float func<float>(const float& x, const float& y) {           \
    return ::func##f(x,y);                                      \
  }                                                             \
  template <> __device__ __forceinline__                        \
  double func<double>(const double& x, const double& y) {       \
    return ::func(x,y);                                         \
  }
template <typename T> __device__ __forceinline__
T min(const T& x, const T& y) { return y < x ? y : x; }
template <> __device__ __forceinline__
float min<float>(const float& x, const float& y) { return ::fminf(x,y); }
template <> __device__ __forceinline__
double min<double>(const double& x, const double& y) { return ::fmin(x,y); }
template <typename T> __device__ __forceinline__
T max(const T& x, const T& y) { return y > x ? y : x; }
template <> __device__ __forceinline__
float max<float>(const float& x, const float& y) { return ::fmaxf(x,y); }
template <> __device__ __forceinline__
double max<double>(const double& x, const double& y) { return ::fmax(x,y); }
template <typename T> __device__ __forceinline__
T mod(const T& x, const T& y) { return x % y; }
template <> __device__ __forceinline__
float mod<float>(const float& x, const float& y) { return ::fmodf(x,y); }
template <> __device__ __forceinline__
double mod<double>(const double& x, const double& y) { return ::fmod(x,y); }
WRAP_BINARY_CUDA_MATH_FUNCTION(pow)
#undef WRAP_BINARY_CUDA_MATH_FUNCTION

template <> __device__ __forceinline__
__half pow<__half>(const __half& x, const __half& y)
{ return pow(float(x), float(y)); }

template <> __device__ __forceinline__
__half mod<__half>(const __half& x, const __half& y)
{ return mod(float(x), float(y)); }

#if __CUDA_ARCH__ >= 530
template <> __device__ __forceinline__
__half min<__half>(const __half& x, const __half& y)
{ return ::__hle(x, y) ? x : y; }

template <> __device__ __forceinline__
__half max<__half>(const __half& x, const __half& y)
{ return ::__hle(x, y) ? y : x; }
#endif // __CUDA_ARCH__ >= 530

// Numeric limits
#ifdef __CUDACC_RELAXED_CONSTEXPR__
template <typename T> constexpr __device__ __forceinline__ T min() {
  return std::numeric_limits<T>::min();
}
template <typename T> constexpr __device__ __forceinline__ T max() {
  return std::numeric_limits<T>::min();
}
template <typename T> constexpr __device__ __forceinline__ T epsilon() {
  return std::numeric_limits<T>::epsilon();
}
template <typename T> __device__ __forceinline__ T infinity() {
  return std::numeric_limits<T>::infinity();
}
#else // __CUDACC_RELAXED_CONSTEXPR__
#define SPECIFIERS template <> __device__ __forceinline__
SPECIFIERS constexpr float min<float>()                 { return FLT_MIN;   }
SPECIFIERS constexpr double min<double>()               { return DBL_MIN;   }
SPECIFIERS constexpr int min<int>()                     { return INT_MIN;   }
SPECIFIERS constexpr long int min<long int>()           { return LONG_MIN;  }
SPECIFIERS constexpr long long int min<long long int>() { return LLONG_MIN; }
SPECIFIERS constexpr float max<float>()                 { return FLT_MAX;   }
SPECIFIERS constexpr double max<double>()               { return DBL_MAX;   }
SPECIFIERS constexpr int max<int>()                     { return INT_MAX;   }
SPECIFIERS constexpr long int max<long int>()           { return LONG_MAX;  }
SPECIFIERS constexpr long long int max<long long int>() { return LLONG_MAX; }
SPECIFIERS constexpr float epsilon<float>()   { return FLT_EPSILON; }
SPECIFIERS constexpr double epsilon<double>() { return DBL_EPSILON; }
SPECIFIERS float infinity<float>()   { return CUDART_INF_F; }
SPECIFIERS double infinity<double>() { return CUDART_INF;   }
#undef SPECIFIERS
#endif // __CUDACC_RELAXED_CONSTEXPR__

// FIXME (TRB): I think this is right? Borrowed the values from the
// sourceforge half library.
template <> __device__ __forceinline__ __half min<__half>() {
  return __short_as_half(0x0400);
}
template <> __device__ __forceinline__ __half max<__half>() {
  return __short_as_half(0x7BFF);
}
template <> __device__ __forceinline__ __half epsilon<__half>() {
  return __short_as_half(0x1400);
}
template <> __device__ __forceinline__ __half infinity<__half>() {
  return __short_as_half(0x7C00);
}

// Array member functions
template <typename T, size_t N>
__host__ __device__ __forceinline__
size_t array<T,N>::size() const {
  return N;
}
template <typename T, size_t N>
__host__ __device__ __forceinline__
T& array<T,N>::operator[](size_t i) {
  return vals[i];
}
template <typename T, size_t N>
__host__ __device__ __forceinline__
const T& array<T,N>::operator[](size_t i) const {
  return vals[i];
}

#endif // __CUDACC__

// -------------------------------------------------------------
// Helper functions for entrywise operations
// -------------------------------------------------------------
#ifdef __CUDACC__

/** CUDA kernel to apply an entry-wise unary operator. */
template <template <typename> class UnaryOperator, typename TensorDataType>
__global__
void entrywise_unary_operator_kernel(El::Int height, El::Int width,
                                     const TensorDataType* __restrict__ input,
                                     El::Int input_ldim,
                                     TensorDataType* __restrict__ output,
                                     El::Int output_ldim) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int size = height * width;
  const El::Int num_threads = blockDim.x * gridDim.x;
  UnaryOperator<TensorDataType> op;
  for (El::Int pos = gid; pos < size; pos += num_threads) {
    const auto& row = pos % height;
    const auto& col = pos / height;
    const auto& x = input[row + col * input_ldim];
    auto& y = output[row + col * output_ldim];
    y = op(x);
  }
}

/** CUDA kernel to apply an entry-wise binary operator. */
template <template <typename> class BinaryOperator, typename TensorDataType>
__global__
void entrywise_binary_operator_kernel(El::Int height, El::Int width,
                                     const TensorDataType* __restrict__ input1,
                                     El::Int input1_ldim,
                                     const TensorDataType* __restrict__ input2,
                                     El::Int input2_ldim,
                                     TensorDataType* __restrict__ output,
                                     El::Int output_ldim) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int size = height * width;
  const El::Int num_threads = blockDim.x * gridDim.x;
  BinaryOperator<TensorDataType> op;
  for (El::Int pos = gid; pos < size; pos += num_threads) {
    const auto& row = pos % height;
    const auto& col = pos / height;
    const auto& x1 = input1[row + col * input1_ldim];
    const auto& x2 = input2[row + col * input2_ldim];
    auto& y = output[row + col * output_ldim];
    y = op(x1, x2);
  }
}

/** Apply an entry-wise unary operator to GPU data.
 *  The input and output data must be on GPU and must have the same
 *  dimensions.
 */
template <template <typename> class UnaryOp, typename TensorDataType>
void apply_entrywise_unary_operator(
  const El::AbstractMatrix<TensorDataType>& input,
  El::AbstractMatrix<TensorDataType>& output) {

  // Check that input and output are valid
  if (input.GetDevice() != El::Device::GPU) {
    LBANN_ERROR("input is not on GPU");
  } else if (output.GetDevice() != El::Device::GPU) {
    LBANN_ERROR("output is not on GPU");
  } else if (input.Height() != output.Height()
             || input.Width() != output.Width()) {
    LBANN_ERROR("input matrix dimensions "
                "(", input.Height(), " x ", input.Width(), ")"
                "don't match output matrix dimensions "
                "(", output.Height(), " x ", output.Width(), ")");
  }

  // Get CUDA grid dimensions
  // Note: Maximum CUDA grid dimension is 2^32-1
  // (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications).
  const El::Int height = input.Height();
  const El::Int width = input.Width();
  const El::Int block_dim = 256;
  El::Int grid_dim = (height * width + block_dim - 1) / block_dim;
  if (sizeof(El::Int) > sizeof(unsigned int)
      && grid_dim > std::numeric_limits<uint32_t>::max()) {
    grid_dim = std::numeric_limits<uint32_t>::max();
  }

  // Launch CUDA kernel
  if (grid_dim > 0) {
    CHECK_CUDA(cudaSetDevice(El::GPUManager::Device()));
    entrywise_unary_operator_kernel<UnaryOp>
      <<<grid_dim, block_dim, 0, El::GPUManager::Stream()>>>(
        height, width, input.LockedBuffer(), input.LDim(),
        output.Buffer(), output.LDim());
  }

}

/** Apply an entry-wise binary operator to GPU data.
 *  The input and output data must be on GPU and must have the same
 *  dimensions.
 */
template <template <typename> class BinaryOp, typename TensorDataType>
void apply_entrywise_binary_operator(
  const El::AbstractMatrix<TensorDataType>& input1,
  const El::AbstractMatrix<TensorDataType>& input2,
  El::AbstractMatrix<TensorDataType>& output) {

  // Check that input and output are valid
  if (input1.GetDevice() != El::Device::GPU
      || input2.GetDevice() != El::Device::GPU) {
    LBANN_ERROR("input is not on GPU");
  } else if (output.GetDevice() != El::Device::GPU) {
    LBANN_ERROR("output is not on GPU");
  } else if (input1.Height() != input2.Height()
             || input1.Width() != input2.Width()
             || input1.Height() != output.Height()
             || input1.Width() != output.Width()) {
    LBANN_ERROR("input matrix dimensions "
                "(", input1.Height(), " x ", input1.Width(), ", ",
                input2.Height(), " x ", input2.Width(), ")"
                "don't match output matrix dimensions "
                "(", output.Height(), " x ", output.Width(), ")");
  }

  // Get CUDA grid dimensions
  // Note: Maximum CUDA grid dimension is 2^32-1
  // (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications).
  const El::Int height = input1.Height();
  const El::Int width = input1.Width();
  const El::Int block_dim = 256;
  El::Int grid_dim = (height * width + block_dim - 1) / block_dim;
  if (sizeof(El::Int) > sizeof(unsigned int)
      && grid_dim > std::numeric_limits<uint32_t>::max()) {
    grid_dim = std::numeric_limits<uint32_t>::max();
  }

  // Launch CUDA kernel
  if (grid_dim > 0) {
    CHECK_CUDA(cudaSetDevice(El::GPUManager::Device()));
    entrywise_binary_operator_kernel<BinaryOp>
      <<<grid_dim, block_dim, 0, El::GPUManager::Stream()>>>(
        height, width,
        input1.LockedBuffer(), input1.LDim(),
        input2.LockedBuffer(), input2.LDim(),
        output.Buffer(), output.LDim());
  }

}

/** Apply an entry-wise unary operator to GPU data.
 *  The input and output data must be on GPU, have the same
 *  dimensions, and be aligned.
 */
template <template <typename> class UnaryOperator, typename TensorDataType>
void apply_entrywise_unary_operator(
  const El::AbstractDistMatrix<TensorDataType>& input,
  El::AbstractDistMatrix<TensorDataType>& output) {
  if (input.Height() != output.Height()
      || input.Width() != output.Width()) {
    LBANN_ERROR("input matrix dimensions "
                "(", input.Height(), " x ", input.Width(), ")"
                "don't match output matrix dimensions "
                "(", output.Height(), " x ", output.Width(), ")");
  } else if (input.DistData() != output.DistData()) {
    LBANN_ERROR("input and output matrix distributions don't match");
  }
  apply_entrywise_unary_operator<UnaryOperator>(input.LockedMatrix(),
                                                output.Matrix());
}

/** Apply an entry-wise binary operator to GPU data.
 *  The input and output data must be on GPU, have the same
 *  dimensions, and be aligned.
 */
template <template <typename> class BinaryOperator, typename TensorDataType>
void apply_entrywise_binary_operator(
  const El::AbstractDistMatrix<TensorDataType>& input1,
  const El::AbstractDistMatrix<TensorDataType>& input2,
  El::AbstractDistMatrix<TensorDataType>& output) {
  if (input1.Height() != input2.Height()
      || input1.Width() != input2.Width()
      || input1.Height() != output.Height()
      || input1.Width() != output.Width()) {
    LBANN_ERROR("input matrix dimensions "
                "(", input1.Height(), " x ", input1.Width(), ", ",
                input2.Height(), " x ", input2.Width(), ")"
                "don't match output matrix dimensions "
                "(", output.Height(), " x ", output.Width(), ")");
  } else if (input1.DistData() != input2.DistData()
             || input1.DistData() != output.DistData()) {
    LBANN_ERROR("input and output matrix distributions don't match");
  }
  apply_entrywise_binary_operator<BinaryOperator>(input1.LockedMatrix(),
                                                  input2.LockedMatrix(),
                                                  output.Matrix());
}

#endif // __CUDACC__

// -------------------------------------------------------------
// Utilities for Thrust
// -------------------------------------------------------------
#ifndef DOXYGEN_SHOULD_SKIP_THIS

namespace thrust {

template <typename T>
allocator<T>::allocator(cudaStream_t stream)
  : m_stream(stream),
    m_system(stream) {}

template <typename T>
typename allocator<T>::pointer allocator<T>::allocate(allocator<T>::size_type size) {
  value_type* buffer = nullptr;
  if (size > 0) {
#ifdef HYDROGEN_HAVE_CUB
    auto& memory_pool = El::cub::MemoryPool();
    CHECK_CUDA(memory_pool.DeviceAllocate(reinterpret_cast<void**>(&buffer),
                                          size * sizeof(value_type),
                                          m_stream));
#else
    CHECK_CUDA(cudaMalloc(&buffer, size * sizeof(value_type)));
#endif // HYDROGEN_HAVE_CUB
  }
  return pointer(buffer);
}

template <typename T>
void allocator<T>::deallocate(allocator<T>::pointer buffer,
                              allocator<T>::size_type size) {
  auto&& ptr = buffer.get();
  if (ptr != nullptr) {
#ifdef HYDROGEN_HAVE_CUB
    auto& memory_pool = El::cub::MemoryPool();
    CHECK_CUDA(memory_pool.DeviceFree(ptr));
#else
    CHECK_CUDA(cudaFree(ptr));
#endif // HYDROGEN_HAVE_CUB
  }
}

template <typename T>
typename allocator<T>::system_type& allocator<T>::system() {
  return m_system;
}

} // namespace thrust
#endif // !DOXYGEN_SHOULD_SKIP_THIS

} // namespace cuda
} // namespace lbann
