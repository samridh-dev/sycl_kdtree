/*!
 * \file        libsycl.hpp
 * \author      Samridh D. Singh
 * \date        2025-02-01
 * \brief       
 * \details     
 *
 * \copyright   This file is part of the sycl_kdtree project.
 * \copyright   Copyright (C) 2025, Samridh D. Singh
 * \copyright   
 *              sycl_kdtree is free software: you can redistribute it and/or
 *              modify it under the terms of the GNU General Public License as
 *              published by the Free Software Foundation, either version 3 of
 *              the License, or (at your option) any later version.
 *
 *              sycl_kdtree is distributed in the hope that it will be useful,
 *              but WITHOUT ANY WARRANTY; without even the implied warranty of
 *              MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *              GNU General Public License for more details.
 *
 *              A copy of the GNU General Public License should be provided
 *              along with sycl_kdtree. 
 *              If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef KDTREE_PRECOMPILED_HEADER_HPP
#define KDTREE_PRECOMPILED_HEADER_HPP

//#include <concepts> 
//#include <type_traits>

#include <cstddef>
#include <stdexcept>

#include <concepts>
#include <type_traits>
#include <thread>

// NOTE: tested for the following compilers
#if defined(__INTEL_LLVM_COMPILER) || defined(__ADAPTIVECPP__) 
  #include <sycl/sycl.hpp>
  #define KD__USING_SYCL
  #define KD__IMPLEMENTATION SYCL_EXTERNAL
#else
  #define KD__IMPLEMENTATION
#endif

namespace kdtree {

#ifdef KD__USING_SYCL

struct context {
  sycl::queue& queue;
  context() : queue(sycl::default_selector{}) {}
  explicit context(sycl::queue q) : queue(std::move(q)) {}
};

#else

struct context {
  std::size_t nthreads;
  context() : nthreads(std::thread::hardware_concurrency()) {}
  explicit context(std::size_t threads) : nthreads(threads) {}
};

#endif // KD__USING_SYCL

} // namespace kdtree

#endif // KDTREE_PRECOMPILED_HEADER_HPP
