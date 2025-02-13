/*!
 * \file        create/create.hpp
 * \author      Samridh D. Singh
 * \date        2025-02-01
 * \brief       kdtree create header and implementation
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

#ifndef KDTREE_CREATE_HPP
#define KDTREE_CREATE_HPP


#include "../container.hpp"

namespace kdtree {

template <typename T, T dim, 
          kdtree::container::layout maj = kdtree::container::layout::row_major, 
          typename C, typename N>
requires kdtree::container::container<C> 
      && std::is_integral_v<T>
      && std::is_integral_v<N>
void
create(kdtree::context& ctx, C& src, const N n);

} // namespace kdtree
  
///////////////////////////////////////////////////////////////////////////////
///                                                                         ///
///                                                                         ///
///                                                                         ///
///                                                                         ///
///                                                                         ///
///                             IMPLEMENTATION                              ///
///                                                                         ///
///                                                                         ///
///                                                                         ///
///                                                                         ///
///                                                                         ///
///////////////////////////////////////////////////////////////////////////////

#include "internal/payload.hpp"
#include "internal/F.hpp"

#include "../internal/bsr.hpp"
#include "../sort/sort.hpp"

#include "tags.hpp"

#define USE_BENCHMARK 0

#if USE_BENCHMARK
#include <chrono>
#endif

template <typename T, T dim, kdtree::container::layout maj, 
          typename C, typename N>
requires kdtree::container::container<C> 
      && std::is_integral_v<T>
      && std::is_integral_v<N>
void
kdtree::create(kdtree::context& ctx, C& src, const N n) {

  using namespace kdtree::internal::create;

  const T n_{static_cast<T>(n)};

  std::vector<T> tag(n_, 0);

  for (T l{0}; l < kdtree::internal::bsr(n_) + T{1}; ++l) {

    #if USE_BENCHMARK
    auto beg = std::chrono::high_resolution_clock::now();
    #endif

    {
      const T d  {l % dim}; // TODO: template this away
      const T n0 {0};
      payload<T, dim, maj, decltype(src), decltype(tag)> p(src, tag, n_, d);
      kdtree::sort(ctx, p, n0, n_);
    }

    #if USE_BENCHMARK
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg);
    std::cout << "\t[sort] time: " << dur.count() << " ms" << std::endl;
    #endif

    #if USE_BENCHMARK
    beg = std::chrono::high_resolution_clock::now();
    #endif

    tags::update(ctx, tag, n_, l);

    #if USE_BENCHMARK
    end = std::chrono::high_resolution_clock::now();
    dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg);
    std::cout << "\t[update] time: " << dur.count() << " ms" << std::endl;
    #endif

  }

}

#endif // KDTREE_CREATE_HPP
