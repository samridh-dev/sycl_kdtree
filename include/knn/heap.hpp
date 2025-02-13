/*!
 * \file        knn/heap.hpp
 * \author      Samridh D. Singh
 * \date        2025-02-01
 * \brief       containers for templates 
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

#ifndef KDTREE_KNN_HEAP_HPP
#define KDTREE_KNN_HEAP_HPP

#include "../pch.hpp"
#include "../container.hpp"

namespace kdtree   {
namespace internal {
namespace knn      {

template <typename T, T dim, kdtree::container::layout maj,
          typename C_idx, typename C_dst>
requires std::is_integral_v<T>
      && kdtree::container::container_1d<C_idx>
      && kdtree::container::container_1d<C_dst>
      && std::is_same_v<kdtree::container::get_primitive_t<C_idx>, T>
      && std::is_arithmetic_v<kdtree::container::get_primitive_t<C_dst>>

constexpr void
maxheapify(C_idx& idx, C_dst& dst, const T k, const T i = T{0});


template <typename T, T dim, kdtree::container::layout maj,
          typename C_idx, typename C_dst>

requires std::is_integral_v<T>
      && kdtree::container::container_1d<C_idx>
      && kdtree::container::container_1d<C_dst>
      && std::is_same_v<kdtree::container::get_primitive_t<C_idx>, T>
      && std::is_arithmetic_v<kdtree::container::get_primitive_t<C_dst>>

constexpr void
heapsort(C_idx& idx, C_dst& dst, const T k);

} // namespace kdtree
} // namespace internal
} // namespace knn

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


template <typename T, T dim, kdtree::container::layout maj,
          typename C_idx, typename C_dst>
requires std::is_integral_v<T>
      && kdtree::container::container_1d<C_idx>
      && kdtree::container::container_1d<C_dst>
      && std::is_same_v<kdtree::container::get_primitive_t<C_idx>, T>
      && std::is_arithmetic_v<kdtree::container::get_primitive_t<C_dst>>

constexpr void
kdtree::internal::knn::maxheapify(C_idx& idx, C_dst& dst, const T k, const T i) 
{
  using namespace kdtree::container;

  T j{i};
  while (1) {
    T b       {     j     };
    const T l { 2 * j + 1 };
    const T r { 2 * j + 2 };
    if (l < k && id<T>(dst, k, l) > id<T>(dst, k, b)) b = l;
    if (r < k && id<T>(dst, k, r) > id<T>(dst, k, b)) b = r;
    if (b == j) break;
    swap<T, T{1}, maj>(idx, k, j, b);
    swap<T, T{1}, maj>(dst, k, j, b);
    j = b;
  }

}

template <typename T, T dim, kdtree::container::layout maj,
          typename C_idx, typename C_dst>
requires std::is_integral_v<T>
      && kdtree::container::container_1d<C_idx>
      && kdtree::container::container_1d<C_dst>
      && std::is_same_v<kdtree::container::get_primitive_t<C_idx>, T>
      && std::is_arithmetic_v<kdtree::container::get_primitive_t<C_dst>>

constexpr void
kdtree::internal::knn::heapsort(C_idx& idx, C_dst& dst, const T k) {
  for (T i{k / 2}; i > 0; ) {
    --i;
    maxheapify<T, dim, maj>(idx, dst, k, i);
  }
  for (T i{k - 1}; i > 0; --i) {
    kdtree::container::swap<T, T{1}, maj>(idx, k, 0, i);
    kdtree::container::swap<T, T{1}, maj>(dst, k, 0, i);
    maxheapify<T, dim, maj>(idx, dst, i, 0);
  }
}

#endif // KDTREE_KNN_HEAP_HPP
