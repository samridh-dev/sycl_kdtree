/*!
 * \file        knn.hpp
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

#ifndef KDTREE_KNN_HPP
#define KDTREE_KNN_HPP

#include "pch.hpp"
#include "container.hpp"
#include <limits>
#include <vector>

namespace kdtree {

template<typename F, typename T, T dim,
         kdtree::container::layout maj = kdtree::container::layout::row_major,
         typename C_query, typename C_tree> 

requires kdtree::container::container_1d<C_query> 
      && kdtree::container::container<C_tree>
      && std::is_integral_v<T>
      && std::is_arithmetic_v<F>
      && std::is_same_v<kdtree::container::get_primitive_t<C_query>,
                        kdtree::container::get_primitive_t<C_tree>>

constexpr std::vector<T>
knn(const kdtree::context& ctx,
    const C_query&        q,
    const C_tree&         tree,
    const T               n,
    const T               k,
    const F               rmax = std::numeric_limits<F>::max());

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

#include "traverse/traverse.hpp"
#include "internal/dist.hpp"

namespace kdtree {
namespace internal {
namespace knn {

template <typename T, T dim, kdtree::container::layout maj,
          typename C_idx, typename C_dst>
requires std::is_integral_v<T>
      && kdtree::container::container_1d<C_idx>
      && kdtree::container::container_1d<C_dst>
      && std::is_same_v<kdtree::container::get_primitive_t<C_idx>, T>
      && std::is_arithmetic_v<kdtree::container::get_primitive_t<C_dst>>

constexpr void
maxheapify(C_idx& idx, C_dst& dst, const T k, const T i = T{0}) {

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
heapsort(C_idx& idx, C_dst& dst, const T k) {
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

template <typename F, typename T>
requires
    std::is_arithmetic_v<F> &&
    std::is_integral_v<T>
struct result_t {

  std::vector<T> idx;
  std::vector<F> dst;
  const T        k;

  result_t(T k_)
    : idx(static_cast<std::size_t>(k_), T{0}),
      dst(static_cast<std::size_t>(k_), std::numeric_limits<F>::max()),
      k{k_}
  {}
};

template <typename F, typename T, T dim, kdtree::container::layout maj,
          typename C_query, typename C_tree>
struct f_process {
  void operator()(
      result_t<F, T>&   res,
      const C_query&    q,
      const C_tree&     src,
      const T           n,
      const T           idx,
      F*                rmax
  ) const {
    using kdtree::internal::dist::euclidian;

    const F dst {
      euclidian<F, T, dim, maj, C_query, maj, C_tree>(q, 1, 0, src, n, idx)
    };

    if (dst < res.dst[0]) {
      res.dst[0] = dst;
      res.idx[0] = idx;

      if (dst < *rmax) {
        *rmax = dst;
      }

      maxheapify<T, dim, maj>(res.idx, res.dst, res.k);

    }
  }
};

} // namespace knn
} // namespace internal
} // namespace kdtree


//---------------------------------------------------------------------------------------
//  kdtree::knn IMPLEMENTATION
//---------------------------------------------------------------------------------------
template<typename F, typename T, T dim, kdtree::container::layout maj,
         typename C_query, typename C_tree> 
requires
    kdtree::container::container_1d<C_query> &&
    kdtree::container::container<C_tree> &&
    std::is_integral_v<T> &&
    std::is_arithmetic_v<F> &&
    std::is_same_v<kdtree::container::get_primitive_t<C_query>,
                   kdtree::container::get_primitive_t<C_tree>>
constexpr std::vector<T>
kdtree::knn(const kdtree::context& ctx,
            const C_query&        q,
            const C_tree&         tree,
            const T               n,
            const T               k,
            const F               rmax) {

  using kdtree::internal::traverse::f_splitdim;
  using kdtree::internal::knn::f_process;
  using kdtree::internal::knn::result_t;

  (void) ctx;

  result_t<F, T> result(k);

  kdtree::traverse<
    result_t<F, T>,
    f_process<F, T, dim, maj, C_query, C_tree>,
    f_splitdim<T, dim, maj, C_tree>,
    F, T, dim, maj,
    C_query, C_tree
  >(result, q, tree, n, rmax);

  kdtree::internal::knn::heapsort<T, dim, maj>(result.idx, result.dst, k);

  return result.idx;
}

#endif // KDTREE_KNN_HPP
