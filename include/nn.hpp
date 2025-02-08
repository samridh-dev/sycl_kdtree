/*!
 * \file        nn.hpp
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

#ifndef KDTREE_NN_HPP
#define KDTREE_NN_HPP

#include "pch.hpp"
#include "container.hpp"

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
T
nn(const kdtree::context& ctx, const C_query& q, const C_tree& tree, 
   const T n, const F rmax = std::numeric_limits<F>::max());

}

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

namespace kdtree   {
namespace internal {
namespace nn       {

template <typename F, typename T>
requires std::is_arithmetic_v<F> && std::is_integral_v<T>
struct result_t {
  F dst;
  T idx;
  result_t() : dst{std::numeric_limits<F>::max()}, idx{0} {}
};

template <typename F, typename T, T dim, kdtree::container::layout maj,
          typename C_query, typename C_tree>
struct f_process {

  void 
  operator()(result_t<F, T>& res, const C_query& q, const C_tree& src,
             const T n, const T idx, F* rmax) const {

    using kdtree::internal::dist::euclidian;

    F dst{
      euclidian<F, T, dim, maj, C_query, maj, C_tree>(q, 1, 0, src, n, idx)
    };

    if (dst < res.dst) {

      res.dst = dst;
      res.idx = idx;

      if (dst < *rmax) {
        *rmax = dst;
      }

    }

  }
};

}  // namespace nn
}  // namespace internal
}  // namespace kdtree

template<typename F, typename T, T dim, kdtree::container::layout maj,
         typename C_query, typename C_tree> 
requires kdtree::container::container_1d<C_query> 
      && kdtree::container::container<C_tree>
      && std::is_integral_v<T>
      && std::is_arithmetic_v<F>
      && std::is_same_v<kdtree::container::get_primitive_t<C_query>,
                        kdtree::container::get_primitive_t<C_tree>>
T
kdtree::nn(const kdtree::context& ctx, const C_query& q, const C_tree& tree, 
           const T n, const F rmax) {

  using kdtree::internal::traverse::f_splitdim;
  using kdtree::internal::nn::f_process;
  using kdtree::internal::nn::result_t;

  (void) ctx;

  struct result_t<F, T> result;

  kdtree::traverse<
    result_t<F, T>,
    f_process<F, T, dim, maj, C_query, C_tree>,
    f_splitdim<T, dim, maj, C_tree>,
    F, T, dim, maj,
    C_query, C_tree
  > (result, q, tree, n, rmax);

  return result.idx;

}

#endif // KDTREE_NN_HPP

