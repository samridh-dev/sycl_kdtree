/*!
 * \file        traverse/traverse.hpp
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

#ifndef KDTREE_TRAVERSE_HPP
#define KDTREE_TRAVERSE_HPP

#include "pch.hpp"
#include "../container.hpp"

namespace kdtree {

template<typename result_t, typename f_process, typename f_splitdim, 
         typename F, typename T, T dim, kdtree::container::layout maj,
         typename C_query, typename C_tree> 
requires kdtree::container::container_1d<C_query> 
      && kdtree::container::container<C_tree>
      && std::is_integral_v<T>
      && std::is_arithmetic_v<F>
      && std::is_same_v<kdtree::container::get_primitive_t<C_query>,
                        kdtree::container::get_primitive_t<C_tree>>
constexpr void
traverse(result_t& result, const C_query& q, const C_tree& tree, 
         const T n, F rmax);

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

#include "../internal/bsr.hpp"
namespace kdtree {
namespace internal {
namespace traverse {

template <typename T, T dim, kdtree::container::layout maj, typename C>
requires kdtree::container::container<C> && std::integral<T>
struct f_splitdim {

  constexpr inline T
  operator()(const C& v, const T s) {
    (void) v;
    const T l {kdtree::internal::bsr(s+1)};
    const T d {l % dim};
    return d;
  }

};

} // namespace traverse
} // namespace internal
} // namespace kdtree

#include "../internal/abs.hpp"

template<typename result_t, typename f_process, typename f_splitdim, 
         typename F, typename T, T dim, kdtree::container::layout maj,
         typename C_query, typename C_tree> 
requires kdtree::container::container_1d<C_query> 
      && kdtree::container::container<C_tree>
      && std::is_integral_v<T>
      && std::is_arithmetic_v<F>
      && std::is_same_v<kdtree::container::get_primitive_t<C_query>,
                        kdtree::container::get_primitive_t<C_tree>>
constexpr void
kdtree::traverse(result_t& result, const C_query& q, const C_tree& tree, 
                 const T n, F rmax) {

  T curr{0};
  T prev{-1};

  using kdtree::container::id;
  using kdtree::internal::abs;

  while (1) {

    const bool from_parent { prev < curr              };
    const T         parent { (curr + 1) / T{2} - T{1} };

    if (curr >= n) {
      prev = curr;
      curr = parent;
      continue;
    }

    if (from_parent) {
      f_process{}(result, q, tree, n, curr, &rmax);
    }

    const auto s_dim        { f_splitdim{}(tree, curr)                 };
    const auto s_pos        { id<T, dim,  maj>(tree, n,   curr, s_dim) };
    const auto q_pos        { id<T, T{1}, maj>(q,    dim, T{0}, s_dim) };
    const auto sign_dist    { static_cast<F>(q_pos - s_pos)            };
    const auto close_side   { sign_dist > F{0}                         };
    const auto close_child  { T{2} * curr + T{1} + close_side          };
    const auto far_child    { T{2} * curr + T{2} - close_side          };
    const auto far_in_range { abs(sign_dist) <= rmax                   };

    T next;
    if (from_parent) {
      next = close_child;
    } else if (prev == close_child) {
      next = far_in_range ? far_child :parent;
    } else {
      next = parent;
    }

    if (next == T{-1}) {
      return;
    }

    prev = curr;
    curr = next;

  }

}

#endif  // KDTREE_TRAVERSE_HPP
