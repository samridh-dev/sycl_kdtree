/*!
 * \file        internal/dist.hpp
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

#ifndef KDTREE_INTERNAL_DIST
#define KDTREE_INTERNAL_DIST

#include "../pch.hpp"
#include "../container.hpp"

namespace kdtree {
namespace internal {
namespace dist {

template <typename F, typename T, T dim, 
          kdtree::container::layout maj_x, typename C_x,
          kdtree::container::layout maj_y, typename C_y>

requires std::is_arithmetic_v<F> && std::is_integral_v<T>
      && kdtree::container::container<C_x> 
      && kdtree::container::container<C_y>
      && std::is_same_v<kdtree::container::get_primitive_t<C_x>,
                        kdtree::container::get_primitive_t<C_y>>

constexpr inline F
euclidian(const C_x& x, const T x_n, const T x_i,
          const C_y& y, const T y_n, const T y_i);

} // namespace dist
} // namespace internal
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

template <typename F, typename T, T dim, 
          kdtree::container::layout maj_x, typename C_x,
          kdtree::container::layout maj_y, typename C_y>

requires std::is_arithmetic_v<F> && std::is_integral_v<T>
      && kdtree::container::container<C_x> 
      && kdtree::container::container<C_y>
      && std::is_same_v<kdtree::container::get_primitive_t<C_x>,
                        kdtree::container::get_primitive_t<C_y>>

constexpr inline F
kdtree::internal::dist::euclidian(const C_x& x, const T x_n, const T x_i,
                                  const C_y& y, const T y_n, const T y_i) {

  const auto d = [&](const T i_) constexpr {
    using kdtree::container::id;
    const auto sq = [](const F v_) constexpr { return v_ * v_; };
    return sq(static_cast<F>(id<T, dim, maj_x>(x, x_n, x_i, i_)) -
              static_cast<F>(id<T, dim, maj_y>(y, y_n, y_i, i_)));
  };

  if constexpr (dim > 8) {
    F v{0};
    for (T i{0}; i < dim; ++i) v += d(i);
    return v;
  } else {
    if constexpr (dim == 1) return d(0);
    if constexpr (dim == 2) return d(0)+d(1);
    if constexpr (dim == 3) return d(0)+d(1)+d(2);
    if constexpr (dim == 4) return d(0)+d(1)+d(2)+d(3);
    if constexpr (dim == 5) return d(0)+d(1)+d(2)+d(3)+d(4);
    if constexpr (dim == 6) return d(0)+d(1)+d(2)+d(3)+d(4)+d(5);
    if constexpr (dim == 7) return d(0)+d(1)+d(2)+d(3)+d(4)+d(5)+d(6);
    if constexpr (dim == 8) return d(0)+d(1)+d(2)+d(3)+d(4)+d(5)+d(6)+d(7);
  }
}

#endif // KDTREE_INTERNAL_DIST
