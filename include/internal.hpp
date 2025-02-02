/*!
 * \file        internal.hpp
 * \author      Samridh D. Singh
 * \date        2025-02-01
 * \brief       general internal functions library.
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

#ifndef KDTREE_INTERNAL_HPP
#define KDTREE_INTERNAL_HPP

#include "container.hpp"
#include "layout.hpp"

#include <type_traits>
#include <climits>

namespace kdtree {
namespace internal {

template <typename Ts, Ts dim, kdtree::layout maj, typename C>
constexpr auto& 
id(C& v, const Ts n, const Ts i, const Ts j) {

  if constexpr (container::is_1d_v<C>) {
    if constexpr (maj == kdtree::layout::rowmajor) {
      return v[dim * i + j];
    } else {
      return v[n * j + i];
    }
  } 

  else if constexpr (container::is_2d_v<C>) {
    if constexpr (maj == kdtree::layout::rowmajor) {
      return v[i][j];
    } else {
      return v[j][i];
    }
  }

  else {
    static_assert(!std::is_same_v<C, C>,
      "`C` must be a container of fundamental types or a "
      "container of containers of fundamental types.\n"
      "\tvalid:   std::vector<int>,\n"
      "\t     std::vector<std::array<int, 3>>\n"
      "\tinvalid: std::vector<std::vector<std::vector<int>>>"
    );
  }
}

template <typename Ts, Ts dim, kdtree::layout maj, typename C>
constexpr void 
swap(C& v, const Ts n, const Ts i, const Ts j) {

  using kdtree::internal::id;
  static_assert(std::is_same_v<decltype(dim), Ts>, "dim is not of type Ts.");

  auto lswap = [&](Ts idx) {
    const auto tmp = id<Ts, dim, maj>(v, n, i, idx);
    id<Ts, dim, maj>(v, n, i, idx) = id<Ts, dim, maj>(v, n, j, idx);
    id<Ts, dim, maj>(v, n, j, idx) = tmp;
  };

  if constexpr (dim > 8) {
    for (Ts d = 0; d < dim; ++d) lswap(d);
  } else {
    if constexpr (dim >= 1) lswap(0);
    if constexpr (dim >= 2) lswap(1);
    if constexpr (dim >= 3) lswap(2);
    if constexpr (dim >= 4) lswap(3);
    if constexpr (dim >= 5) lswap(4);
    if constexpr (dim >= 6) lswap(5);
    if constexpr (dim >= 7) lswap(6);
    if constexpr (dim >= 8) lswap(7);
  }

}


// function assumes that n is positive
template <typename T>
constexpr T 
clz(const T n) {

  static_assert(std::is_integral<T>::value, "clz requires an integral type");

#if defined(__GNU__) || defined(__clang__)

  if constexpr (sizeof(T) == sizeof(unsigned int)) {
    return __builtin_clz(n);
  } else if constexpr (sizeof(T) == sizeof(unsigned long)) {
    return __builtin_clzl(n);
  } else if constexpr (sizeof(T) == sizeof(unsigned long long)) {
    return __builtin_clzll(n);
  } else if constexpr (sizeof(T) < sizeof(unsigned int)) {
    return __builtin_clz(static_cast<unsigned int>(n)) 
         - ((sizeof(unsigned int) + sizeof(T)) * CHAR_BIT);
  }

#else

  T cnt = 0;
  T cur = sizeof(T) * CHAR_BIT;

  if (n == 0) {
    return sizeof(T) * CHAR_BIT;
  }

  while (cur > 0) {
    --cur;
    if (n & (T(1) << cur)) {
      break;
    }
    ++cnt;
  }

  return cnt;

#endif

}

template <typename T>
constexpr T 
abs(const T n) {
  return (n < T(0)) ? -n : n;
}

template <typename T>
constexpr T 
min(const T a, const T b) {
  return (a < b) ? a : b;
}

template <typename T>
constexpr T 
max(const T a, const T b) {
  return (a > b) ? a : b;
}


namespace dist {

template <typename Ts, typename Tf, Ts dim, kdtree::layout maj,
          typename Cx, typename Cy>
constexpr Tf
euclidian(const Cx& x, const Ts x_n, const Cy& y, const Ts y_n,
          const Ts x_i, const Ts y_i) {

  static_assert(std::is_same_v<container::get_primitive_type_t<Cx>,
                               container::get_primitive_type_t<Cy>>,
                "`Cx` and `Cy` must have the same primitive type");

  using Tv = container::get_primitive_type_t<Cx>;

  const auto d = [&](const Ts i_) constexpr { 
    const auto sq = [](const Tf v_) constexpr { return v_ * v_; };
    return sq(static_cast<Tf>(id<Ts, dim, maj>(x, x_n, x_i, i_)) - 
              static_cast<Tf>(id<Ts, dim, maj>(y, y_n, y_i, i_)));
  };

  if constexpr (dim > 8) {
    Tf sum = Tf{0};
    for (Ts i = 0; i < dim; ++i) sum += d(i);
    return sum;
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

} // namespace dist


} // namespace internal 
} // namespace kdtree 

#endif // KDTREE_COMMON_HPP
