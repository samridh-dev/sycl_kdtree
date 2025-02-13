/*!
 * \file        container.hpp
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

#ifndef KDTREE_INTERNAL_CONTAINER_HPP
#define KDTREE_INTERNAL_CONTAINER_HPP

#include "pch.hpp"

namespace kdtree {
namespace container {

enum layout { row_major, col_major };

template <typename C>
concept container_1d =
  requires(C c, std::size_t i) { { c[i] }; } 
  &&
  std::is_arithmetic_v<std::remove_cvref_t<
    decltype(std::declval<C&>()[std::declval<std::size_t>()])
  >>;

template <typename C>
concept container_2d =
  requires(C c, std::size_t i, std::size_t j) { { c[i] }; { c[i][j] }; } 
  &&
  container_1d<std::remove_cvref_t<
    decltype(std::declval<C&>()[std::declval<std::size_t>()])
  >>
  &&
  std::is_arithmetic_v<std::remove_cvref_t<
    decltype(std::declval<C&>()[std::declval<std::size_t>()]
                               [std::declval<std::size_t>()])
  >>;

template <typename C> concept container = container_1d<C> || container_2d<C>;

template <typename C>
requires container<C>
struct get_primitive_impl {
  using type = std::remove_cvref_t<
    decltype(std::declval<C&>()[std::declval<std::size_t>()])
  >;
};

template <container_1d C>
struct get_primitive_impl<C> {
  using type = std::remove_cvref_t<
    decltype(std::declval<C&>()[std::declval<std::size_t>()])
  >;
};
template <container_2d C>
struct get_primitive_impl<C> {
  using type = std::remove_cvref_t<
    decltype(std::declval<C&>()[std::declval<std::size_t>()]
                               [std::declval<std::size_t>()])
 >;
};

template <typename C>
using get_primitive_t = typename get_primitive_impl<C>::type;

// ID

template <typename T, T dim, kdtree::container::layout maj, typename C>
requires container<C> && std::is_integral_v<T>
constexpr auto&
id(C& v, const T n, const T i_, const T j_) {

  using enum kdtree::container::layout;

  if constexpr (container_1d<C>) {
    if constexpr (maj == row_major) {
      return v[static_cast<std::make_unsigned_t<T>>(dim * i_ + j_)];
    } else {
      return v[static_cast<std::make_unsigned_t<T>>(n * j_ + i_)];
    }
  }

  else if constexpr (container_2d<C>) {
    if constexpr (maj == row_major) {
      return v[static_cast<std::make_unsigned_t<T>>(i_)]
              [static_cast<std::make_unsigned_t<T>>(j_)];
    } else {
      return v[static_cast<std::make_unsigned_t<T>>(j_)]
              [static_cast<std::make_unsigned_t<T>>(i_)];
    }
  }

}

template <typename T, typename C>
requires container<C> && std::is_integral_v<T>
constexpr auto&
id(C& v, const T n, const T i_) {
  (void) n;
  return v[static_cast<std::make_unsigned_t<T>>(i_)];
}

template <typename T, T dim, kdtree::container::layout maj, typename C>
requires container<C> && std::is_integral_v<T>
constexpr void
swap(C& v, const T n, const T i_, const T j_) {

  auto lswap = [&](T ax) {
    const auto tmp = id<T, dim, maj>(v, n, i_, ax);
    id<T, dim, maj>(v, n, i_, ax) = id<T, dim, maj>(v, n, j_, ax);
    id<T, dim, maj>(v, n, j_, ax) = tmp;
  };

  if constexpr (dim > 8) {
    for (T d = 0; d < dim; ++d) lswap(d);
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

} // namespace container
} // namespace kdtree

#endif // KDTREE_INTERNAL_CONTAINER_HPP
