/*!
 * \file        sort/odd_even.hpp
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

#ifndef KDTREE_SORT_ODD_EVEN_HPP
#define KDTREE_SORT_ODD_EVEN_HPP

#include "../pch.hpp"
#include "../container.hpp"
#include "internal.hpp"

namespace kdtree {
namespace odd_even {

  template <typename payload_t, typename T>
  requires kdtree::internal::sort::payload<payload_t> && std::integral<T>
  void 
  sort(const kdtree::context& ctx, payload_t& p, const T n0, const T n1);

} // namespace odd_even
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

template <typename payload_t, typename T>
requires kdtree::internal::sort::payload<payload_t> && std::integral<T>
void 
kdtree::odd_even::sort(const kdtree::context& ctx, payload_t& p,
                       const T n0, const T n1) {

  if (n1 < n0) {
    throw std::out_of_range("`n1` must be greater than `n0`.");
  } else if (n1 == n0) {
    return;
  }

  (void) ctx;

  using U = std::make_unsigned_t<T>;

  bool is_sorted{false};

  while (!is_sorted) {

    is_sorted = true;

    for (U i{static_cast<U>(n0 + 1)}; i < static_cast<U>(n1 - 1); i += U(2)) {
      if (p.less(i+1, i)) {
        p.swap(i, i+1);
        is_sorted = false;
      }
    }

    for (U i{static_cast<U>(n0)}; i < static_cast<U>(n1 - 1); i += U(2)) {
      if (p.less(i+1, i)) {
        p.swap(i, i+1);
        is_sorted = false;
      }
    }

  }

}

#endif // KDTREE_SORT_ODD_EVEN_HPP
