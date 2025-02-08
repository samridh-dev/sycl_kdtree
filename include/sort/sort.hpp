/*!
 * \file        sort/sort.hpp
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

#ifndef KDTREE_SORT_HPP
#define KDTREE_SORT_HPP

#include "../pch.hpp"
#include "../container.hpp"
#include "internal.hpp"

namespace kdtree {

  template <typename payload_t, typename T>
  requires kdtree::internal::sort::payload<payload_t> && std::integral<T>
  void 
  sort(const kdtree::context& ctx, payload_t& p, const T n0, const T n1);

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

#include "bitonic.hpp"
#include "odd_even.hpp"

template <typename payload_t, typename T>
requires kdtree::internal::sort::payload<payload_t> && std::integral<T>
void 
kdtree::sort(const kdtree::context& ctx, payload_t& p, 
             const T n0, const T n1) {

  if (n1 < n0) {
    throw std::out_of_range("`n1` must be greater than `n0`.");
  } else if (n1 == n0) {
    return;
  } 

  (void) ctx;

  const T n{n1 - n0}; 

  if ((n & (n - 1)) == 0) { 
    kdtree::bitonic::sort(ctx, p, n0, n1);
  } 
  else {

    kdtree::bitonic::sort(ctx, p, n0, n1);

  }

}

#endif // KDTREE_SORT_HPP
