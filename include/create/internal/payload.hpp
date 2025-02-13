/*!
 * \file        create/internal/payload.hpp
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

#ifndef KDTREE_CREATE_INTERNAL_PAYLOAD_HPP
#define KDTREE_CREATE_INTERNAL_PAYLOAD_HPP

#include "../../container.hpp"

namespace kdtree   {
namespace internal {
namespace create   {

template <typename T, T dim, kdtree::container::layout maj,
          typename C_src, typename C_tag>
requires kdtree::container::container<C_src>
      && kdtree::container::container<C_tag>
struct payload;

} // namespace create
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

template <typename T, T dim, kdtree::container::layout maj,
          typename C_src, typename C_tag>
requires kdtree::container::container<C_src>
      && kdtree::container::container<C_tag>
struct kdtree::internal::create::payload {

  C_src& src;
  C_tag& tag;
  const T n;
  const T d;

  explicit payload(C_src& src_, C_tag& tag_, const T n_, const T d_)
    : src(src_), tag(tag_), n(n_), d(d_) {}

  template <typename int_t>
  requires std::is_integral_v<int_t>
  inline void
  swap(const int_t i_, const int_t j_) {
    kdtree::container::swap<int_t, dim,  maj>(src, n, i_, j_);
    kdtree::container::swap<int_t, T{1}, maj>(tag, n, i_, j_);
  }

  template <typename int_t>
  requires std::is_integral_v<int_t>
  inline bool
  less(const int_t i_, const int_t j_) {

    return (
             kdtree::container::id<int_t>(tag, n, i_)
             <
             kdtree::container::id<int_t>(tag, n, j_)
           ) || (
             (
               kdtree::container::id<int_t>(tag, n, i_)
               ==
               kdtree::container::id<int_t>(tag, n, j_)
             ) && (
               kdtree::container::id<int_t, dim, maj>(src, n, i_, d) 
               <
               kdtree::container::id<int_t, dim, maj>(src, n, j_, d)
             )
           );

  }

};

#endif // KDTREE_CREATE_INTERNAL_SS_HPP
