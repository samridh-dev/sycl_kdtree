/*!
 * \file        create/tag.hpp
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

#ifndef KDTREE_CREATE_TAG_HPP
#define KDTREE_CREATE_TAG_HPP

namespace kdtree   {
namespace internal {
namespace create   {
namespace tags     {

template <typename C, typename T>
requires kdtree::container::container<C> && std::is_integral_v<T>
void
update(const kdtree::context& ctx, C& arr, const T n, const T l); 

} // namespace tag
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

#include "internal/ss.hpp"
#include "internal/sb.hpp"
#include "internal/F.hpp"
#include "../internal/lrchild.hpp"
#include "../internal/bsr.hpp"

template <typename C, typename T>
requires kdtree::container::container<C> && std::is_integral_v<T>
void
kdtree::internal::create::tags::update(const kdtree::context& ctx,
                                       C& arr, const T n, const T l) {

  using kdtree::internal::create::F;
  using kdtree::internal::l_child;
  using kdtree::internal::r_child;
  using kdtree::internal::bsr;

  (void) ctx;

  (void) l;
  
  const T L{bsr(n)+1};

  for (T i{F(l)}; i < n; ++i) {

    const T c{arr[i]};
    const T p{sb(c, n, L) + ss(l_child(c), n, L)};

#if 0
    if (i < p) {
      arr[i] = l_child(c);
    } else if (i > p) {
      arr[i] = r_child(c);
    } else {
    }
#else
    if (i < p) {
      kdtree::container::id<T>(arr, n, i) = l_child(c);
    } else if (i > p) {
      kdtree::container::id<T>(arr, n, i) = r_child(c);
    } else {
    }
#endif

  }

}

#endif // KDTREE_CREATE_TAG_HPP
