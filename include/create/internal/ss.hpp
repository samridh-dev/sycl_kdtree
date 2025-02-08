/*!
 * \file        create/internal/ss.hpp
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

#ifndef KDTREE_CREATE_INTERNAL_SS_HPP
#define KDTREE_CREATE_INTERNAL_SS_HPP

namespace kdtree   {
namespace internal {
namespace create   {

template <typename T>
requires std::is_integral_v<T>
constexpr inline T
ss(const T s, const T n, const T L);

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

#include "../../internal/bsr.hpp"
#include "../../internal/lrchild.hpp"

template <typename T>
requires std::is_integral_v<T>
constexpr inline T
kdtree::internal::create::ss(const T s, const T n, const T L) {

  using std::min;
  using std::max;

#if 1
  if (s >= n) return 0;
  T l         { bsr(s+1) };
  T fllc_s    { ~((~s) << (L - l - T{1})) };
  T nn_fllc_s { min((n > fllc_s) ? n - fllc_s : T{0}, T{1} << (L - l - T{1})) };
  T ss_s      { (T{1} << (L - l - T{1})) - T{1} + nn_fllc_s };
  return ss_s;
#else

  (void) L;
  if (s >= n) {
    return 0;
  }
  return T{1} + ss(l_child(s), n, L) + ss(r_child(s), n, L);

#endif

}

#endif // KDTREE_CREATE_INTERNAL_SS_HPP
