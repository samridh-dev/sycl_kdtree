/*!
 * \file        create/internal/sb.hpp
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

#ifndef KDTREE_CREATE_INTERNAL_SB_HPP
#define KDTREE_CREATE_INTERNAL_SB_HPP

namespace kdtree   {
namespace internal {
namespace create   {

template <typename T>
requires std::is_integral_v<T>
constexpr inline T
sb(const T s, const T n, const T L);

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

#include "F.hpp"
#include "ss.hpp"
#include "../../internal/bsr.hpp"
#include "../../internal/lrchild.hpp"
#include "../../internal/minmax.hpp"

template <typename T>
requires std::is_integral_v<T>
constexpr inline T
kdtree::internal::create::sb(const T s, const T n, const T L) {

  using kdtree::internal::bsr;
  using kdtree::internal::min;

#if 1

  const T l       { bsr(s+1) };;
  const T nls_s   { s - ((T{1} << l) - T{1}) };
  const T sb_s_l  { (T{1} << l) - T{1}
                  + nls_s * ((T{1} << (L - l - T{1})) - T{1})
                  + min(
                      nls_s * (T{1} << (L - l - T{1})),
                      n - ((T{1} << (L - T{1})) - T{1})
                    )
                  };
  return sb_s_l;

#else

  const T l{bsr(s+1)};
  T sb_s_l{F(l)};
  for (T i{F(l)}; i < s; ++i) sb_s_l += ss(i, n, L);
  return sb_s_l;

#endif

}

#endif // KDTREE_CREATE_INTERNAL_SB_HPP
