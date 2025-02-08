/*!
 * \file        internal/clz.hpp
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

#ifndef KDTREE_INTERNAL_CLZ
#define KDTREE_INTERNAL_CLZ

#include "../pch.hpp"

namespace kdtree   {
namespace internal {

template <typename T>
requires std::is_integral_v<T>
constexpr inline T
clz(const T n);

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

template <typename T>
requires std::is_integral_v<T>
constexpr inline T
kdtree::internal::clz(const T n) {

#ifdef KD__USING_SYCL

  return sycl::clz(n);

#elif defined(__GNUC__) || defined(__clang__)

  constexpr T b{std::numeric_limits<std::make_unsigned_t<T>>::digits};
  if (!n) { 
    return b;
  };

  if constexpr (sizeof(T) <= sizeof(unsigned int)) {
    int cnt = __builtin_clz(static_cast<unsigned int>(n));
    int d   = static_cast<int>(std::numeric_limits<unsigned int>::digits) - b;
    return static_cast<T>(cnt - d);
  }
  else if constexpr (sizeof(T) <= sizeof(unsigned long)) {
    int cnt = __builtin_clzl(static_cast<unsigned long>(n));
    int d   = static_cast<int>(std::numeric_limits<unsigned long>::digits) - b;
    return static_cast<T>(cnt - d);
  }
  else {
    int cnt = __builtin_clzll(static_cast<unsigned long long>(n));
    int d   = static_cast<int>(std::numeric_limits<unsigned long long>::digits) 
            - b;
    return static_cast<T>(cnt - d);
  }

#else

#error "kdtree::internal::clz undefined"

#endif

}

#endif // KDTREE_INTERNAL_CLZ
