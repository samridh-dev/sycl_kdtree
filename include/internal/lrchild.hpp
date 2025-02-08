/*!
 * \file        internal/lrchild.hpp
 * \author      Samridh D. Singh
 * \date        2025-02-01
 * \brief       
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

#ifndef KDTREE_INTERNAL_LRCHILD_HPP
#define KDTREE_INTERNAL_LRCHILD_HPP

#include "../pch.hpp"
#include "../container.hpp"

namespace kdtree   {
namespace internal {

template <typename T>
requires std::is_integral_v<T>
constexpr inline T
l_child(const T n);


template <typename T>
requires std::is_integral_v<T>
constexpr inline T
r_child(const T n);

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
kdtree::internal::l_child(const T n) {
  return T{2} * n + T{1};
}


template <typename T>
requires std::is_integral_v<T>
constexpr inline T
kdtree::internal::r_child(const T n) {
  return T{2} * n + T{2};
}

#endif // KDTREE_INTERNAL_LRCHILD_HPP
