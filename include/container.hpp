/*!
 * \file        container.hpp
 * \author      Samridh D. Singh
 * \date        2025-02-01
 * \brief       Type traits to enforce template parameter types.
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

#ifndef KDTREE_CONTAINER_HPP
#define KDTREE_CONTAINER_HPP

#include <type_traits>
#include <utility>

namespace kdtree {

namespace container {

template <class T, class = void>
struct is_1d : std::false_type {};

template <class T, class = void>
struct is_2d : std::false_type {};

template <class T>
struct is_1d<T, std::void_t<
  typename T::value_type,
  decltype(std::declval<T&>()[std::declval<std::size_t>()])
>> : std::bool_constant< std::is_fundamental_v<typename T::value_type> > {};

template <class T>
struct is_2d<T, std::void_t<
  typename T::value_type,
  decltype(std::declval<T&>()[std::declval<std::size_t>()]),
  decltype(std::declval<T&>()[std::declval<std::size_t>()]
                             [std::declval<std::size_t>()])
>> : std::bool_constant< is_1d<typename T::value_type>::value > {};

template <typename T>
struct is_1d<T*> : std::true_type {};

template <typename T>
struct is_2d<T**> : std::true_type {};

template <typename T>
constexpr bool is_1d_v = is_1d<T>::value;

template <typename T>
constexpr bool is_2d_v = is_2d<T>::value;


template <typename T, typename enable = void>
struct get_primitive_type {
    using type = void;
};

template <typename T>
struct get_primitive_type<T, std::enable_if_t<is_1d_v<T>>> {
    using type = typename T::value_type;
};

template <typename T>
struct get_primitive_type<T, std::enable_if_t<is_2d_v<T>>> {
    using type = typename get_primitive_type<typename T::value_type>::type;
};

template <typename T>
struct get_primitive_type<T*> {
    using type = typename get_primitive_type<T>::type;
};

template <typename T>
struct get_primitive_type<T**> {
    using type = typename get_primitive_type<T>::type;
};

template <typename T>
using get_primitive_type_t = typename get_primitive_type<T>::type;


} // namespace container

} // namespace kdtree
  
#endif // KDTREE_CONTAINER_HPP
