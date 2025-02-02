/*!
 * \file        traverse.hpp
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

#ifndef KDTREE_TRAVERSE_HPP
#define KDTREE_TRAVERSE_HPP

///////////////////////////////////////////////////////////////////////////////
/// HEADER                                                                  ///
///////////////////////////////////////////////////////////////////////////////

namespace kdtree {

template<typename result_t,
         typename process_node,
         typename Ts, typename Tf, Ts dim, kdtree::layout maj, 
         typename Cq, typename Cs>
void
traverse(result_t& result, const Cq& q, const Cs& src, const Ts n, 
         Tf rmax = std::numeric_limits<Tf>::max());

} // namespace kdtree

///////////////////////////////////////////////////////////////////////////////
/// IMPLEMENTATION                                                          ///
///////////////////////////////////////////////////////////////////////////////

#include "container.hpp"
#include "layout.hpp"
#include "internal.hpp"

namespace kdtree {
namespace internal {
namespace traverse {

template<typename Ts, Ts dim, kdtree::layout maj, typename Cs>
Ts 
split_dim_of(const Cs& s, Ts i) {
  (void) s;
  Ts l = 0;
  i = i + 1; 
  while (i > 1) {
    i >>= 1;
    ++l;
  }
  return l % dim;
}

template<typename process_node,
         typename result_t,
         typename Ts, typename Tf, Ts dim, kdtree::layout maj, 
         typename Cq, typename Cs>
struct is_valid_signature {
  static constexpr bool value = std::is_invocable_r_v<
    void, 
    process_node,
    result_t&, 
    const Cq&, 
    const Cs&,
    Ts, 
    Ts, 
    Tf&
  >;
};

} // namespace traverse
} // namespace internal
} // namespace kdtree

template<typename result_t,
         typename process_node,
         typename Ts, typename Tf, Ts dim, kdtree::layout maj, 
         typename Cq, typename Cs>
void
kdtree::traverse(result_t& result, const Cq& q, const Cs& src, const Ts n, 
                 Tf rmax) {

  using kdtree::internal::traverse::is_valid_signature;

  static_assert(container::is_1d_v<Cq>, 
                "`Cq` is not a 1 dimensional container"); 

  static_assert(container::is_1d_v<Cq> || container::is_2d_v<Cs>, 
                "`Cs` is not a 1 or 2 dimensional container"); 

  static_assert(std::is_signed<Ts>::value, 
                "`Ts` must be a signed integer");

  static_assert(std::is_same<container::get_primitive_type<Cq>,
                             container::get_primitive_type<Cs>>::value,
                "`Cq` and `Cs` does not share same primitive type"); 

  static_assert(
    is_valid_signature<
      process_node,
      result_t,
      Ts,
      Tf,
      dim,
      maj,
      Cq,
      Cs
    >::value,
    "oops"
  );

  using kdtree::internal::id;
  using kdtree::internal::abs;
  using kdtree::internal::traverse::split_dim_of;

  Ts curr = Ts( 0);
  Ts prev = Ts(-1);

  while(1) {

    const Ts parent = (curr+1) / Ts(2) - Ts(1);

    if (curr >= n) {
      prev = curr;
      curr = parent;
      continue;
    }

    const bool from_parent = (prev < curr);

    if (from_parent) {
      process_node{}(result, q, src, n, curr, rmax);
    }

    Ts   s_dim        = split_dim_of<Ts, dim, maj, Cs>(src, curr);
    Tf   s_pos        = id<Ts, dim, maj>(src, n, curr, s_dim);
    Tf   sign_dist    = static_cast<Tf>(q[s_dim]) - static_cast<Tf>(s_pos);
    Ts   close_side   = sign_dist > Tf(0);
    Ts   close_child  = 2 * curr + 1 + close_side;
    Ts   far_child    = 2 * curr + 2 - close_side;
    bool far_in_range = abs(sign_dist) <= rmax;

    Ts next;
    if (from_parent) {
      next = close_child;
    } else if (prev == close_child) {
      next = far_in_range ? far_child : parent;
    } else {
      next = parent;
    }

    if (next == -1) {
      return;
    }

    prev = curr;
    curr = next;

  }

}

#endif // KDTREE_TRAVERSE_HPP
