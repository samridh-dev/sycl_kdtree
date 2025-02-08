/*!
 * \file        sort/bitonic.hpp
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

#ifndef KDTREE_SORT_BITONIC_HPP
#define KDTREE_SORT_BITONIC_HPP

#include "../pch.hpp"
#include "../container.hpp"
#include "internal.hpp"

namespace kdtree {
namespace bitonic {

  template <typename payload_t, typename T>
  requires kdtree::internal::sort::payload<payload_t> && std::integral<T>
  void 
  sort(const kdtree::context& ctx, payload_t& p, const T n0, const T n1);

} // namespace bitonic
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

#include <future>
#include <cmath>
#include <functional>
#include <stdexcept>

template <typename payload_t, typename T>
requires kdtree::internal::sort::payload<payload_t> && std::integral<T>
void 
kdtree::bitonic::sort(const kdtree::context& ctx, payload_t& p,
                           const T n0, const T n1) {

  if (n1 < n0) {
    throw std::out_of_range("n1 must be greater than n0.");
  } if (n1 == n0) {
    return;
  }

  T n{n1 - n0};

  std::size_t max_d { ctx.nthreads > 1
                        ? static_cast<std::size_t>(std::log2(ctx.nthreads))
                        : 0 };

  std::function<void(T, T, bool, std::size_t)> bmerge =
    [&](T lo, T hi, bool dir, std::size_t d) {
      auto pow2_le_n{[&](T i) -> T {
        T k = 1;
        while (k > 0 && k < i)
          k <<= 1;
        return k >> 1;
      }};
      if (hi > 1) {
        T m{pow2_le_n(hi)};
        for (T i{lo}; i < lo + hi - m; ++i) {
          if (dir == p.less(i + m, i))
            p.swap(i + m, i);
        }
        if (d < max_d) {
          auto fut{std::async(std::launch::async, bmerge, lo, m, dir, d + 1)};
          bmerge(lo + m, hi - m, dir, d + 1);
          fut.get();
        } else {
          bmerge(lo, m, dir, d + 1);
          bmerge(lo + m, hi - m, dir, d + 1);
        }
      }
    };

  std::function<void(T, T, bool, std::size_t)> bsort {
    [&](T lo, T hi, bool dir, std::size_t d) {
      if (hi > 1) {
        T m{hi / 2};
        if (d < max_d) {
          auto fut{std::async(std::launch::async, bsort, lo, m, !dir, d + 1)};
          bsort(lo + m, hi - m, dir, d + 1);
          fut.get();
        } else {
          bsort(lo, m, !dir, d + 1);
          bsort(lo + m, hi - m, dir, d + 1);
        }
        bmerge(lo, hi, dir, d);
      }
    }};

  bsort(n0, n, true, 0);

}

#endif // KDTREE_SORT_BITONIC_HPP
