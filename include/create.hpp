/*!
 * \file        create.hpp
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

#ifndef KDTREE_CREATE_HPP
#define KDTREE_CREATE_HPP

///////////////////////////////////////////////////////////////////////////////
/// HEADER                                                                  ///
///////////////////////////////////////////////////////////////////////////////

#include "layout.hpp"

namespace kdtree {

template <typename Ts, Ts dim, layout maj = layout::rowmajor, 
          typename C, typename Tn>
void 
create(C& src, const Tn n);

} // namespace kdtree

///////////////////////////////////////////////////////////////////////////////
/// IMPLEMENTATION                                                          ///
///////////////////////////////////////////////////////////////////////////////

#define USE_PRINT 0

#include "internal.hpp"

#include <climits>
#include <type_traits>
#include <vector>
#include <thread>
#include <utility>
#include <algorithm>

namespace kdtree {

namespace internal {

namespace create {

template <typename Ts> 
constexpr Ts 
F(Ts l) { 
  return (Ts(1) << l) - Ts(1); 
}
template <typename Ts> 
constexpr Ts 
l_child(Ts i) {
  return Ts(2) * i + Ts(1); 
}

template <typename Ts> 
constexpr Ts 
r_child(Ts i) {
  return Ts(2) * i + Ts(2); 
}

template <typename Ts>
constexpr Ts 
ss(Ts s, Ts n) {
#if 0
  const Ts L = (CHAR_BIT * sizeof(Ts) - 0) - clz(n);
  const Ts l = (CHAR_BIT * sizeof(Ts) - 1) - clz(static_cast<Ts>(s+1));

  const Ts fllc_s = ~((~s) << (L - l - Ts(1)));

  const Ts ss_s = (Ts(1) << (L - l - Ts(1))) - Ts(1)
                + min(
                    max(Ts(0), static_cast<Ts>(n - fllc_s)), 
                    static_cast<Ts>(Ts(1) << (L - l - Ts(1)))
                  );

  return ss_s;
#else
  Ts ss_s = 0;
  Ts w = 1;
  while (s < n) {
    const Ts beg = s;
    ss_s += min(w, static_cast<Ts>(n - beg));
    s = l_child(s);
    w += w;
  }
  return ss_s;
#endif
}

template <typename Ts>
constexpr Ts sb(Ts s, Ts n) {

#if 0
  
  const Ts L = (CHAR_BIT * sizeof(Ts) - 0) - clz(n);
  const Ts l = (CHAR_BIT * sizeof(Ts) - 1) - clz(static_cast<Ts>(s+1));

  const Ts nls_s  = s - ((Ts(1) << l) - 1);

  const Ts sb_s_l = (Ts(1) << l) - Ts(1)
                  + nls_s * ((Ts(1) << (L - l - Ts(1))) - Ts(1))
                  + min(
                      static_cast<Ts>(nls_s * (Ts(1) << (L - l - Ts(1)))),
                      static_cast<Ts>(n - ((Ts(1) << (L - Ts(1))) - Ts(1)))
                   );

#else

  const Ts l = (CHAR_BIT * sizeof(Ts) - 1) - clz(static_cast<Ts>(s+1));

  Ts sb_s_l = F(l);
  for (Ts i = sb_s_l; i < s; ++i) sb_s_l += ss(i, n);

#endif

  return sb_s_l;
  
}

template <typename Ts, Ts dim, kdtree::layout maj, typename Cs, typename Cv>
void
sort(Cs& tag, Cv& src, const Ts n, const Ts l) {

  static_assert(std::is_same_v<typename Cs::value_type, Ts>, 
                "`Cs` does not have underlying type `Ts`.");

  using kdtree::internal::swap;
  using kdtree::internal::id;

  const Ts d = l % dim;

  const auto is_pow2 = [](const Ts n_) { return (n_ & (n_-1)) == 0; }; 
  const auto less = [&](const Ts i_, const Ts j_) {
    return (tag[i_] < tag[j_]) ||
           (
             (tag[i_] == tag[j_]) &&
             (id<Ts, dim, maj>(src, n, i_, d) < 
              id<Ts, dim, maj>(src, n, j_, d))
           );
  };

  if (n == 0) {
    return; // throw error;
  }

  if (is_pow2(n)) {

    // bitonic sort
    for (Ts k = 2; k <= n; k <<= 1) {
    for (Ts j = k >> 1; j > 0; j >>= 1) {
    for (Ts i = 0; i < n; ++i) {
      Ts ixj = i ^ j;
      if (ixj > i) {
        bool ascending = ((i & k) == 0);
        if ((ascending && less(ixj, i)) ||
          (!ascending && less(i, ixj))) {
          swap<Ts, dim,   maj>(src, n, i, ixj);
          swap<Ts, Ts(1), maj>(tag, n, i, ixj);
        }
      }
    }}}

  }

  else {

    // bubble sort (ew)
    for (Ts i = 0; i < n - 1; ++i) {
    for (Ts j = 0; j < n - i - 1; ++j) {
      if (less(j + 1, j)) {
        swap<Ts, dim,   maj>(src, n, j, j + 1);
        swap<Ts, Ts(1), maj>(tag, n, j, j + 1);
      }
    }}

  }

}

template <typename Ts, Ts dim, kdtree::layout maj, typename Cs>
void
update_tags(Cs& tag, const Ts n, const Ts l_) {

  static_assert(std::is_same_v<typename Cs::value_type, Ts>, 
                "`Cs` does not have underlying type `Ts`.");

  using kdtree::internal::create::l_child;
  using kdtree::internal::create::r_child;
  using kdtree::internal::create::ss;
  using kdtree::internal::create::sb;
  using kdtree::internal::create::F;

  auto work = [&](Ts beg, Ts end) {

    for (Ts idx = beg; idx < end; ++idx) {

      const Ts cur = tag[idx];
      const Ts p = sb(cur, n) + ss(l_child(cur), n);
      
      if (idx < p) {
        tag[idx] = l_child(cur);
      } else if (idx > p) {
        tag[idx] = r_child(cur);
      }

    }

  };

  const Ts offs = F(l_);

  const Ts nthreads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads;
  const Ts chunk_size = (n - offs) / nthreads;

  for (Ts i = 0; i < nthreads; ++i) {
    Ts beg = offs + i * chunk_size;
    Ts end = (i == nthreads - 1) ? n : beg + chunk_size;
    threads.emplace_back(work, beg, end);
  }

  for (auto& t : threads) {
    t.join();
  }
}

}  // namespace create
}  // namespace internal
}  // namespace kdtree

template <typename Ts, Ts dim, kdtree::layout maj, typename C, typename Tn>
void 
kdtree::create(C& src, const Tn n) {

  static_assert(std::is_integral_v<Tn>, "`n` must be an integer type");

  using kdtree::internal::clz;

  const Ts N = static_cast<Ts>(n); 

  if (N <= 0) {
    return;
  }

  std::vector<Ts> tag(N, 0);

  for (Ts l = 0; l < Ts(CHAR_BIT * sizeof(Ts)) - clz(N); ++l) {
    internal::create::sort<Ts, dim, maj>(tag, src, N, l);
    internal::create::update_tags<Ts, dim, maj>(tag, N, l);
  }

}

#endif // KDTREE_CREATE_HPP
