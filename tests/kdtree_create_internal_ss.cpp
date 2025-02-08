/*
 * Filename: tests/kdtree_create_internal_ss.cpp
 * Author:   Samridh D. Singh
 * Date:     2025-02-01
 *
 * This file is part of sycl_kdtree.
 *
 * sycl_kdtree is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * sycl_kdtree is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with sycl_kdtree. If not, see <https://www.gnu.org/licenses/>.
 */

#include "pch.h"

#include <create/internal/ss.hpp>
#include <internal/bsr.hpp>

using kdtree::internal::create::ss;

template <typename T>
requires std::is_integral_v<T>
constexpr inline T
ss(const T s, const T n) {
  using kdtree::internal::bsr;
  const T L{bsr(n)+1};
  return ss(s, n, L);
}

namespace ref {
template <typename T>
requires std::is_integral_v<T>
constexpr inline T
ss(const T s, const T n) {
  if (s >= n) return T{0};
  return T{1} + ss(T{2} * s + T{1}, n) + ss(T{2} * s + T{2}, n);
}

} // namespace ref

template <typename T>
static void 
check(void) {

  std::mt19937_64 rng(12345ULL);
  std::uniform_int_distribution<T> dist(0, 127);

  const T imax{std::numeric_limits<T>::max() < 1 << 16 ?  
               std::numeric_limits<T>::max() : 1 << 16 };

  for (T i{0}; i < imax; i++) {
    T s{dist(rng)};
    T n{dist(rng)};
    if (s >= n) {
      std::swap(s, n);
    }
    if (s == n) {
      if (n < std::numeric_limits<T>::max()) n++;
      else if (s > 0) s--;
    }
    CAPTURE(imax);
    CAPTURE(s);
    CAPTURE(n);
    CHECK_EQ(ss(s, n), ref::ss(s, n));
  }

}

TEST_CASE("[case=1][std::uint32_t] random tests ") { check<std::uint32_t>(); }
TEST_CASE("[case=2][std::uint64_t] random tests ") { check<std::uint64_t>(); }

TEST_CASE("[case=3]") {

  using T = std::int32_t;

  for (T k{1}; k <= T{sizeof(T) - 2}; ++k) {

    T n{(T{1} << k) - T{1}};

    CAPTURE(n);
    CAPTURE(k);

    CHECK_EQ(ss<T>(T{0}, n), n);

  }

}

TEST_CASE("[case=4]") {

  using T = std::uint64_t;

  CHECK_EQ(ss<T>(T{0}, T{0}), T{0});
  CHECK_EQ(ss<T>(T{0}, T{1}), T{1});
  CHECK_EQ(ss<T>(T{1}, T{1}), T{0});
  CHECK_EQ(ss<T>(T{1}, T{0}), T{0});

}
