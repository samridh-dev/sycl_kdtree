/*
 * Filename: tests/kdtree_create_internal_sb.cpp
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

#include <create/internal/sb.hpp>
#include <create/internal/ss.hpp>
#include <create/internal/F.hpp>
#include <internal/bsr.hpp>

using kdtree::internal::create::ss;
using kdtree::internal::create::sb;
using kdtree::internal::create::F;
using kdtree::internal::bsr;

template <typename T>
requires std::is_integral_v<T>
constexpr inline T
sb(const T s, const T n) {
  const T L{bsr(n)+1};
  return sb(s, n, L);
}

namespace ref {

template <typename T>
requires std::is_integral_v<T>
constexpr inline T 
ss(T s, T n) {
  T ss_s{0};
  T w{1};
  while (s < n) {
    T beg{s};
    ss_s += (w < (n - beg)) ? w : (n - beg);
    s = T{2} * s + T{1};
    w += w;
  }
  return ss_s;
}

template <typename T>
requires std::is_integral_v<T>
constexpr inline T 
sb(const T s, const T n) {
  const T l{bsr(s+1)};
  T sb_s_l{F(l)};
  for (T i{sb_s_l}; i < s; ++i) sb_s_l += ss(i, n);
  return sb_s_l;
}

} // namespace ref

TEST_CASE("[case=1]") {

  using namespace std;

  CHECK_EQ(sb<int32_t>(1, 2), ref::sb<int32_t>(1, 2));
  CHECK_EQ(sb<int32_t>(1, 3), ref::sb<int32_t>(1, 3));
  CHECK_EQ(sb<int32_t>(2, 3), ref::sb<int32_t>(2, 3));

  CHECK_EQ(sb<int32_t>(3, 4), ref::sb<int32_t>(3, 4));
  CHECK_EQ(sb<int32_t>(4, 5), ref::sb<int32_t>(4, 5));
  CHECK_EQ(sb<int32_t>(7, 8), ref::sb<int32_t>(7, 8));
  CHECK_EQ(sb<int32_t>(8, 9), ref::sb<int32_t>(8, 9));

  CHECK_EQ(sb<int32_t>(15, 16), ref::sb<int32_t>(15, 16));
  CHECK_EQ(sb<int32_t>(16, 17), ref::sb<int32_t>(16, 17));
  CHECK_EQ(sb<int32_t>(31, 32), ref::sb<int32_t>(31, 32));
  CHECK_EQ(sb<int32_t>(32, 33), ref::sb<int32_t>(32, 33));

  CHECK_EQ(sb<int64_t>(63,   64),  ref::sb<int64_t>(63, 64));
  CHECK_EQ(sb<int64_t>(64,   65),  ref::sb<int64_t>(64, 65));
  CHECK_EQ(sb<int64_t>(127,  128), ref::sb<int64_t>(127, 128));
  CHECK_EQ(sb<int64_t>(128,  129), ref::sb<int64_t>(128, 129));

}
