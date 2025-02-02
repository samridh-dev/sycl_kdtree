/*
 * Filename: kdtree_create_sb.cpp
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

#include <kdtree.hpp>

using kdtree::internal::create::l_child;
using kdtree::internal::create::sb;
using kdtree::internal::create::F;
using kdtree::internal::min;
using kdtree::internal::clz;

namespace reference {

template <typename Ts>
constexpr Ts ss(Ts s, Ts n) {
  Ts ss_s = 0;
  Ts w      = 1;
  while (s < n) {
    Ts beg = s;
    ss_s += (w < (n - beg)) ? w : (n - beg);
    s = l_child(s);
    w += w;
  }
  return ss_s;
}

template <typename Ts>
constexpr Ts sb(Ts s, Ts n) {

  const Ts l = (CHAR_BIT * sizeof(Ts) - 1) - clz(static_cast<Ts>(s+1));

  Ts sb_s_l = F(l);
  for (Ts i = sb_s_l; i < s; ++i) sb_s_l += ss(i, n);

  return sb_s_l;

}

} // namespace reference

template <typename T>
static void check_sb(T maxVal, bool doRandom = false) {
  if (!doRandom) {
    for (T s = 1; s < maxVal; s++) {
      for (T n = s + 1; n <= maxVal; n++) {
        CHECK_EQ(sb(s, n), reference::sb(s, n));
      }
    }
  } else {

    std::mt19937_64 rng(12345ULL);
    std::uniform_int_distribution<T> dist(0, maxVal);

    for (int i = 0; i < 1000; i++) {

      T s = dist(rng);
      T n = dist(rng);

      if (s >= n) {
        std::swap(s, n);
        if (s == n) {
          if (n < std::numeric_limits<T>::max()) {
            n++;
          } else if (s > 0) {
            s--;
          }
        }
      }

      CHECK_EQ(sb(s, n), reference::sb(s, n));

    }

  }

}

#define TEST_SB_TYPE(T, BOUND, RAND) \
  TEST_CASE("sb<" #T "> tests") {    \
    check_sb<T>(BOUND, RAND);        \
  }

TEST_SB_TYPE(std::uint16_t,  200,   false)
TEST_SB_TYPE(std::uint32_t,  2000,  true)
TEST_SB_TYPE(std::uint64_t,  20000, true)

TEST_CASE("sb boundary and corner cases") {
  using namespace std;

  // 1) Very small edge cases
  CHECK_EQ(sb<uint16_t>(1, 2), reference::sb<uint16_t>(1, 2));
  CHECK_EQ(sb<uint16_t>(1, 3), reference::sb<uint16_t>(1, 3));
  CHECK_EQ(sb<uint16_t>(2, 3), reference::sb<uint16_t>(2, 3));

  // 2) Around small powers of two (for 16-bit)
  CHECK_EQ(sb<uint16_t>(3, 4),  reference::sb<uint16_t>(3, 4));
  CHECK_EQ(sb<uint16_t>(4, 5),  reference::sb<uint16_t>(4, 5));
  CHECK_EQ(sb<uint16_t>(7, 8),  reference::sb<uint16_t>(7, 8));
  CHECK_EQ(sb<uint16_t>(8, 9),  reference::sb<uint16_t>(8, 9));

  // 3) Near the maximum for 16-bit
  //  e.g. 65534 = 0xFFFE, 65535 = 0xFFFF
  CHECK_EQ(sb<uint16_t>(65530, 65535), reference::sb<uint16_t>(65530, 65535));
  CHECK_EQ(sb<uint16_t>(65534, 65535), reference::sb<uint16_t>(65534, 65535));

  // 4) 32-bit powers of two
  CHECK_EQ(sb<uint32_t>(15, 16),   reference::sb<uint32_t>(15, 16));
  CHECK_EQ(sb<uint32_t>(16, 17),   reference::sb<uint32_t>(16, 17));
  CHECK_EQ(sb<uint32_t>(31, 32),   reference::sb<uint32_t>(31, 32));
  CHECK_EQ(sb<uint32_t>(32, 33),   reference::sb<uint32_t>(32, 33));

#if 0 // takes too long
  // 5) Near the maximum for 32-bit
  //  e.g. 4294967295 = 0xFFFFFFFF
  CHECK_EQ(sb<uint32_t>(4294967290UL, 4294967295UL),
       reference::sb<uint32_t>(4294967290UL, 4294967295UL));
  CHECK_EQ(sb<uint32_t>(4294967294UL, 4294967295UL),
       reference::sb<uint32_t>(4294967294UL, 4294967295UL));
#endif

  // 6) 64-bit powers of two
  CHECK_EQ(sb<uint64_t>(63,   64),  reference::sb<uint64_t>(63, 64));
  CHECK_EQ(sb<uint64_t>(64,   65),  reference::sb<uint64_t>(64, 65));
  CHECK_EQ(sb<uint64_t>(127,  128), reference::sb<uint64_t>(127, 128));
  CHECK_EQ(sb<uint64_t>(128,  129), reference::sb<uint64_t>(128, 129));

#if 0 // takes too long
  // 7) Near the maximum for 64-bit
  static constexpr uint64_t U64MAX = std::numeric_limits<uint64_t>::max();
  CHECK_EQ(sb<uint64_t>(U64MAX - 5, U64MAX),
       reference::sb<uint64_t>(U64MAX - 5, U64MAX));
  CHECK_EQ(sb<uint64_t>(U64MAX - 1, U64MAX),
       reference::sb<uint64_t>(U64MAX - 1, U64MAX));
#endif
}
