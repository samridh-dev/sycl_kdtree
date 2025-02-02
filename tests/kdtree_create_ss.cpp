/*
 * Filename: kdtree_create_ss.cpp
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
using kdtree::internal::create::ss;
using kdtree::internal::min;

namespace reference {
template <typename Ts>
constexpr Ts 
ss(Ts s, Ts n) {
  Ts ss_s = 0;
  Ts w  = 1;
  while (s < n) {
    Ts beg = s;
    ss_s += min(w, Ts(n - beg));
    s = l_child(s);
    w += w;
  }
  return ss_s;
}
} // namespace reference


template <typename T>
static void 
check(T vmax, bool do_random = false) {
  
  if (!do_random) {

    for (T s = 0; s < vmax; s++) {
      for (T n = static_cast<T>(s + 1); n <= vmax; n++) {
        CHECK_EQ(ss(s, n), reference::ss(s, n));
      }
    }

  } 

  else {

    std::mt19937_64 rng(12345ULL);
    std::uniform_int_distribution<T> dist(0, vmax);

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
      CHECK_EQ(ss(s, n), reference::ss(s, n));
    }

  }

}

#define TEST_SS_TYPE(T, BOUND, RAND) \
  TEST_CASE("ss<" #T "> tests") {    \
    check<T>(BOUND, RAND);           \
  }

TEST_SS_TYPE(std::uint16_t,  200,  false)
TEST_SS_TYPE(std::uint32_t,  2000,  true)
TEST_SS_TYPE(std::uint64_t,  20000, true)

TEST_CASE("ss manual edge cases") {

  CHECK_EQ(ss(std::uint16_t(0), std::uint16_t(300)), 
       reference::ss(std::uint16_t(0), std::uint16_t(300)));

  CHECK_EQ(ss(std::uint16_t(40000), std::uint16_t(60000)),
       reference::ss(std::uint16_t(40000), std::uint16_t(60000)));

  CHECK_EQ(ss(std::uint32_t(1'000'000'000),
        std::uint32_t(1'000'000'010)),
       reference::ss(std::uint32_t(1'000'000'000), 
               std::uint32_t(1'000'000'010)));
  
}

TEST_CASE("ss boundary and corner cases") {
  using namespace std;

  CHECK_EQ(ss<uint16_t>(0, 1), reference::ss<uint16_t>(0, 1));
  CHECK_EQ(ss<uint16_t>(1, 2), reference::ss<uint16_t>(1, 2));
  CHECK_EQ(ss<uint16_t>(2, 3), reference::ss<uint16_t>(2, 3));

  CHECK_EQ(ss<uint16_t>(7, 8),   reference::ss<uint16_t>(7, 8));
  CHECK_EQ(ss<uint16_t>(8, 9),   reference::ss<uint16_t>(8, 9));
  CHECK_EQ(ss<uint16_t>(15, 16), reference::ss<uint16_t>(15, 16));
  CHECK_EQ(ss<uint16_t>(16, 17), reference::ss<uint16_t>(16, 17));

  CHECK_EQ(ss<uint16_t>(65530, 65535), reference::ss<uint16_t>(65530, 65535));
  CHECK_EQ(ss<uint16_t>(65534, 65535), reference::ss<uint16_t>(65534, 65535));

  CHECK_EQ(ss<uint32_t>(127, 128),   reference::ss<uint32_t>(127, 128));
  CHECK_EQ(ss<uint32_t>(128, 129),   reference::ss<uint32_t>(128, 129));
  CHECK_EQ(ss<uint32_t>(255, 256),   reference::ss<uint32_t>(255, 256));
  CHECK_EQ(ss<uint32_t>(256, 257),   reference::ss<uint32_t>(256, 257));

  CHECK_EQ(ss<uint32_t>(4294967290UL, 4294967295UL),
           reference::ss<uint32_t>(4294967290UL, 4294967295UL));
  CHECK_EQ(ss<uint32_t>(4294967294UL, 4294967295UL),
           reference::ss<uint32_t>(4294967294UL, 4294967295UL));

  CHECK_EQ(ss<uint64_t>(1023,  1024), reference::ss<uint64_t>(1023,  1024));
  CHECK_EQ(ss<uint64_t>(1024,  1025), reference::ss<uint64_t>(1024,  1025));
  CHECK_EQ(ss<uint64_t>(2047,  2048), reference::ss<uint64_t>(2047,  2048));
  CHECK_EQ(ss<uint64_t>(2048,  2049), reference::ss<uint64_t>(2048,  2049));

  static constexpr uint64_t U64MAX = std::numeric_limits<uint64_t>::max();
  CHECK_EQ(ss<uint64_t>(U64MAX - 5, U64MAX),
           reference::ss<uint64_t>(U64MAX - 5, U64MAX));
  CHECK_EQ(ss<uint64_t>(U64MAX - 1, U64MAX),
           reference::ss<uint64_t>(U64MAX - 1, U64MAX));
}


TEST_CASE("ss additional randomized stress tests") {
  using namespace std;

  std::mt19937_64 rng2(98765ULL);

  // 1) 32-bit stress
  {
    std::uniform_int_distribution<uint32_t> dist(0, std::numeric_limits<uint32_t>::max());
    for(int i = 0; i < 2000; i++) {
      uint32_t s = dist(rng2);
      uint32_t n = dist(rng2);
      if (s >= n) {
        std::swap(s, n);
        if (s == n && n < std::numeric_limits<uint32_t>::max()) {
          n++;
        }
      }
      CHECK_EQ(ss(s, n), reference::ss(s, n));
    }
  }

  // 2) 64-bit stress
  {
    std::uniform_int_distribution<uint64_t> dist64(0, std::numeric_limits<uint64_t>::max());
    for(int i = 0; i < 2000; i++) {
      uint64_t s = dist64(rng2);
      uint64_t n = dist64(rng2);
      if (s >= n) {
        std::swap(s, n);
        if (s == n && n < std::numeric_limits<uint64_t>::max()) {
          n++;
        }
      }
      CHECK_EQ(ss(s, n), reference::ss(s, n));
    }
  }

}

