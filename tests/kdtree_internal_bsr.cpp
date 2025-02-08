/*!
 * \file        test/kdtree_internal_bsr.cpp
 * \author      Samridh D. Singh
 * \date        2025-02-01
 * \details     
 *
 * \copyright   This file is part of the sycl_kdtree project.
 * \copyright   Copyright (C) 2025, Samridh D. Singh
 * \copyright   
 *              sycl_kdtree is free software: you can redistribute it and/or
 *              modify it under the terms of the GNU General Public License as
 *              published by the Free Software Foundation, either version 3 of
 *              the License or (at your option) any later version.
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

#include "pch.h"

#include <internal/bsr.hpp>

namespace ref {

template <typename T>
requires std::is_integral_v<T>
constexpr inline int bsr(const T n) {
    using UT = std::make_unsigned_t<T>;
    constexpr int b = std::numeric_limits<UT>::digits;
    UT x = static_cast<UT>(n);
    if (x == 0)
        return -1;
    for (int i = b - 1; i >= 0; --i) {
        if ((x >> i) & UT(1))
            return i;
    }
    return -1; // Should never reach here.
}

} // namespace ref

template <typename T>
void test_bsr(const char* tag) {

  using kdtree::internal::bsr;

  SUBCASE(tag) {

    // 1. Test powers of 2 within domain.
    {
      constexpr int bits = std::numeric_limits<std::make_unsigned_t<T>>::digits;
      for (T p{0}; p < bits; ++p) {
        using U = std::make_unsigned_t<T>;
        U candidate = (U{1} << p);
        T val = static_cast<T>(candidate);
        if (val != 0 && val < std::numeric_limits<T>::max()) {
          INFO("Powers-of-2 candidate: " << int(val));
          CHECK(bsr(val) == ref::bsr(val));
        }
      }
    }

    // 2. Test random values.
    {
      std::mt19937_64 rng(12345);
      std::uniform_int_distribution<unsigned long long> dist(1ULL,
          static_cast<T>(std::numeric_limits<T>::max()));
      constexpr T num_random_tests{10};
      for (T i{0}; i < num_random_tests; ++i) {
        auto rand_val_ull{dist(rng)};
        if constexpr (sizeof(T) < sizeof(unsigned long long))
          rand_val_ull &= ((1ULL << (sizeof(T) * CHAR_BIT)) - 1ULL);
        T val = static_cast<T>(rand_val_ull);
        if (val == 0)
          val = static_cast<T>(1);
        if (val > 1 && val < std::numeric_limits<T>::max()) {
          INFO("Random candidate: " << int(val));
          CHECK(bsr(val) == ref::bsr(val));
        }
      }
    }

    // 3. Test near the maximum value of T.
    {
      constexpr T max_val = std::numeric_limits<T>::max();
      for (T i = 1; i <= 3; ++i) {
        T val = static_cast<T>(max_val - T(i));
        if (val > 1) {
          INFO("Near-max candidate: " << int(val));
          CHECK(bsr(val) == ref::bsr(val));
        }
      }
    }

    // --- Additional tests ---

    // Edge cases: test 0 and the maximum value explicitly.
    SUBCASE("Edge Cases") {
      INFO("Testing bsr with 0 and max value");
      CHECK(bsr(static_cast<T>(0)) == ref::bsr(static_cast<T>(0)));
      CHECK(bsr(std::numeric_limits<T>::max()) == ref::bsr(std::numeric_limits<T>::max()));
    }

    // Test a range of small consecutive numbers.
    SUBCASE("Small values (0 to 15)") {
      for (int i = 0; i < 16; ++i) {
        T val = static_cast<T>(i);
        INFO("Testing small value: " << i);
        CHECK(bsr(val) == ref::bsr(val));
      }
    }

    // Test numbers of the form (2^p - 1).
    SUBCASE("Numbers of the form (2^p - 1)") {
      constexpr int bits = std::numeric_limits<std::make_unsigned_t<T>>::digits;
      for (int p = 1; p < bits; ++p) {
        T candidate = (static_cast<T>(1) << p) - 1;
        if (candidate != 0) {
          INFO("Testing candidate (2^" << p << " - 1): " << int(candidate));
          CHECK(bsr(candidate) == ref::bsr(candidate));
        }
      }
    }

    // For 8-bit types, run an exhaustive test over all non-negative values.
    if constexpr (sizeof(T) == 1) {
      SUBCASE("Exhaustive test for 8-bit non-negative values") {
        const int upper = std::numeric_limits<T>::max();
        for (int i = 0; i <= upper; ++i) {
          T val = static_cast<T>(i);
          INFO("Exhaustive 8-bit candidate: " << i);
          CHECK(bsr(val) == ref::bsr(val));
        }
      }
    }
  }
}

TEST_CASE("kdtree::internal::bsr") {
  test_bsr<std::uint32_t>("[uint32_t]");
  test_bsr<std::uint64_t>("[uint64_t]");
  test_bsr<std::int32_t >("[int32_t]");
  test_bsr<std::int64_t >("[int64_t]");
}
