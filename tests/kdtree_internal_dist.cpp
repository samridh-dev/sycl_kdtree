/*
 * Filename: kdtree_internal_dist.cpp
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

#include "layout.hpp" 
#include "container.hpp" 
#include "internal.hpp" 

namespace reference {

template <typename Ts, typename Tf, Ts dim, kdtree::layout maj,
          typename Cx, typename Cy>
constexpr Tf
euclidian(const Cx& x, const Ts x_n, const Cy& y, const Ts y_n,
          const Ts x_i, const Ts y_i) {

  static_assert(std::is_same_v<kdtree::container::get_primitive_type_t<Cx>,
                               kdtree::container::get_primitive_type_t<Cy>>,
                "`Cx` and `Cy` must have the same primitive type");

  // using Tv = kdtree::container::get_primitive_type_t<Cx>; // no longer needed

  using kdtree::internal::id;

  Tf sum = Tf{0};  // Use Tf as the accumulator type
  for (Ts i = 0; i < dim; ++i) {
    // Compute the difference as Tf so no overflow happens
    Tf d = static_cast<Tf>(id<Ts, dim, maj>(x, x_n, x_i, i)) - 
           static_cast<Tf>(id<Ts, dim, maj>(y, y_n, y_i, i));
    sum += d * d;
  }

  return sum;
}

} // namespace reference
  
template <typename T>
static std::vector<T> make_random_data(std::mt19937_64& rng, int size) {
  std::uniform_int_distribution<T> dist(std::numeric_limits<T>::min(),
                                        std::numeric_limits<T>::max());
  std::vector<T> v(size);
  for (auto& val : v) {
    val = dist(rng);
  }
  return v;
}

template <typename Ts, typename Tf, Ts dim, 
          kdtree::layout maj = kdtree::layout::rowmajor>
void test(int max_items, int num_tests) {
  std::mt19937_64 rng(12345ULL);

  using namespace kdtree::internal;

  for (int i = 0; i < num_tests; i++) {

    std::uniform_int_distribution<int> dsize(1, max_items);

    int x_n = dsize(rng);
    int y_n = dsize(rng);

    int total_x = x_n * 4;  
    int total_y = y_n * 4;

    auto x_data = make_random_data<Ts>(rng, total_x);
    auto y_data = make_random_data<Ts>(rng, total_y);

    std::uniform_int_distribution<int> dix(0, total_x > 0 ? total_x - 1 : 0);
    std::uniform_int_distribution<int> diy(0, total_y > 0 ? total_y - 1 : 0);

    int x_i = dix(rng);
    int y_i = diy(rng);

    if (x_i >= x_n) x_i = x_n - 1;
    if (y_i >= y_n) y_i = y_n - 1;

    auto dval  = dist::euclidian<Ts, Tf, dim, maj>(x_data, Ts(x_n), y_data, 
                                             Ts(y_n), Ts(x_i), Ts(y_i));
    auto rval  = reference::euclidian<Ts, Tf, dim, maj>(x_data, Ts(x_n), y_data,
                                                   Ts(y_n), Ts(x_i), Ts(y_i));

    CHECK(dval == rval);

  }

}

// -----------------------------------------------------------------------------
// TEST CASES using doctest
// -----------------------------------------------------------------------------

TEST_CASE("[1][uint16_t][float]") { test<uint16_t, float, 1>(100, 500); }
TEST_CASE("[2][uint16_t][float]") { test<uint16_t, float, 2>(100, 500); }
TEST_CASE("[3][uint16_t][float]") { test<uint16_t, float, 3>(100, 500); }
TEST_CASE("[4][uint16_t][float]") { test<uint16_t, float, 4>(100, 500); }
TEST_CASE("[9][uint16_t][float]") { test<uint16_t, float, 9>(100, 500); }
TEST_CASE("[2][uint32_t][float]") { test<uint32_t, float, 2>(100, 500); }
TEST_CASE("[9][uint32_t][float]") { test<uint32_t, float, 9>(100, 500); }
TEST_CASE("[2][uint64_t][float]") { test<uint64_t, float, 2>(100, 500); }
TEST_CASE("[9][uint64_t][float]") { test<uint64_t, float, 9>(100, 500); }

TEST_CASE("[1][uint16_t][double]") { test<uint16_t, double, 1>(100, 500); }
TEST_CASE("[2][uint16_t][double]") { test<uint16_t, double, 2>(100, 500); }
TEST_CASE("[3][uint16_t][double]") { test<uint16_t, double, 3>(100, 500); }
TEST_CASE("[4][uint16_t][double]") { test<uint16_t, double, 4>(100, 500); }
TEST_CASE("[9][uint16_t][double]") { test<uint16_t, double, 9>(100, 500); }
TEST_CASE("[2][uint32_t][double]") { test<uint32_t, double, 2>(100, 500); }
TEST_CASE("[9][uint32_t][double]") { test<uint32_t, double, 9>(100, 500); }
TEST_CASE("[2][uint64_t][double]") { test<uint64_t, double, 2>(100, 500); }
TEST_CASE("[9][uint64_t][double]") { test<uint64_t, double, 9>(100, 500); }
