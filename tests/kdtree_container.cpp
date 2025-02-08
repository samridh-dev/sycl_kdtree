/*
 * Filename: kdtree_internal_container.cpp
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

#include <container.hpp>

using namespace kdtree::container;

template <typename C>
static void 
fill_increasing(C& c, const int v0 = 1) {
  int val = v0;
  for (auto& x : c) x = val++;
}

// custom container
template <typename T>
struct example_t {
  std::vector<T>     a;
  std::vector<float> b;
  T&       operator[](std::size_t i_)       { return a[i_]; }
  const T& operator[](std::size_t i_) const { return a[i_]; }
};

template <typename T>
struct example2_t {
  std::vector<std::vector<T>> a;
  std::vector<float> b;
  std::vector<T>&       operator[](std::size_t i)       { return a[i]; }
  const std::vector<T>& operator[](std::size_t i) const { return a[i]; }
};


TEST_CASE("[container_1d]") {
  
  // valid types
  static_assert(container_1d<std::vector<int>>,     "[fail]");
  static_assert(container_1d<std::vector<double>>,  "[fail]");
  static_assert(container_1d<std::array<float, 3>>, "[fail]");
  static_assert(container_1d<std::span<int>>,       "[fail]");
  static_assert(container_1d<std::span<double>>,    "[fail]");
  static_assert(container_1d<int*>,                 "[fail]");
  static_assert(container_1d<double*>,              "[fail]");
  static_assert(container_1d<float*>,               "[fail]");

  // custom container
  static_assert(container_1d<example_t<int>>,     "[fail]");

  // invalid types
  static_assert(!container_1d<std::vector<std::string>>,   "[pass]");
  static_assert(!container_1d<std::array<std::string, 3>>, "[pass]");
  static_assert(!container_1d<std::list<int>>,             "[pass]");
  static_assert(!container_1d<std::vector<void*>>,         "[pass]");

  static_assert(!container_1d<int>,    "[pass]");
  static_assert(!container_1d<double>, "[pass]");

  CHECK(1==1);

}

TEST_CASE("[container_2d]") {

  static_assert(container_2d<std::vector<std::vector<int>>>,     "[fail]");
  static_assert(container_2d<std::vector<std::vector<double>>>,  "[fail]");
  static_assert(container_2d<int**>,                             "[fail]");
  static_assert(container_2d<std::vector<std::array<float, 3>>>, "[fail]");
  static_assert(container_2d<example2_t<int>>,                   "[fail]");

  static_assert(!container_2d<std::vector<int>>,                  "[pass]");
  static_assert(!container_2d<std::array<int, 3>>,                "[pass]");
  static_assert(!container_2d<int*>,                              "[pass]");
  // static_assert(!container_2d<std::vector<std::string>>,          "[pass]");
  static_assert(!container_2d<std::span<std::span<std::string>>>, "[pass]");
  static_assert(!container_2d<int>,                               "[pass]");

  CHECK(1 == 1);

}


TEST_CASE("[id][dim=3][type=[]][maj=row]") {

  std::vector<int> v = {1, 2, 3, 4, 5, 6};

  CHECK(id<int, 3, row_major>(v, 2, 0, 0) == 1);
  CHECK(id<int, 3, row_major>(v, 2, 0, 1) == 2);
  CHECK(id<int, 3, row_major>(v, 2, 0, 2) == 3);
  CHECK(id<int, 3, row_major>(v, 2, 1, 0) == 4);
  CHECK(id<int, 3, row_major>(v, 2, 1, 1) == 5);
  CHECK(id<int, 3, row_major>(v, 2, 1, 2) == 6);

  id<int, 3, row_major>(v, 2, 1, 2) = 42;
  CHECK(v[5] == 42);

}

TEST_CASE("[id][dim=3][type=[]][maj=col]") {

  std::vector<int> v = {1, 2, 3, 4, 5, 6};

  CHECK(id<int, 3, col_major>(v, 2, 0, 0) == 1);
  CHECK(id<int, 3, col_major>(v, 2, 1, 0) == 2);
  CHECK(id<int, 3, col_major>(v, 2, 0, 1) == 3);
  CHECK(id<int, 3, col_major>(v, 2, 1, 1) == 4);
  CHECK(id<int, 3, col_major>(v, 2, 0, 2) == 5);
  CHECK(id<int, 3, col_major>(v, 2, 1, 2) == 6);

  id<int, 3, col_major>(v, 2, 1, 2) = 99;
  CHECK(v[5] == 99);

}

TEST_CASE("[id][dim=1][type=[][]][maj=row]") {

  std::vector<std::array<int, 3>> v2 = { { {1, 2, 3} }, { {4, 5, 6} } };

  CHECK(id<int, 3, row_major>(v2, 2, 0, 0) == 1);
  CHECK(id<int, 3, row_major>(v2, 2, 0, 1) == 2);
  CHECK(id<int, 3, row_major>(v2, 2, 0, 2) == 3);
  CHECK(id<int, 3, row_major>(v2, 2, 1, 0) == 4);
  CHECK(id<int, 3, row_major>(v2, 2, 1, 1) == 5);
  CHECK(id<int, 3, row_major>(v2, 2, 1, 2) == 6);

  id<int, 3, row_major>(v2, 2, 0, 1) = 77;
  CHECK(v2[0][1] == 77);

}

TEST_CASE("[id][dim=1][type=[][]][maj=col]") {

  std::vector<std::array<int, 2>> v2 = { { {1, 4} }, { {2, 5} }, { {3, 6} } };

  CHECK(id<int, 3, col_major>(v2, 2, 0, 0) == 1);
  CHECK(id<int, 3, col_major>(v2, 2, 0, 1) == 2);
  CHECK(id<int, 3, col_major>(v2, 2, 0, 2) == 3);
  CHECK(id<int, 3, col_major>(v2, 2, 1, 0) == 4);
  CHECK(id<int, 3, col_major>(v2, 2, 1, 1) == 5);
  CHECK(id<int, 3, col_major>(v2, 2, 1, 2) == 6);

  id<int, 3, col_major>(v2, 2, 1, 2) = 100;
  CHECK(v2[2][1] == 100);

}

TEST_CASE("[swap][dim=3][type=[]][maj=row]") {

  std::vector<int> v = {1, 2, 3, 4, 5, 6};

  swap<int, 3, row_major>(v, 2, 0, 1);

  CHECK(v[0] == 4);
  CHECK(v[1] == 5);
  CHECK(v[2] == 6);
  CHECK(v[3] == 1);
  CHECK(v[4] == 2);
  CHECK(v[5] == 3);
}

TEST_CASE("[swap][dim=3][type=[]][maj=col]") {

  std::vector<int> v = {1, 2, 3, 4, 5, 6};

  swap<int, 3, col_major>(v, 2, 0, 1);

  CHECK(v[0] == 2);
  CHECK(v[1] == 1);
  CHECK(v[2] == 4);
  CHECK(v[3] == 3);
  CHECK(v[4] == 6);
  CHECK(v[5] == 5);

}

TEST_CASE("[swap][dim=1][type=[][]][maj=row]") {

  std::vector<std::array<int, 3>> v2 = { { {1, 2, 3} }, { {4, 5, 6} } };

  swap<int, 3, row_major>(v2, 2, 0, 1);

  CHECK(v2[0][0] == 4);
  CHECK(v2[0][1] == 5);
  CHECK(v2[0][2] == 6);
  CHECK(v2[1][0] == 1);
  CHECK(v2[1][1] == 2);
  CHECK(v2[1][2] == 3);
}

TEST_CASE("[swap][dim=1][type=[][]][maj=col]") {

  std::vector<std::array<int, 2>> v2 = { { {1, 4} }, { {2, 5} }, { {3, 6} } };

  swap<int, 3, col_major>(v2, 2, 0, 1);

  CHECK(v2[0][0] == 4);
  CHECK(v2[1][0] == 5);
  CHECK(v2[2][0] == 6);
  CHECK(v2[0][1] == 1);
  CHECK(v2[1][1] == 2);
  CHECK(v2[2][1] == 3);

}
