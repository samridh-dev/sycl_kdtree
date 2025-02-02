/*
 * Filename: kdtree_internal_id.cpp
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

#include <internal.hpp>

using kdtree::internal::id;

template <typename Container>
void fill_increasing(Container& c, int start_value = 1) {
  int val = start_value;
  for (auto& x : c) {
    x = val++;
  }
}

TEST_SUITE("kdtree::internal::id") {

  TEST_CASE("[dim=3][type=[]][maj=row]") {

    std::vector<int> v = {1, 2, 3, 4, 5, 6};

    CHECK(id<int, 3, kdtree::layout::rowmajor>(v, 2, 0, 0) == 1);
    CHECK(id<int, 3, kdtree::layout::rowmajor>(v, 2, 0, 1) == 2);
    CHECK(id<int, 3, kdtree::layout::rowmajor>(v, 2, 0, 2) == 3);
    CHECK(id<int, 3, kdtree::layout::rowmajor>(v, 2, 1, 0) == 4);
    CHECK(id<int, 3, kdtree::layout::rowmajor>(v, 2, 1, 1) == 5);
    CHECK(id<int, 3, kdtree::layout::rowmajor>(v, 2, 1, 2) == 6);

    id<int, 3, kdtree::layout::rowmajor>(v, 2, 1, 2) = 42;
    CHECK(v[5] == 42);

  }

  TEST_CASE("[dim=3][type=[]][maj=col]") {

    std::vector<int> v = {1, 2, 3, 4, 5, 6};

    CHECK(id<int, 3, kdtree::layout::colmajor>(v, 2, 0, 0) == 1);
    CHECK(id<int, 3, kdtree::layout::colmajor>(v, 2, 1, 0) == 2);
    CHECK(id<int, 3, kdtree::layout::colmajor>(v, 2, 0, 1) == 3);
    CHECK(id<int, 3, kdtree::layout::colmajor>(v, 2, 1, 1) == 4);
    CHECK(id<int, 3, kdtree::layout::colmajor>(v, 2, 0, 2) == 5);
    CHECK(id<int, 3, kdtree::layout::colmajor>(v, 2, 1, 2) == 6);

    id<int, 3, kdtree::layout::colmajor>(v, 2, 1, 2) = 99;
    CHECK(v[5] == 99);

  }

  TEST_CASE("[dim=1][type=[][]][maj=row]") {

    std::vector<std::array<int, 3>> v2 = { { {1, 2, 3} }, { {4, 5, 6} } };

    CHECK(id<int, 3, kdtree::layout::rowmajor>(v2, 2, 0, 0) == 1);
    CHECK(id<int, 3, kdtree::layout::rowmajor>(v2, 2, 0, 1) == 2);
    CHECK(id<int, 3, kdtree::layout::rowmajor>(v2, 2, 0, 2) == 3);
    CHECK(id<int, 3, kdtree::layout::rowmajor>(v2, 2, 1, 0) == 4);
    CHECK(id<int, 3, kdtree::layout::rowmajor>(v2, 2, 1, 1) == 5);
    CHECK(id<int, 3, kdtree::layout::rowmajor>(v2, 2, 1, 2) == 6);

    id<int, 3, kdtree::layout::rowmajor>(v2, 2, 0, 1) = 77;
    CHECK(v2[0][1] == 77);

  }

  TEST_CASE("[dim=1][type=[][]][maj=col]") {

    std::vector<std::array<int, 2>> v2 = { { {1, 4} }, { {2, 5} }, { {3, 6} } };

    CHECK(id<int, 3, kdtree::layout::colmajor>(v2, 2, 0, 0) == 1);
    CHECK(id<int, 3, kdtree::layout::colmajor>(v2, 2, 0, 1) == 2);
    CHECK(id<int, 3, kdtree::layout::colmajor>(v2, 2, 0, 2) == 3);
    CHECK(id<int, 3, kdtree::layout::colmajor>(v2, 2, 1, 0) == 4);
    CHECK(id<int, 3, kdtree::layout::colmajor>(v2, 2, 1, 1) == 5);
    CHECK(id<int, 3, kdtree::layout::colmajor>(v2, 2, 1, 2) == 6);

    id<int, 3, kdtree::layout::colmajor>(v2, 2, 1, 2) = 100;
    CHECK(v2[2][1] == 100);

  }

  TEST_CASE("random [dim=3][maj=row]") {

    static constexpr int n   = 5;
    static constexpr int dim = 3;

    std::vector<int> v(n * dim);

    fill_increasing(v, 10);

    for(int i = 0; i < n; ++i) {
      for(int j = 0; j < dim; ++j) {
      CHECK(id<int, dim, kdtree::layout::rowmajor>(v, n, i, j) 
          == v[i*dim + j]);
      }
    }

    CHECK(id<int, dim, kdtree::layout::rowmajor>(v, n, 4, 2) == v[14]);

    id<int, dim, kdtree::layout::rowmajor>(v, n, 3, 1) = 999;
    CHECK(v[3*3 + 1] == 999);

  }

  TEST_CASE("random [dim=3][maj=col]") {

    static constexpr int n   = 5;
    static constexpr int dim = 3;

    std::vector<int> v(n * dim);
    fill_increasing(v, 100);

    for(int j = 0; j < dim; ++j) {
      for(int i = 0; i < n; ++i) {
      CHECK(id<int, dim, kdtree::layout::colmajor>(v, n, i, j) 
          == v[n*j + i]);
      }
    }

    CHECK(id<int, dim, kdtree::layout::colmajor>(v, n, 4, 2) == v[14]);

    id<int, dim, kdtree::layout::colmajor>(v, n, 2, 2) = 777;
    CHECK(v[5*2 + 2] == 777);

  }

  TEST_CASE("[dim=1][type=[][]][maj=row][random]") {

    static constexpr int n = 4;
    std::vector<std::array<int, 3>> v2(n);

    int base = 50;
    for(auto &arr : v2) { for(auto &x : arr) { x = base++; } }

    for(int i = 0; i < n; ++i) {
    for(int j = 0; j < 3; ++j) {
      CHECK(id<int, 3, kdtree::layout::rowmajor>(v2, n, i, j) == v2[i][j]);
    }}

    CHECK(id<int, 3, kdtree::layout::rowmajor>(v2, n, 3, 2) == v2[3][2]);
    
    id<int, 3, kdtree::layout::rowmajor>(v2, n, 2, 1) = 12345;
    CHECK(v2[2][1] == 12345);

  }

  TEST_CASE("[dim=1][type=[][]][maj=col][random]") {

    static constexpr int n = 4;
    static constexpr int dim = 3;
    std::vector<std::array<int, dim>> v2(n);
    
    int base = 1000;
    for (auto &arr : v2) { for (auto &x : arr) { x = base++; } }

    for (int i = 0; i < n; ++i) {
    for (int j = 0; j < dim; ++j) {
      CHECK(id<int, dim, kdtree::layout::colmajor>(v2, n, i, j) == v2[j][i]);
    }}

    CHECK(id<int, dim, kdtree::layout::colmajor>(v2, n, 3, 2) == v2[2][3]);

    id<int, dim, kdtree::layout::colmajor>(v2, n, 1, 2) = 9999;
    CHECK(v2[2][1] == 9999);

  }

}
