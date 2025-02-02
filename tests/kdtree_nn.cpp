/*
 * Filename: kdtree_nn.cpp
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

#include <layout.hpp>
#include <nn.hpp>

TEST_CASE("[basic_example] kdtree::nn") {

    constexpr kdtree::layout maj = kdtree::layout::rowmajor;
    constexpr uint8_t        dim = 2;
    constexpr std::size_t    n   = 10;

    std::vector<int> set = {
      46, 63,
      15, 43,
      53, 67,
      40, 33,
      44, 58,
      68, 21,
      62, 69,
      10, 15,
      45, 40,
      25, 54,
    };

    SUBCASE("q={11,12}") {

      std::vector<int> q{11, 12};
      std::vector<int> ans{10, 15};

      const auto i = kdtree::nn<int8_t, int, dim, maj>(q, set, n);

      for (int j = 0; j < dim; ++j) CHECK(set[dim * i + j] == ans[j]);

    }

  SUBCASE("q={50, 65}") {
    std::vector<int> q{50, 65};
    std::vector<int> ans{53, 67};

    const auto i = kdtree::nn<int8_t, int, dim, maj>(q, set, n);

    for (int j = 0; j < dim; ++j) CHECK(set[dim * i + j] == ans[j]);
  }

  SUBCASE("q={20, 50}") {
    std::vector<int> q{20, 50};
    std::vector<int> ans{25, 54};

    const auto i = kdtree::nn<int8_t, int, dim, maj>(q, set, n);

    for (int j = 0; j < dim; ++j) CHECK(set[dim * i + j] == ans[j]);
  }

  SUBCASE("q={60, 70}") {
    std::vector<int> q{60, 70};
    std::vector<int> ans{62, 69};

    const auto i = kdtree::nn<int8_t, int, dim, maj>(q, set, n);

    for (int j = 0; j < dim; ++j) CHECK(set[dim * i + j] == ans[j]);
  }

  SUBCASE("q={39, 32}") {
    std::vector<int> q{39, 32};
    std::vector<int> ans{40, 33};

    const auto i = kdtree::nn<int8_t, int, dim, maj>(q, set, n);

    for (int j = 0; j < dim; ++j) CHECK(set[dim * i + j] == ans[j]);
  }

  SUBCASE("q={47, 60}") {
    std::vector<int> q{47, 60};
    std::vector<int> ans{46, 63};

    const auto i = kdtree::nn<int8_t, int, dim, maj>(q, set, n);

    for (int j = 0; j < dim; ++j) CHECK(set[dim * i + j] == ans[j]);
  }

}
