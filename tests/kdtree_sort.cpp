/*
 * Filename: kdtree_sort.cpp
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
#include <sort/sort.hpp>

#include <vector>
#include <algorithm>
#include <random>

// Payload structure used for sorting.
template <typename T>
struct e1 {
  std::vector<T> v;
  
  template <typename Ts>
  bool less(Ts i_, Ts j_) { 
    return v[i_] < v[j_]; 
  }
  
  template <typename Ts>
  void swap(Ts i_, Ts j_) { 
    std::swap(v[i_], v[j_]); 
  }

};

TEST_CASE("[odd_even::sort]") {
  kdtree::context ctx;
  using U = std::size_t;

  SUBCASE("Empty vector") {
    e1<int> arr{ {} };
    kdtree::odd_even::sort(ctx, arr, U{0}, U{arr.v.size()});
    REQUIRE(arr.v.empty());
  }

  SUBCASE("Single element") {
    e1<int> arr{ {42} };
    kdtree::odd_even::sort(ctx, arr, U{0}, U{arr.v.size()});
    REQUIRE(arr.v.size() == 1);
    REQUIRE(arr.v[0] == 42);
  }

  SUBCASE("Already sorted vector") {
    e1<int> arr{ {1, 2, 3, 4, 5} };
    kdtree::odd_even::sort(ctx, arr, U{0}, U{arr.v.size()});
    REQUIRE(arr.v == std::vector<int>{1, 2, 3, 4, 5});
  }

  SUBCASE("Unsorted vector") {
    e1<int> arr{ {3, 1, 4, 1, 5, 9, 2, 6, 5} };
    kdtree::odd_even::sort(ctx, arr, U{0}, U{arr.v.size()});
    auto expected = arr.v;
    std::sort(expected.begin(), expected.end());
    REQUIRE(arr.v == expected);
  }

  SUBCASE("Vector with duplicates") {
    e1<int> arr{ {5, 3, 5, 3, 5, 3} };
    kdtree::odd_even::sort(ctx, arr, U{0}, U{arr.v.size()});
    auto expected = arr.v;
    std::sort(expected.begin(), expected.end());
    REQUIRE(arr.v == expected);
  }

  SUBCASE("Sort subrange with non-zero offset") {
    e1<int> arr{ {10, 3, 5, 2, 9} };
    kdtree::odd_even::sort(ctx, arr, U{1}, U{4});
    std::vector<int> expected = {10, 2, 3, 5, 9};
    REQUIRE(arr.v == expected);
  }

  SUBCASE("Sort subrange in the middle") {
    e1<int> arr{ {9, 3, 4, 1, 7, 8} };
    kdtree::odd_even::sort(ctx, arr, U{1}, U{5});
    std::vector<int> expected = {9, 1, 3, 4, 7, 8};
    REQUIRE(arr.v == expected);
  }

  SUBCASE("Empty subrange with offset") {
    e1<int> arr{ {8, 6, 7, 5, 3, 0, 9} };
    auto original = arr.v;
    kdtree::odd_even::sort(ctx, arr, U{3}, U{3});
    REQUIRE(arr.v == original);
  }

  SUBCASE("Sort subrange starting from offset to end") {
    e1<int> arr{ {4, 2, 3, 1, 0} };
    kdtree::odd_even::sort(ctx, arr, U{2}, U{arr.v.size()});
    std::vector<int> expected = {4, 2, 0, 1, 3};
    REQUIRE(arr.v == expected);
  }

  SUBCASE("Large random dataset") {
    const std::size_t size = 10000;
    e1<int> arr;
    arr.v.resize(size);
    std::mt19937 rng(42); // fixed seed for reproducibility
    std::uniform_int_distribution<int> dist(0, 100000);
    for (auto& x : arr.v) {
      x = dist(rng);
    }
    
    auto expected = arr.v;
    std::sort(expected.begin(), expected.end());
    
    kdtree::odd_even::sort(ctx, arr, U{0}, U{arr.v.size()});
    REQUIRE(arr.v == expected);

  }

}

TEST_CASE("[sort]") {
  kdtree::context ctx;
  using U = std::size_t;

  SUBCASE("Empty vector") {
    e1<int> arr{ {} };
    kdtree::sort(ctx, arr, U{0}, U{arr.v.size()});
    REQUIRE(arr.v.empty());
  }

  SUBCASE("Single element") {
    e1<int> arr{ {42} };
    kdtree::sort(ctx, arr, U{0}, U{arr.v.size()});
    REQUIRE(arr.v.size() == 1);
    REQUIRE(arr.v[0] == 42);
  }

  SUBCASE("Already sorted vector") {
    e1<int> arr{ {1, 2, 3, 4, 5} };
    kdtree::sort(ctx, arr, U{0}, U{arr.v.size()});
    REQUIRE(arr.v == std::vector<int>{1, 2, 3, 4, 5});
  }

  SUBCASE("Unsorted vector") {
    e1<int> arr{ {3, 1, 4, 1, 5, 9, 2, 6, 5} };
    kdtree::sort(ctx, arr, U{0}, U{arr.v.size()});
    auto expected = arr.v;
    std::sort(expected.begin(), expected.end());
    REQUIRE(arr.v == expected);
  }

  SUBCASE("Vector with duplicates") {
    e1<int> arr{ {5, 3, 5, 3, 5, 3} };
    kdtree::sort(ctx, arr, U{0}, U{arr.v.size()});
    auto expected = arr.v;
    std::sort(expected.begin(), expected.end());
    REQUIRE(arr.v == expected);
  }

  SUBCASE("Sort subrange with non-zero offset") {
    e1<int> arr{ {10, 3, 5, 2, 9} };
    kdtree::sort(ctx, arr, U{1}, U{4});
    std::vector<int> expected = {10, 2, 3, 5, 9};
    REQUIRE(arr.v == expected);
  }

  SUBCASE("Sort subrange in the middle") {
    e1<int> arr{ {9, 3, 4, 1, 7, 8} };
    kdtree::sort(ctx, arr, U{1}, U{5});
    std::vector<int> expected = {9, 1, 3, 4, 7, 8};
    REQUIRE(arr.v == expected);
  }

  SUBCASE("Empty subrange with offset") {
    e1<int> arr{ {8, 6, 7, 5, 3, 0, 9} };
    auto original = arr.v;
    kdtree::sort(ctx, arr, U{3}, U{3});
    REQUIRE(arr.v == original);
  }

  SUBCASE("Sort subrange starting from offset to end") {
    e1<int> arr{ {4, 2, 3, 1, 0} };
    kdtree::sort(ctx, arr, U{2}, U{arr.v.size()});
    std::vector<int> expected = {4, 2, 0, 1, 3};
    REQUIRE(arr.v == expected);
  }
  
  SUBCASE("Large random dataset") {
    const std::size_t size = 10000;
    e1<int> arr;
    arr.v.resize(size);
    std::mt19937 rng(43);
    std::uniform_int_distribution<int> dist(0, 100000);
    for (auto& x : arr.v) {
      x = dist(rng);
    }
    
    auto expected = arr.v;
    std::sort(expected.begin(), expected.end());
    
    kdtree::sort(ctx, arr, U{0}, U{arr.v.size()});
    REQUIRE(arr.v == expected);
  }
}

TEST_CASE("[bitonic::sort]") {
  kdtree::context ctx;
  using U = std::size_t;

  SUBCASE("Empty vector") {
    e1<int> arr{ {} };
    kdtree::bitonic::sort(ctx, arr, U{0}, U{arr.v.size()});
    REQUIRE(arr.v.empty());
  }

  SUBCASE("Single element") {
    e1<int> arr{ {42} };
    kdtree::bitonic::sort(ctx, arr, U{0}, U{arr.v.size()});
    REQUIRE(arr.v.size() == 1);
    REQUIRE(arr.v[0] == 42);
  }

  SUBCASE("Already sorted vector (size = 4)") {
    e1<int> arr{ {1, 2, 3, 4} }; // 4 is 2^2.
    kdtree::bitonic::sort(ctx, arr, U{0}, U{arr.v.size()});
    REQUIRE(arr.v == std::vector<int>{1, 2, 3, 4});
  }

  SUBCASE("Unsorted vector (size = 8)") {
    e1<int> arr{ {8, 3, 7, 4, 2, 6, 5, 1} }; // 8 is 2^3.
    kdtree::bitonic::sort(ctx, arr, U{0}, U{arr.v.size()});
    auto expected = arr.v;
    std::sort(expected.begin(), expected.end());
    REQUIRE(arr.v == expected);
  }

  SUBCASE("Vector with duplicates (size = 8)") {
    e1<int> arr{ {5, 3, 5, 3, 5, 3, 5, 3} };
    kdtree::bitonic::sort(ctx, arr, U{0}, U{arr.v.size()});
    auto expected = arr.v;
    std::sort(expected.begin(), expected.end());
    REQUIRE(arr.v == expected);
  }

  SUBCASE("Sort subrange in the middle (subrange size = 4)") {
    e1<int> arr{ {9, 7, 3, 4, 1, 2, 8, 6} };
    kdtree::bitonic::sort(ctx, arr, U{2}, U{6});
    std::vector<int> expected = {9, 7, 1, 2, 3, 4, 8, 6};
    REQUIRE(arr.v == expected);
  }

  SUBCASE("Large random dataset (size = 1024)") {
    const std::size_t size = 1024;
    e1<int> arr;
    arr.v.resize(size);
    std::mt19937 rng(44);
    std::uniform_int_distribution<int> dist(0, 100000);
    for (auto& x : arr.v) {
      x = dist(rng);
    }
    
    auto expected = arr.v;
    std::sort(expected.begin(), expected.end());
    
    kdtree::bitonic::sort(ctx, arr, U{0}, U{arr.v.size()});
    REQUIRE(arr.v == expected);
  }
}
