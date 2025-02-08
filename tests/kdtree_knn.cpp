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

#include <create/create.hpp>

#include <knn.hpp>

#include <random>
template <typename C>
void 
generate_random_dataset(C& v) {

  using T = typename C::value_type;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<T> dist(std::numeric_limits<T>::min(),
                                        std::numeric_limits<T>::max());

  for (auto& i : v) i = dist(gen);

}

TEST_CASE("[nn specialization] kdtree::knn with k=1") {

  using type_v = int;
  using type_s = std::size_t;

  constexpr type_s dim = 2;
  constexpr type_s n   = 10;

  kdtree::context ctx;

  std::vector<type_v> vec = {
    10, 15,
    46, 63,
    68, 21,
    40, 33,
    25, 54,
    15, 43,
    44, 58,
    45, 40,
    62, 69,
    53, 67,
  };

  kdtree::create<type_s, dim>(ctx, vec, n);

  SUBCASE("q = {1, 1}") {
    std::vector<type_v> q   { 1, 1 };
    std::vector<type_v> ans { 10, 15 };
    
    const auto idxv = kdtree::knn<int, int, dim>(ctx, q, vec, n, 1);
    const auto idx  = idxv[0];
    
    CHECK(vec[idx*dim + 0] == ans[0]);
    CHECK(vec[idx*dim + 1] == ans[1]);
  }

  SUBCASE("q equals existing point {46, 63}") {
    std::vector<type_v> q   { 46, 63 };
    std::vector<type_v> ans { 46, 63 };
    
    const auto idxv = kdtree::knn<int, int, dim>(ctx, q, vec, n, 1);
    const auto idx  = idxv[0];

    CHECK(vec[idx*dim + 0] == ans[0]);
    CHECK(vec[idx*dim + 1] == ans[1]);
  }

  SUBCASE("q = {50, 50}") {
    std::vector<type_v> q   { 50, 50 };
    std::vector<type_v> ans { 44, 58 };
    
    const auto idxv = kdtree::knn<int, int, dim>(ctx, q, vec, n, 1);
    const auto idx  = idxv[0];

    CHECK(vec[idx*dim + 0] == ans[0]);
    CHECK(vec[idx*dim + 1] == ans[1]);
  }

  SUBCASE("q = {70, 20}") {
    std::vector<type_v> q   { 70, 20 };
    std::vector<type_v> ans { 68, 21 };
    
    const auto idxv = kdtree::knn<int, int, dim>(ctx, q, vec, n, 1);
    const auto idx  = idxv[0];

    CHECK(vec[idx*dim + 0] == ans[0]);
    CHECK(vec[idx*dim + 1] == ans[1]);
  }

  SUBCASE("q = {48, 45}") {
    std::vector<type_v> q   { 48, 45 };
    std::vector<type_v> ans { 45, 40 };
    
    const auto idxv = kdtree::knn<int, int, dim>(ctx, q, vec, n, 1);
    const auto idx  = idxv[0];

    CHECK(vec[idx*dim + 0] == ans[0]);
    CHECK(vec[idx*dim + 1] == ans[1]);
  }

  SUBCASE("q = {100, 100}") {
    std::vector<type_v> q   { 100, 100 };
    std::vector<type_v> ans { 62, 69 };
    
    const auto idxv = kdtree::knn<int, int, dim>(ctx, q, vec, n, 1);
    const auto idx  = idxv[0];

    CHECK(vec[idx*dim + 0] == ans[0]);
    CHECK(vec[idx*dim + 1] == ans[1]);
  }
}

namespace ref {

template<typename F, typename T, T dim,
         kdtree::container::layout maj = kdtree::container::layout::row_major,
         typename C_query, typename C_tree>
requires kdtree::container::container_1d<C_query>
      && kdtree::container::container<C_tree>
      && std::is_integral_v<T>
      && std::is_arithmetic_v<F>
      && std::is_same_v<kdtree::container::get_primitive_t<C_query>,
                        kdtree::container::get_primitive_t<C_tree>>
std::vector<T>
knn(const kdtree::context& ctx,
    const C_query&        q,
    const C_tree&         tree,
    const T               n,
    const T               k,
    F                     rmax = std::numeric_limits<F>::max())
{
  using kdtree::internal::dist::euclidian;
  (void)ctx;  // not used in this brute force

  // We'll store (distance, index) pairs
  std::vector<std::pair<F,T>> dist_idx;
  dist_idx.reserve(n);

  for (T i = 0; i < n; ++i) {
    F d = euclidian<F, T, dim, maj, C_query, maj, C_tree>(q, 1, 0, tree, n, i);
    // If you want to respect rmax, do:
    // if (d <= rmax) dist_idx.emplace_back(d, i);
    // else skip
    //
    // For now, we just store them all:
    dist_idx.emplace_back(d, i);
  }

  // Sort by distance ascending
  std::sort(dist_idx.begin(), dist_idx.end(),
            [](auto& lhs, auto& rhs){
                return lhs.first < rhs.first;
            });

  // Take first k
  std::vector<T> out;
  out.reserve(k);

  // But watch for the case k > n (avoid out of range)
  const T limit = std::min(k, n);
  for (T i = 0; i < limit; ++i) {
    out.push_back(dist_idx[i].second);
  }
  return out;
}

}

template <std::size_t dim, kdtree::container::layout maj, std::size_t n>
void test_knn_impl(void)
{

  using type_v = int;
  using type_s = std::size_t;

  kdtree::context ctx;
  std::vector<type_v> vec(dim * n);

  generate_random_dataset(vec);

  kdtree::create<type_s, dim>(ctx, vec, n);

  std::string layout = (maj == kdtree::container::layout::row_major) 
                     ? "row_major" 
                     : "col_major";

  SUBCASE(("[dim="+std::to_string(dim)
         +"][layout="+layout
         +"][n="+std::to_string(n)+"]").c_str()) {

    std::vector<type_v> q(dim);
    generate_random_dataset(q);

    for (type_s k=1; k < 5; ++k) {

      auto ref_idx  =    ref::knn<float, int, dim, maj>(ctx, q, vec, n, k);
      auto test_idx = kdtree::knn<float, int, dim, maj>(ctx, q, vec, n, k);

      REQUIRE(test_idx.size() == ref_idx.size());
      for (std::size_t j=0; j < test_idx.size(); j++) {
        CHECK(test_idx[j] == ref_idx[j]);
      }
    }
  }
}

TEST_CASE("[random] kdtree::knn multiple k values") {

  test_knn_impl<1, kdtree::container::layout::row_major, 1 << 6>();
  test_knn_impl<1, kdtree::container::layout::col_major, 1 << 6>();

  test_knn_impl<2, kdtree::container::layout::row_major, 1 << 6>();
  test_knn_impl<2, kdtree::container::layout::col_major, 1 << 6>();

  test_knn_impl<3, kdtree::container::layout::row_major, 1 << 6>();
  test_knn_impl<3, kdtree::container::layout::col_major, 1 << 6>();

  test_knn_impl<4, kdtree::container::layout::row_major, 1 << 6>();
  test_knn_impl<4, kdtree::container::layout::col_major, 1 << 6>();

  test_knn_impl<6, kdtree::container::layout::row_major, 1 << 6>();
  test_knn_impl<6, kdtree::container::layout::col_major, 1 << 6>();

}
