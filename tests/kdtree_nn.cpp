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

#include <nn/nn.hpp>
#include <create/create.hpp>

TEST_CASE("[basic_example] kdtree::nn") {

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
    const auto idx = kdtree::nn<int, int, dim>(ctx, q, vec, n);
    CHECK(vec[static_cast<std::size_t>(idx) * dim + 0] == ans[0]);
    CHECK(vec[static_cast<std::size_t>(idx) * dim + 1] == ans[1]);
  }

  SUBCASE("q doesnt equals existing point {46, 63}") {
    std::vector<type_v> q   { 46, 63 };
    std::vector<type_v> ans { 46, 63 };
    const auto idx = kdtree::nn<int, int, dim>(ctx, q, vec, n);
    CHECK(vec[static_cast<std::size_t>(idx) * dim + 0] != ans[0]);
    CHECK(vec[static_cast<std::size_t>(idx) * dim + 1] != ans[1]);
  }

  SUBCASE("q = {50, 50}") {
    std::vector<type_v> q   { 50, 50 };
    std::vector<type_v> ans { 44, 58 };
    const auto idx = kdtree::nn<int, int, dim>(ctx, q, vec, n);
    CHECK(vec[static_cast<std::size_t>(idx) * dim + 0] == ans[0]);
    CHECK(vec[static_cast<std::size_t>(idx) * dim + 1] == ans[1]);
  }

  SUBCASE("q = {70, 20}") {
    std::vector<type_v> q   { 70, 20 };
    std::vector<type_v> ans { 68, 21 };
    const auto idx = kdtree::nn<int, int, dim>(ctx, q, vec, n);
    CHECK(vec[static_cast<std::size_t>(idx) * dim + 0] == ans[0]);
    CHECK(vec[static_cast<std::size_t>(idx) * dim + 1] == ans[1]);
  }

  SUBCASE("q = {48, 45}") {
    std::vector<type_v> q   { 48, 45 };
    std::vector<type_v> ans { 45, 40 };
    const auto idx = kdtree::nn<int, int, dim>(ctx, q, vec, n);
    CHECK(vec[static_cast<std::size_t>(idx) * dim + 0] == ans[0]);
    CHECK(vec[static_cast<std::size_t>(idx) * dim + 1] == ans[1]);
  }

  SUBCASE("q = {100, 100}") {
    std::vector<type_v> q   { 100, 100 };
    std::vector<type_v> ans { 62, 69 };
    const auto idx = kdtree::nn<int, int, dim>(ctx, q, vec, n);
    CHECK(vec[static_cast<std::size_t>(idx) * dim + 0] == ans[0]);
    CHECK(vec[static_cast<std::size_t>(idx) * dim + 1] == ans[1]);
  }

}

#if 1
TEST_CASE("[basic_example][unsigned] kdtree::nn") {

  using type_v = int;
  using type_s = std::size_t;
  using type_f = float;

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
    const auto idx = kdtree::nn<type_f, type_s, dim>(ctx, q, vec, n);
    CHECK(vec[static_cast<std::size_t>(idx) * dim + 0] == ans[0]);
    CHECK(vec[static_cast<std::size_t>(idx) * dim + 1] == ans[1]);
  }

  SUBCASE("q doesnt equals existing pouint32_t {46, 63}") {
    std::vector<type_v> q   { 46, 63 };
    std::vector<type_v> ans { 46, 63 };
    const auto idx = kdtree::nn<type_f, type_s, dim>(ctx, q, vec, n);
    CHECK(vec[static_cast<std::size_t>(idx) * dim + 0] != ans[0]);
    CHECK(vec[static_cast<std::size_t>(idx) * dim + 1] != ans[1]);
  }

  SUBCASE("q = {50, 50}") {
    std::vector<type_v> q   { 50, 50 };
    std::vector<type_v> ans { 44, 58 };
    const auto idx = kdtree::nn<type_f, type_s, dim>(ctx, q, vec, n);
    CHECK(vec[static_cast<std::size_t>(idx) * dim + 0] == ans[0]);
    CHECK(vec[static_cast<std::size_t>(idx) * dim + 1] == ans[1]);
  }

  SUBCASE("q = {70, 20}") {
    std::vector<type_v> q   { 70, 20 };
    std::vector<type_v> ans { 68, 21 };
    const auto idx = kdtree::nn<type_f, type_s, dim>(ctx, q, vec, n);
    CHECK(vec[static_cast<std::size_t>(idx) * dim + 0] == ans[0]);
    CHECK(vec[static_cast<std::size_t>(idx) * dim + 1] == ans[1]);
  }

  SUBCASE("q = {48, 45}") {
    std::vector<type_v> q   { 48, 45 };
    std::vector<type_v> ans { 45, 40 };
    const auto idx = kdtree::nn<type_f, type_s, dim>(ctx, q, vec, n);
    CHECK(vec[static_cast<std::size_t>(idx) * dim + 0] == ans[0]);
    CHECK(vec[static_cast<std::size_t>(idx) * dim + 1] == ans[1]);
  }

  SUBCASE("q = {100, 100}") {
    std::vector<type_v> q   { 100, 100 };
    std::vector<type_v> ans { 62, 69 };
    const auto idx = kdtree::nn<type_f, type_s, dim>(ctx, q, vec, n);
    CHECK(vec[static_cast<std::size_t>(idx) * dim + 0] == ans[0]);
    CHECK(vec[static_cast<std::size_t>(idx) * dim + 1] == ans[1]);
  }

}
#endif

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
static T
nn(const kdtree::context& ctx, const C_query& q, const C_tree& tree, 
   const T n, F rmax = std::numeric_limits<F>::max()) {

  (void) ctx;

  using kdtree::internal::dist::euclidian;

  T best_idx  {0};
  F best_dist {rmax};

  for (T i{0}; i < n; ++i) {

    F dst{
      euclidian<F, T, dim, maj, C_query, maj, C_tree>(q, 1, 0, tree, n, i)
    };

    if (dst == F{0}) {
      continue;
    }

    if (dst < best_dist) {
      best_dist = dst;
      best_idx = i;
    }

  }
  
  return best_idx;
}


}

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


template <std::size_t dim, kdtree::container::layout maj, std::size_t n>
void 
test_nn_impl(void) {
  constexpr std::size_t imax = 1e1;
  using type_v = int;
  using type_s = std::size_t;

  kdtree::context ctx;

  std::vector<type_v> vec(dim * n);
  generate_random_dataset(vec);

  kdtree::create<type_s, dim>(ctx, vec, n);

  for (std::size_t i = 0; i < imax; ++i) {
    SUBCASE(("nn check, i=" + std::to_string(i)).c_str()) {
      std::vector<type_v> q(dim);
      generate_random_dataset(q);

      const auto idx = kdtree::nn<double, int, dim, maj>(ctx, q, vec, n);
      const auto ans = ref::nn<double, int, dim, maj>(ctx, q, vec, n);
      CHECK(idx == ans);
    }
  }

  
  constexpr std::size_t nmax = 16 < n ? 10 : n;

  for (std::size_t i = 0; i < nmax; ++i) {
    SUBCASE(("nn check, idx=" + std::to_string(i)).c_str()) {

      std::vector<type_v> q(dim);
      for (std::size_t j = 0; j < dim; ++j) {
        q[j] = kdtree::container::id<type_s, dim, maj>(vec, n, i, j);
      }

      const auto idx = kdtree::nn<double, uint32_t, dim, maj>(ctx, q, vec, n);
      const auto ans = ref::nn<double, int, dim, maj>(ctx, q, vec, n);
      CHECK(idx == ans);
    }
  }

}

TEST_CASE("[random][case=1] kdtree::nn dim=1 n=1<<6") {
  test_nn_impl<1, kdtree::container::layout::row_major, 1 << 6>();
  test_nn_impl<1, kdtree::container::layout::col_major, 1 << 6>();
}

TEST_CASE("[random][case=1] kdtree::nn dim=2 n=1<<6") {
  test_nn_impl<2, kdtree::container::layout::row_major, 1 << 6>();
  test_nn_impl<2, kdtree::container::layout::col_major, 1 << 6>();
}

TEST_CASE("[random][case=1] kdtree::nn dim=3 n=1<<6") {
  test_nn_impl<3, kdtree::container::layout::row_major, 1 << 6>();
  test_nn_impl<3, kdtree::container::layout::col_major, 1 << 6>();
}

TEST_CASE("[random][case=1] kdtree::nn dim=3 n=1<<6") {
  test_nn_impl<3, kdtree::container::layout::row_major, 1 << 6>();
  test_nn_impl<3, kdtree::container::layout::col_major, 1 << 6>();
}

TEST_CASE("[random][case=1] kdtree::nn dim=8 n=1<<8") {
  test_nn_impl<8, kdtree::container::layout::row_major, 1 << 8>();
  test_nn_impl<8, kdtree::container::layout::col_major, 1 << 8>();
}

TEST_CASE("[random][case=1] kdtree::nn dim=16 n=512") {
  test_nn_impl<16, kdtree::container::layout::row_major, 1 << 9>();
  test_nn_impl<16, kdtree::container::layout::col_major, 1 << 9>();
}

TEST_CASE("[random][case=1] kdtree::nn multiple dims/n") {

  SUBCASE("dim=8, n=1024") {
    test_nn_impl<8, kdtree::container::layout::row_major, 1 << 10>();
    test_nn_impl<8, kdtree::container::layout::col_major, 1 << 10>();
  }

  SUBCASE("dim=16, n=512") {
    test_nn_impl<16, kdtree::container::layout::row_major, 1 << 9>();
    test_nn_impl<16, kdtree::container::layout::col_major, 1 << 9>();
  }

}

