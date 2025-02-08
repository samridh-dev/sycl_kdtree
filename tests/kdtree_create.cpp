/*
 * Filename: kdtree_create.cpp
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

#define USE_LARGE_TEST 1

TEST_CASE("[basic_example] kdtree::create") {

  using Tv = int;
  using Ts = std::size_t;

  constexpr Ts dim    = 2;
  constexpr Ts n = 10;

  kdtree::context ctx;

  auto log_vec = [](const auto& vec) {
    std::ostringstream oss;
    oss << "{ ";
    for (size_t i = 0; i < vec.size(); ++i) {
      oss << vec[i];
      if (i + 1 < vec.size()) oss << ", ";
    }
    oss << " }";
    return oss.str();
  };

  auto log_vec2 = [](const auto& vec) {
    std::ostringstream oss;
    oss << "{ ";
    for (size_t i = 0; i < vec.size(); ++i) {
      oss << "{ ";
      for (size_t j = 0; j < vec[i].size(); ++j) {
        oss << vec[i][j];
        if (j + 1 < vec[i].size()) oss << ", ";
      }
      oss << " }";
      if (i + 1 < vec.size()) oss << ", ";
    }
    oss << " }";
    return oss.str();
  };

  SUBCASE("[type=[]][maj=row]") {
    std::vector<Tv> vec = {
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

    std::vector<Tv> ans = {
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

    kdtree::create<Ts, dim, kdtree::container::layout::row_major>(ctx, vec, n);

    INFO(log_vec(vec));
    INFO(log_vec(ans));

    CHECK(vec == ans);
  }

  SUBCASE("[type=[]][maj=col]") {

    std::vector<Tv> vec = {
      10, 46, 68, 40, 25, 15, 44, 45, 62, 53,
      15, 63, 21, 33, 54, 43, 58, 40, 69, 67,
    };

    std::vector<Tv> ans = {
      46, 15, 53, 40, 44, 68, 62, 10, 45, 25,
      63, 43, 67, 33, 58, 21, 69, 15, 40, 54,
    };

    kdtree::create<Ts, dim, kdtree::container::layout::col_major>(ctx, vec, n);

    INFO(log_vec(vec));
    INFO(log_vec(ans));

    CHECK(vec == ans);
  }

  SUBCASE("[type=[][]][maj=row]") {
    std::vector<std::vector<Tv>> vec = {
      { 10, 15 },
      { 46, 63 },
      { 68, 21 },
      { 40, 33 },
      { 25, 54 },
      { 15, 43 },
      { 44, 58 },
      { 45, 40 },
      { 62, 69 },
      { 53, 67 },
    };

    std::vector<std::vector<Tv>> ans = {
      { 46, 63 },
      { 15, 43 },
      { 53, 67 },
      { 40, 33 },
      { 44, 58 },
      { 68, 21 },
      { 62, 69 },
      { 10, 15 },
      { 45, 40 },
      { 25, 54 },
    };

    kdtree::create<Ts, dim, kdtree::container::layout::row_major>(ctx, vec, n);

    INFO(log_vec2(vec));
    INFO(log_vec2(ans));

    CHECK(vec == ans);
  }

  SUBCASE("[type=[][]][maj=col]") {
    std::vector<std::vector<Tv>> vec = {
      { 10, 46, 68, 40, 25, 15, 44, 45, 62, 53 },
      { 15, 63, 21, 33, 54, 43, 58, 40, 69, 67 },
    };

    std::vector<std::vector<Tv>> ans = {
      { 46, 15, 53, 40, 44, 68, 62, 10, 45, 25 },
      { 63, 43, 67, 33, 58, 21, 69, 15, 40, 54 },
    };

    kdtree::create<Ts, dim, kdtree::container::layout::col_major>(ctx, vec, n);

    INFO(log_vec2(vec));
    INFO(log_vec2(ans));

    CHECK(vec == ans);
  }

}

#if USE_LARGE_TEST

// ingowald/cpukd
namespace internal {

template <typename T>
inline constexpr T
level_of(T i) {
  T l{0};
  i = i + 1;
  while (i > 1) {
    i >>= 1;
    ++l;
  }
  return l;
}

template <typename T, T dim, kdtree::container::layout maj, typename C,
          typename Tv = typename C::value_type>
bool
none_above(const C &src, const T n, const T i, const T d, const Tv &v) {
  if (i >= n) return true;
  Tv x{kdtree::container::id<T, dim, maj>(src, n, i, d)};
  if (x > v) return false;
  return none_above<T, dim, maj>(src, n, 2 * i + 1, d, v)
      && none_above<T, dim, maj>(src, n, 2 * i + 2, d, v);
}

template <typename T, T dim, kdtree::container::layout maj, typename C,
          typename Tv = typename C::value_type>
bool
none_below(const C &src, const T n, const T i, const T d, const Tv &v) {
  if (i >= n) return true;
  Tv x{kdtree::container::id<T, dim, maj>(src, n, i, d)};
  if (x < v) return false;
  return none_below<T, dim, maj>(src, n, 2 * i + 1, d, v)
      && none_below<T, dim, maj>(src, n, 2 * i + 2, d, v);
}

template <typename T, T dim, kdtree::container::layout maj, typename C,
          typename Tv = typename C::value_type>
bool
verify_subtree(const C &src, const T n, const T i = T(0)) {
  if (i >= n) return true;
  T l {level_of(i)};
  T d {l % dim};
  Tv v{kdtree::container::id<T, dim, maj>(src, n, i, d)};
  if (!none_above<T, dim, maj>(src, n,  2 * i + 1, d, v)) return false;
  if (!none_below<T, dim, maj>(src, n,  2 * i + 2, d, v)) return false;
  return verify_subtree<T, dim, maj>(src, n,  2 * i + 1)
      && verify_subtree<T, dim, maj>(src, n,  2 * i + 2);
}

} // namespace internal

template <typename T, T dim,
          kdtree::container::layout maj = kdtree::container::layout::row_major,
          typename C>
bool 
verify_kdtree(const C &src) {
  const T n{static_cast<T>(src.size()) / dim};
  return internal::verify_subtree<T, dim, maj>(src, n);
}

#include <random>
template <typename C, typename T = typename C::value_type>
static void 
generate_random_dataset(C& src, T xmin, T xmax) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<T> dist(xmin, xmax);
  std::generate(src.begin(), src.end(), [&]() { return dist(gen); });
}

template <typename T, std::size_t dim>
static void 
test(const std::string& tag, std::size_t N, int xlim[2]) {

  if (N <= 0) {
    return;
  }

  kdtree::context ctx;

  SUBCASE(("[N=" + std::to_string(N) + "]" + tag).c_str()) {
    
    CAPTURE(xlim[0]);
    CAPTURE(xlim[1]);
    CAPTURE(N);
    CAPTURE(dim);
        
    SUBCASE("[Ts=uint32_t][type=[]][maj=row]") {
      using Ts = uint32_t;
      constexpr auto maj{kdtree::container::layout::row_major};

      std::vector<T> vec(N * dim);
      generate_random_dataset(vec, xlim[0], xlim[1]);

      kdtree::create<Ts, static_cast<Ts>(dim), maj>(ctx, vec, N);
      CHECK(verify_kdtree<Ts, dim, maj>(vec));
    }

    SUBCASE("[Ts=uint32_t][type=[]][maj=col]") {
      using Ts = uint64_t;
      constexpr auto maj{kdtree::container::layout::col_major};

      std::vector<T> vec(N * dim);
      generate_random_dataset(vec, xlim[0], xlim[1]);

      kdtree::create<Ts, dim, maj>(ctx, vec, N);
      CHECK(verify_kdtree<Ts, dim, maj>(vec));
    }

    SUBCASE("[Ts=uint32_t][type=[][]][maj=row]") {
      using Ts = uint32_t;
      constexpr auto maj{kdtree::container::layout::row_major};

      std::vector<std::vector<T>> vec(N);
      {
        std::vector<T> tmp(dim);
        for (auto& v : vec) {
          generate_random_dataset(tmp, xlim[0], xlim[1]);
          v = tmp;
        }
      }

      kdtree::create<Ts, static_cast<Ts>(dim), maj>(ctx, vec, N);
      CHECK(verify_kdtree<Ts, dim, maj>(vec));
    }

    SUBCASE("[Ts=uint32_t][type=[][]][maj=col]") {
      using Ts = uint64_t;
      constexpr auto maj{kdtree::container::layout::col_major};

      std::vector<std::vector<T>> vec(dim);
      {
        std::vector<T> tmp(N);
        for (auto& v : vec) {
          generate_random_dataset(tmp, xlim[0], xlim[1]);
          v = tmp;
        }
      }

      kdtree::create<Ts, dim, maj>(ctx, vec, N);
      CHECK(verify_kdtree<Ts, dim, maj>(vec));
    }

  }

}

#define TEST_CREATE(index, type, dim) \
TEST_CASE("[case=" #index "] kdtree::create") { \
  int xlim[2] = { -64, 64 }; \
  size_t N_MAX{32}; \
  std::vector<size_t> nvec; \
  for (size_t i{0}; i <= N_MAX; ++i) nvec.push_back(i); \
  nvec.insert(nvec.end(), {64, 256, 512}); \
  for (auto n : nvec) test<type, dim>("[type_dim]", n, xlim); \
}

TEST_CREATE(1,  uint16_t, 1)
TEST_CREATE(2,  uint32_t, 1)
TEST_CREATE(3,  uint64_t, 3)
TEST_CREATE(4,  int16_t,  4)
TEST_CREATE(5,  int32_t,  4)
TEST_CREATE(6,  int64_t,  8)
TEST_CREATE(7,  float,    2)
TEST_CREATE(8,  float,    5)
TEST_CREATE(9,  float,    9)
TEST_CREATE(10, double,   3)
TEST_CREATE(11, double,   7)

#endif // USE_LARGE_TEST
