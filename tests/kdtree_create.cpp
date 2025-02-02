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

#include <kdtree.hpp>

#define USE_LARGE_TEST 1

// ingowald/cpukd
namespace internal {

template <typename Ts>
inline Ts 
level_of(Ts i) {
  Ts l = 0;
  i = i + 1; 
  while (i > 1) {
    i >>= 1;
    ++l;
  }
  return l;
}

template <typename Ts, Ts dim, kdtree::layout maj, typename C,
          typename Tv = typename C::value_type>
bool 
none_above(const C &src, Ts n, Ts i, Ts d, const Tv &v) {
  if (i >= n) return true;
  auto x = kdtree::internal::id<Ts, dim, maj>(src, n, i, d);
  if (x > v) return false;
  return none_above<Ts, dim, maj>(src, n, 2*i+1, d, v)
      && none_above<Ts, dim, maj>(src, n, 2*i+2, d, v);
}

template <typename Ts, Ts dim, kdtree::layout maj, typename C,
          typename Tv = typename C::value_type>
bool 
none_below(const C &src, Ts n, Ts i, Ts d, const Tv &v) {
  if (i >= n) return true;
  auto x = kdtree::internal::id<Ts, dim, maj>(src, n, i, d);
  if (x < v) return false;
  return none_below<Ts, dim, maj>(src, n, 2*i+1, d, v)
      && none_below<Ts, dim, maj>(src, n, 2*i+2, d, v);
}

template <typename Ts, Ts dim, kdtree::layout maj, typename C>
bool 
verify_subtree(const C &src, Ts n, Ts i = Ts(0)) {
  if (i >= n) return true;
  Ts l = level_of(i);
  Ts d = l % dim;
  auto v = kdtree::internal::id<Ts, dim, maj>(src, n, i, d);
  if (!none_above<Ts, dim, maj>(src, n, 2*i+1, d, v)) return false;
  if (!none_below<Ts, dim, maj>(src, n, 2*i+2, d, v)) return false;
  return verify_subtree<Ts, dim, maj>(src, n, 2*i+1)
      && verify_subtree<Ts, dim, maj>(src, n, 2*i+2);
}

} // namespace internal

template <typename Ts, Ts dim, kdtree::layout maj = kdtree::layout::rowmajor, 
          typename C>
bool verify_kdtree(const C &src) {
  const Ts n = src.size() / dim;
  return internal::verify_subtree<Ts, dim, maj>(src, n);
}


TEST_CASE("[basic_example] kdtree::create") {

  using type_v = int;
  using type_s = uint8_t;

  constexpr type_s dim    = 2;
  constexpr std::size_t n = 10;

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
    std::vector<type_v> points = {
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

    std::vector<type_v> ans = {
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

    kdtree::create<type_s, dim>(points, n);

    INFO(log_vec(points));
    INFO(log_vec(ans));

    CHECK(points == ans);
  }

  SUBCASE("[type=[]][maj=col]") {

    std::vector<type_v> points = {
      10, 46, 68, 40, 25, 15, 44, 45, 62, 53,
      15, 63, 21, 33, 54, 43, 58, 40, 69, 67,
    };

    std::vector<type_v> ans = {
      46, 15, 53, 40, 44, 68, 62, 10, 45, 25,
      63, 43, 67, 33, 58, 21, 69, 15, 40, 54,
    };

    kdtree::create<type_s, dim, kdtree::layout::colmajor>(points, n);

    INFO(log_vec(points));
    INFO(log_vec(ans));

    CHECK(points == ans);
  }

  SUBCASE("[type=[][]][maj=row]") {
    std::vector<std::vector<type_v>> points = {
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

    std::vector<std::vector<type_v>> ans = {
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

    kdtree::create<type_s, dim>(points, n);

    INFO(log_vec2(points));
    INFO(log_vec2(ans));

    CHECK(points == ans);
  }

  SUBCASE("[type=[][]][maj=col]") {
    std::vector<std::vector<type_v>> points = {
      { 10, 46, 68, 40, 25, 15, 44, 45, 62, 53 },
      { 15, 63, 21, 33, 54, 43, 58, 40, 69, 67 },
    };

    std::vector<std::vector<type_v>> ans = {
      { 46, 15, 53, 40, 44, 68, 62, 10, 45, 25 },
      { 63, 43, 67, 33, 58, 21, 69, 15, 40, 54 },
    };

    kdtree::create<type_s, dim, kdtree::layout::colmajor>(points, n);

    INFO(log_vec2(points));
    INFO(log_vec2(ans));

    CHECK(points == ans);
  }
}


#if USE_LARGE_TEST

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

  SUBCASE(("[N=" + std::to_string(N) + "]" + tag).c_str()) {
    
    CAPTURE(xlim[0]);
    CAPTURE(xlim[1]);
    CAPTURE(N);
    CAPTURE(dim);
        
    SUBCASE("[Ts=uint32_t][type=[]][maj=row]") {
      using Ts = uint32_t;
      constexpr auto maj = kdtree::layout::rowmajor;

      std::vector<T> vec(N * dim);
      generate_random_dataset(vec, xlim[0], xlim[1]);

      kdtree::create<Ts, static_cast<Ts>(dim), maj>(vec, N);
      CHECK(verify_kdtree<Ts, dim, maj>(vec));
    }

    SUBCASE("[Ts=uint32_t][type=[]][maj=col]") {
      using Ts = uint64_t;
      constexpr auto maj = kdtree::layout::colmajor;

      std::vector<T> vec(N * dim);
      generate_random_dataset(vec, xlim[0], xlim[1]);

      kdtree::create<Ts, static_cast<Ts>(dim), maj>(vec, N);
      CHECK(verify_kdtree<Ts, dim, maj>(vec));
    }

    SUBCASE("[Ts=uint32_t][type=[][]][maj=row]") {
      using Ts = uint32_t;
      constexpr auto maj = kdtree::layout::rowmajor;

      std::vector<std::vector<T>> vec(N);
      {
        std::vector<T> tmp(dim);
        for (auto& v : vec) {
          generate_random_dataset(tmp, xlim[0], xlim[1]);
          v = tmp;
        }
      }

      kdtree::create<Ts, static_cast<Ts>(dim), maj>(vec, N);
      CHECK(verify_kdtree<Ts, dim, maj>(vec));
    }

    SUBCASE("[Ts=uint32_t][type=[][]][maj=col]") {
      using Ts = uint64_t;
      constexpr auto maj = kdtree::layout::colmajor;

      std::vector<std::vector<T>> vec(dim);
      {
        std::vector<T> tmp(N);
        for (auto& v : vec) {
          generate_random_dataset(tmp, xlim[0], xlim[1]);
          v = tmp;
        }
      }

      kdtree::create<Ts, static_cast<Ts>(dim), maj>(vec, N);
      CHECK(verify_kdtree<Ts, dim, maj>(vec));
    }

  }

}

TEST_CASE("kdtree::create tests") {
  
  int xlim[2] = { -64, 64 };
  size_t N_max = 4;

  std::vector<size_t> nvec;
  for (size_t i = 0; i <= N_max; ++i) nvec.push_back(i);
  nvec.insert(nvec.end(), {512});

  for (auto n : nvec) {
    test<uint8_t,  8>("[u08_8] kdtree::create", n, xlim);
    test<uint16_t, 9>("[u16_9] kdtree::create", n, xlim);
    test<uint32_t, 6>("[u32_6] kdtree::create", n, xlim);
    test<uint64_t, 3>("[u64_3] kdtree::create", n, xlim);

    test<int8_t,  1>("[i08_1] kdtree::create", n, xlim);
    test<int16_t, 2>("[i16_2] kdtree::create", n, xlim);
    test<int32_t, 4>("[i32_4] kdtree::create", n, xlim);
    test<int64_t, 8>("[i64_8] kdtree::create", n, xlim);

    test<float, 2>("[f32_2] kdtree::create", n, xlim);
    test<float, 5>("[f32_5] kdtree::create", n, xlim);
    test<float, 9>("[f32_9] kdtree::create", n, xlim);

    test<double, 3>("[f64_3] kdtree::create", n, xlim);
    test<double, 7>("[f64_7] kdtree::create", n, xlim);
  }

}

#endif // USE_LARGE_TEST
