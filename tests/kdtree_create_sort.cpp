/*
 * Filename: kdtree_create_sort.cpp
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

template <typename Ts, Ts dim, kdtree::layout maj, typename Ct, typename Cv>
bool 
is_sorted(const Ct &tag, const Cv &src, Ts n, Ts l) {

  const Ts d = l % dim;

  auto less = [&](const Ts i_, const Ts j_) {
    return (tag[i_] < tag[j_]) ||
           (
             (tag[i_] == tag[j_]) &&
             (kdtree::internal::id<Ts, dim, maj>(src, n, i_, d) <
              kdtree::internal::id<Ts, dim, maj>(src, n, j_, d))
           );
  };

  for (Ts i = 0; i + 1 < n; ++i) {
    if (less(i+1, i)) {
      return false;
    }
  }

  return true;

}

template <typename C, typename T = typename C::value_type>
static void 
generate_random_dataset(C& src, T xmin, T xmax) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<T> dist(xmin, xmax);
  std::generate(src.begin(), src.end(), [&]() { return dist(gen); });
}

template <typename T, int dim>
static void 
test(const std::string& tag, int N, int xlim[2]) {

  if (N <= 0) {
    return;
  }
  
  using kdtree::internal::create::sort;

  SUBCASE(("[N=" + std::to_string(N) + "]" + tag).c_str()) {
    
    for (int l = 0; l < dim; ++l) {
      SUBCASE(("[l=" + std::to_string(l) + "]").c_str()) {
      
        SUBCASE("[type=[]][maj=row]") {
          using Ts = uint32_t;
          const auto maj = kdtree::layout::rowmajor;

          std::vector<T> vec(N * dim);
          std::vector<Ts> tag(N);

          generate_random_dataset(vec, xlim[0], xlim[1]);
          generate_random_dataset(tag, T(0), T(8));

          sort<Ts, static_cast<Ts>(dim), maj>(tag, vec, N, l);
          CHECK(is_sorted<Ts, static_cast<Ts>(dim), maj>(tag, vec, N, l));
        }

        SUBCASE("[type=[]][maj=col]") {
          using Ts = uint32_t;
          const auto maj = kdtree::layout::colmajor;

          std::vector<T> vec(N * dim);
          std::vector<Ts> tag(N);

          generate_random_dataset(vec, xlim[0], xlim[1]);
          generate_random_dataset(tag, T(0), T(8));

          sort<Ts, static_cast<Ts>(dim), maj>(tag, vec, N, l);
          CHECK(is_sorted<Ts, static_cast<Ts>(dim), maj>(tag, vec, N, l));
        }

        SUBCASE("[type=[][]][maj=row]") {

          using Ts = uint64_t;
          const auto maj = kdtree::layout::rowmajor;

          std::vector<std::vector<T>> vec(N);
          std::vector<Ts> tag(N);
          
          {
            std::vector<T> tmp(dim);
            for (auto& v : vec) {
              generate_random_dataset(tmp, xlim[0], xlim[1]);
              v = tmp;
            }
          }
          generate_random_dataset(tag, T(0), T(8));

          sort<Ts, static_cast<Ts>(dim), maj>(tag, vec, N, l);
          CHECK(is_sorted<Ts, static_cast<Ts>(dim), maj>(tag, vec, N, l));

        }

        SUBCASE("[type=[][]][maj=col]") {

          using Ts = uint32_t;
          const auto maj = kdtree::layout::colmajor;

          std::vector<std::vector<T>> vec(dim);
          std::vector<Ts> tag(N);
          
          {
            std::vector<T> tmp(N);
            for (auto& v : vec) {
              generate_random_dataset(tmp, xlim[0], xlim[1]);
              v = tmp;
            }
          }
          generate_random_dataset(tag, T(0), T(8));

          sort<Ts, static_cast<Ts>(dim), maj>(tag, vec, N, l);
          CHECK(is_sorted<Ts, static_cast<Ts>(dim), maj>(tag, vec, N, l));

        }

      }
    }

  }

}

TEST_CASE("kdtree::create::sort tests") {
  
  int xlim[2] = { -512, 512 };

  std::vector<size_t> nvec;
  for (size_t i = 0; i <= 32; ++i) nvec.push_back(i);
  nvec.insert(nvec.end(), {512, 1024});
  
  for (auto n : nvec) test<int8_t,  1>("[i08_1] kdtree::create", n, xlim);
  for (auto n : nvec) test<int16_t, 1>("[i16_1] kdtree::create", n, xlim);
  for (auto n : nvec) test<int32_t, 1>("[i32_1] kdtree::create", n, xlim);
  for (auto n : nvec) test<int64_t, 1>("[i64_1] kdtree::create", n, xlim);

  for (auto n : nvec) test<int8_t,  2>("[i08_2] kdtree::create", n, xlim);
  for (auto n : nvec) test<int16_t, 2>("[i16_2] kdtree::create", n, xlim);
  for (auto n : nvec) test<int32_t, 2>("[i32_2] kdtree::create", n, xlim);
  for (auto n : nvec) test<int64_t, 2>("[i64_2] kdtree::create", n, xlim);

  for (auto n : nvec) test<int8_t,  3>("[i08_3] kdtree::create", n, xlim);
  for (auto n : nvec) test<int16_t, 3>("[i16_3] kdtree::create", n, xlim);
  for (auto n : nvec) test<int32_t, 3>("[i32_3] kdtree::create", n, xlim);
  for (auto n : nvec) test<int64_t, 3>("[i64_3] kdtree::create", n, xlim);

  for (auto n : nvec) test<int8_t,  16>("[i08_16] kdtree::create", n, xlim);
  for (auto n : nvec) test<int16_t, 16>("[i16_16] kdtree::create", n, xlim);
  for (auto n : nvec) test<int32_t, 16>("[i32_16] kdtree::create", n, xlim);
  for (auto n : nvec) test<int64_t, 16>("[i64_16] kdtree::create", n, xlim);

}

