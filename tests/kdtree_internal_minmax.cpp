/*
 * Filename: kdtree_internal_minmax.cpp
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
#include "internal/minmax.hpp"

using namespace kdtree::internal;

TEST_CASE("[min][type=int]") {
  CHECK(min( 3,  5) == 3);
  CHECK(min(-2, -5) == -5);
  CHECK(min( 7,  7) == 7);
  CHECK(min(-1,  0) == -1);
}

TEST_CASE("[min][type=float]") {
  CHECK(min( 2.5f,  3.14f) == doctest::Approx(2.5f));
  CHECK(min(-3.14f, 2.0f)  == doctest::Approx(-3.14f));
  CHECK(min( 0.5f,   0.5f) == doctest::Approx(0.5f));
}

TEST_CASE("[min][type=double]") {
  CHECK(min( 2.71828, 3.14159) == doctest::Approx(2.71828));
  CHECK(min(-1.2345, 0.0)      == doctest::Approx(-1.2345));
  CHECK(min( 5.0,     5.0)     == doctest::Approx(5.0));
}

TEST_CASE("[max][type=int]") {
  CHECK(max( 3,  5) == 5);
  CHECK(max(-2, -5) == -2);
  CHECK(max( 7,  7) == 7);
  CHECK(max(-1,  0) == 0);
}

TEST_CASE("[max][type=float]") {
  CHECK(max( 2.5f,  3.14f) == doctest::Approx(3.14f));
  CHECK(max(-3.14f, 2.0f)  == doctest::Approx(2.0f));
  CHECK(max( 0.5f,   0.5f) == doctest::Approx(0.5f));
}

TEST_CASE("[max][type=double]") {
  CHECK(max( 2.71828, 3.14159) == doctest::Approx(3.14159));
  CHECK(max(-1.2345, 0.0)      == doctest::Approx(0.0));
  CHECK(max( 5.0,     5.0)     == doctest::Approx(5.0));
}
