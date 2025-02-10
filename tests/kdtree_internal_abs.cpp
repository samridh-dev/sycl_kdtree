/*
 * Filename: kdtree_internal_abs.cpp
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
#include "internal/abs.hpp"

using namespace kdtree::internal;

TEST_CASE("[abs][type=int]") {
  CHECK(abs(  5) == 5);
  CHECK(abs( -5) == 5);
  CHECK(abs(  0) == 0);
}

TEST_CASE("[abs][type=float]") {
  CHECK(abs(  3.14f) == doctest::Approx(3.14f));
  CHECK(abs( -3.14f) == doctest::Approx(3.14f));
  CHECK(abs(  0.0f)  == doctest::Approx(0.0f));
}

TEST_CASE("[abs][type=double]") {
  CHECK(abs(  2.718281828) == doctest::Approx(2.718281828));
  CHECK(abs( -2.718281828) == doctest::Approx(2.718281828));
  CHECK(abs(  0.0)         == doctest::Approx(0.0));
}
