/*
 * Filename: kdtree_internal_clz.cpp
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

#define TEST_CLZ(TYPE) \
TEST_CASE("[" #TYPE "] kdtree::internal::clz") {                              \
CHECK_EQ(kdtree::internal::clz<TYPE>(0), std::numeric_limits<TYPE>::digits);  \
CHECK_EQ(kdtree::internal::clz<TYPE>(1), std::numeric_limits<TYPE>::digits-1);\
CHECK_EQ(kdtree::internal::clz<TYPE>(2), std::numeric_limits<TYPE>::digits-2);\
CHECK_EQ(kdtree::internal::clz<TYPE>(3), std::numeric_limits<TYPE>::digits-2);\
CHECK_EQ(kdtree::internal::clz<TYPE>(std::numeric_limits<TYPE>::max()), 0);   \
CHECK_EQ(kdtree::internal::clz<TYPE>(std::numeric_limits<TYPE>::max()>>1), 1);\
CHECK_EQ(kdtree::internal::clz<TYPE>(std::numeric_limits<TYPE>::max()>>2), 2);\
}

TEST_CLZ(std::uint8_t )
TEST_CLZ(std::uint16_t)
TEST_CLZ(std::uint32_t)
TEST_CLZ(std::uint64_t)
