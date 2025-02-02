/*!
 * \file        kdtree.hpp
 * \author      Samridh D. Singh
 * \date        2025-02-01
 * \brief       Interface header file 
 * \details     
 *
 * \copyright   This file is part of the sycl_kdtree project.
 * \copyright   Copyright (C) 2025, Samridh D. Singh
 * \copyright   
 *              sycl_kdtree is free software: you can redistribute it and/or
 *              modify it under the terms of the GNU General Public License as
 *              published by the Free Software Foundation, either version 3 of
 *              the License, or (at your option) any later version.
 *
 *              sycl_kdtree is distributed in the hope that it will be useful,
 *              but WITHOUT ANY WARRANTY; without even the implied warranty of
 *              MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *              GNU General Public License for more details.
 *
 *              A copy of the GNU General Public License should be provided
 *              along with sycl_kdtree. 
 *              If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef KDTREE_HPP
#define KDTREE_HPP

#ifdef SYCL_LANGUAGE_VERSION
#define KDTREE__USE_SYCL 1
#include <CL/sycl.hpp>
#else 
#define KDTREE__USE_SYCL 0
#endif 

#include "create.hpp"
#include "nn.hpp"

#endif // KDTREE_HPP
