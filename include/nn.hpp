/*!
 * \file        nn.hpp
 * \author      Samridh D. Singh
 * \date        2025-02-01
 * \brief       Type traits to enforce template parameter types.
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


#ifndef KDTREE_NN_HPP
#define KDTREE_NN_HPP

///////////////////////////////////////////////////////////////////////////////
/// HEADER                                                                  ///
///////////////////////////////////////////////////////////////////////////////

namespace kdtree {

template <typename Ts, typename Tf, Ts dim, 
          kdtree::layout maj = kdtree::layout::rowmajor,
          typename Cq, typename Cs>
Ts 
nn(const Cq& q, const Cs& src, const Ts n,
   Tf rmax = std::numeric_limits<Tf>::max());

} // namespace kdtree

///////////////////////////////////////////////////////////////////////////////
/// IMPLEMENTATION                                                          ///
///////////////////////////////////////////////////////////////////////////////

#include "container.hpp"
#include "internal.hpp"
#include "traverse.hpp"

namespace kdtree {
namespace internal {
namespace nn {

template <typename Ts, typename Tf>
struct result {
  Ts idx;
  Tf dst;
  result() : idx(0), dst(std::numeric_limits<Tf>::max()) {}
};

template <typename Ts, typename Tf, Ts dim, kdtree::layout maj,
          typename Cq, typename Cs>
struct process {
  void operator()(result<Ts, Tf>& res, const Cq& q, const Cs& src,
                  const Ts n, const Ts idx, Tf& r) const {

    using kdtree::internal::dist::euclidian;
    
    Tf dst = euclidian<Ts, Tf, dim, maj, Cq, Cs>(q, dim, src, n, 0, idx);

    if (dst < res.dst) {
      res.dst = dst;
      res.idx = idx;
      r       = dst;
    }

  }
};

}  // namespace nn
}  // namespace internal
}  // namespace kdtree

template <typename Ts, typename Tf, Ts dim, kdtree::layout maj,
          typename Cq, typename Cs>
Ts 
kdtree::nn(const Cq& q, const Cs& src, const Ts n, Tf rmax) {

  using kdtree::internal::nn::process;
  using kdtree::internal::nn::result;

  struct result<Ts, Tf> res;

  kdtree::traverse<result<Ts, Tf>, process<Ts, Tf, dim, maj, Cq, Cs>, 
                   Ts, Tf, dim, maj, Cq, Cs>
                  (res, q, src, n, rmax);

  return res.idx;
}

#endif // KDTREE_NN_HPP
