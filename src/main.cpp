#include <iostream>
#include <random>
#include <chrono>

#include <fstream>

#include <kdtree.hpp>
#include <omp.h>

volatile int sink = 0;

int
main(void) {

#if 1

  using type_v = int;
  using type_s = int;
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

  std::cout << "input : ";
  for (auto v : vec) std::cout << v << " ";
  std::cout << std::endl;

  kdtree::create<type_s, dim>(ctx, vec, n);

  std::cout << "tree : ";
  for (auto v : vec) std::cout << v << " ";
  std::cout << std::endl;

  std::vector<type_v> q{ 1, 1 };

  const auto idx = kdtree::nn<type_f, type_s, dim>(ctx, q, vec, n);

  std::cout << "Nearest neighbor index: " << idx << "\n";
  std::cout << "Nearest neighbor coordinates: ("
            << vec[static_cast<std::size_t>(idx) * dim + 0] << ", "
            << vec[static_cast<std::size_t>(idx) * dim + 1] << ")\n";

  return 0;

#else

  using type_v = float;
  using type_s = int;
  
  constexpr auto   maj  {kdtree::container::layout::row_major};
  constexpr type_s dim  {3};
  constexpr type_s n    {1 << 18};
  constexpr type_s k    {64};

  kdtree::context ctx;

  std::vector<type_v> vec(dim * n);
  std::vector<type_v> q(dim, 0);

  std::cout << "[metadata] "
            << "dim : " << dim << ", "
            << "n : "   << n   << ", "
            << std::endl;

  {
    std::mt19937 rng(std::random_device{}());
    #if 1
      constexpr type_v v_min{0e0};
      constexpr type_v v_max{1e0};
      std::uniform_real_distribution<type_v> dist(v_min, v_max);
    #else
      constexpr type_v mean   {0.5f};
      constexpr type_v stddev {0.1f};
      std::normal_distribution<type_v> dist(mean, stddev);
    #endif
    for (auto& v : vec) { v = dist(rng); }
  }

  { 
    auto beg{std::chrono::high_resolution_clock::now()};
    kdtree::create<type_s, dim, maj>(ctx, vec, n);
    auto end{std::chrono::high_resolution_clock::now()};
    auto dur{std::chrono::duration_cast<std::chrono::milliseconds>(end - beg)};
    std::cout << "[kdtree::create]: " << dur.count() << " ms" << std::endl;
  }

  const type_s imax{n};
  const float  rmax{std::numeric_limits<float>::max()};

  #if 1
  {

    std::ofstream fp("out.dat");

    auto beg{std::chrono::high_resolution_clock::now()};
    for (type_s i = 0; i < imax; ++i) {

      {
        constexpr type_v v_min{0e0};
        constexpr type_v v_max{1e0};
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<type_v> dist(v_min, v_max);
        for (auto& qi : q) { qi = dist(rng); }
      }

      auto idx = kdtree::knn<float, int, dim, maj>(ctx, q, vec, n, k);

      fp<<i<<"\t:\t"; for (const auto& idxi: idx) fp<<idxi<<"\t"; fp<<"\n";

    }

    auto end{std::chrono::high_resolution_clock::now()};
    auto dur{std::chrono::duration_cast<std::chrono::milliseconds>(end - beg)};
    std::cout << "[kdtree::knn]: " << dur.count() << " ms" << std::endl;

  }
  #endif


  {

    std::vector<type_s>    vidx(n, 0);

    auto beg{std::chrono::high_resolution_clock::now()};
    #pragma omp parallel for
    for (type_s i = 0; i < n; ++i) {

      for (type_s j = 0; j < dim; ++j) {
        q[j] = kdtree::container::id<type_s, dim, maj>(vec, n, i, j);
      }

      vidx[i] = kdtree::nn<float, type_s, dim, maj>(ctx, q, vec, n, rmax);
    }

      for (type_s j = 0; j < dim; ++j) {
        std::cout << kdtree::container::id<type_s, dim, maj>(vec, n, 4, j) << " ";
      } std::cout << std::endl;
      for (type_s j = 0; j < dim; ++j) {
        std::cout << kdtree::container::id<type_s, dim, maj>(vec, n, vidx[4], j) << " ";
      } std::cout << std::endl;


    auto end{std::chrono::high_resolution_clock::now()};
    auto dur{std::chrono::duration_cast<std::chrono::milliseconds>(end - beg)};
    std::cout << "[kdtree::nn]: " << dur.count() << " ms" << std::endl;

  }

#endif

}
