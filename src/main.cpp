#include <kdtree.hpp>

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>

int 
main(void) {

#if 0

  using type_v = int;
  using type_s = int;
  using type_f = float;

  constexpr type_s dim = 2;

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

  kdtree::create<type_s, dim>(points, 10);

  std::vector<type_v> query = {11, 12};

  type_f rmax = std::numeric_limits<type_f>::max();
  auto idx = kdtree::nn<type_s, type_f, dim>(query, points, 10, rmax);
  
  /*
  std::cout << "Nearest neighbor index: " << idx << "\n";
  std::cout << "Nearest neighbor coordinates: ("
            << points[idx * dim] << ", "
            << points[idx * dim + 1] << ")\n";
            */


#else
  
  using type_v = float;
  using type_s = int32_t;

  constexpr type_s n   = 1 << 17;
  constexpr type_s dim = 3;
  constexpr auto   maj = kdtree::layout::rowmajor;

  std::vector<type_v> points(n * dim);
  
   {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<type_v> dist(0.0f, 10.0f);
    for (auto &v : points) v = dist(gen);
  }

  auto beg = std::chrono::steady_clock::now();
  kdtree::create<type_s, dim, maj>(points, n);
  auto end = std::chrono::steady_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg);
  std::cout << "KD-Tree creation time: " << dur.count() << " ms\n";

  constexpr std::size_t imax = 1e6;

  std::vector<type_v> q(dim);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<type_v> dist(0.0f, 10.0f);

  beg = std::chrono::steady_clock::now();

#if 0
  type_v rmax = std::numeric_limits<type_f>::max();
#else
  type_v rmax = 1e-9;
#endif

  std::vector<type_s> result(imax);
  for (int i = 0; i < imax; ++i) {
    for (type_s j = 0; j < dim; ++j) { q[j] = dist(gen); }
    result[i] = kdtree::nn<type_s, type_v, dim, maj>(q, points, n, rmax); 
  }
  end = std::chrono::steady_clock::now();
  dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg);

  std::cout << "Nearest neighbor queries (imax = " << imax << ") took: " 
            << dur.count() << " ms\n";

  { // saving results to ensure nn is being run.
    std::ofstream fp("results.txt");
    if (!fp) { std::cerr << "Error opening file for writing.\n"; return 1; }
    for (const auto &idx : result) { fp << idx << "\n"; }
    fp.close();
  }

 
#endif

}
