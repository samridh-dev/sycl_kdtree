#include <iostream>
#include <random>
#include <chrono>
#include <CL/sycl.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>


#include <kdtree.hpp>
#include <omp.h>

using T_v = float;
using T_s = int;

constexpr auto maj  { kdtree::container::layout::row_major   };
constexpr T_s dim   { 3                                      };
constexpr T_s n     { 1 << 27                                };

constexpr T_s block_size  { 512 };
constexpr T_s global_size { ((n + block_size - 1) / block_size) * block_size };

int
main(void) {

  sycl::queue queue;
  {
    sycl::device device = queue.get_device();
    std::cout << "Queue Information:\n";
    std::cout << "  Device Name       : " << device.get_info<sycl::info::device::name>() << "\n";
    std::cout << "  Vendor            : " << device.get_info<sycl::info::device::vendor>() << "\n";
    std::cout << "  Device Type       : ";
    switch (device.get_info<sycl::info::device::device_type>()) {
        case sycl::info::device_type::cpu: std::cout << "CPU"; break;
        case sycl::info::device_type::gpu: std::cout << "GPU"; break;
        case sycl::info::device_type::accelerator: std::cout << "Accelerator"; break;
        default: std::cout << "Unknown";
    }
    std::cout << "\n";
    std::cout << "  Max Compute Units : " << device.get_info<sycl::info::device::max_compute_units>() << "\n";
    std::cout << "  Global Memory     : " 
              << device.get_info<sycl::info::device::global_mem_size>() / (1024 * 1024) << " MB\n";
    std::cout << "  Local Memory      : " 
              << device.get_info<sycl::info::device::local_mem_size>() / 1024 << " KB\n";
    std::cout << "  Max Work Group Size: " 
              << device.get_info<sycl::info::device::max_work_group_size>() << "\n";
  }

  std::cout << "Metadata:\n";
  std::cout << "n          : " << n   << "\n";
  std::cout << "dim        : " << dim << "\n";
  std::cout << "layout     : " << maj << "\n";
  std::cout << std::endl;

  std::vector<T_v>  vec(dim * n, 0.0f);
  std::vector<T_s>  vidx(n, 0);

  kdtree::context ctx;
  
  if (std::filesystem::exists("kdtree_n"+std::to_string(dim * n)+".dat")) {

    std::ifstream ifs("kdtree_n" + std::to_string(dim * n) + ".dat",
                      std::ios::binary);
    if (!ifs) {
      std::cerr << "Error opening kdtree.dat for reading.\n";
      exit(EXIT_FAILURE);
    }

    ifs.read(reinterpret_cast<char*>(vec.data()), vec.size() * sizeof(T_v));
    if (!ifs) {
      std::cerr << "Error reading vector data from kdtree.dat.\n";
      exit(EXIT_FAILURE);
    }

    std::cout << "kd-tree loaded from file.\n";

  } else {

    std::mt19937 rng(std::random_device{}());
    #if 1
      constexpr T_v v_min{0e0};
      constexpr T_v v_max{1e0};
      std::uniform_real_distribution<T_v> dist(v_min, v_max);
    #else
      constexpr T_v mean   {0.5f};
      constexpr T_v stddev {0.1f};
      std::normal_distribution<T_v> dist(mean, stddev);
    #endif
    for (auto &v : vec) v = dist(rng);

    auto beg { std::chrono::high_resolution_clock::now() };
    kdtree::create<T_s, dim, maj>(ctx, vec, n);
    auto end { std::chrono::high_resolution_clock::now() };
    auto dur { std::chrono::duration_cast<std::chrono::milliseconds>(end - beg) };
    std::cout << "[kdtree::create]: " << dur.count() << " ms\n";

    std::ofstream ofs("kdtree_n" + std::to_string(dim * n) + ".dat",
                      std::ios::binary);
    if (!ofs) {
      std::cerr << "Error opening kdtree.dat for writing.\n";
      exit(EXIT_FAILURE);
    }
    const size_t size { vec.size() };
    ofs.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(T_v));
    std::cout << "kd-tree saved to file.\n";

  }

  T_s* usm__vidx { sycl::malloc_device<T_s>(n,     queue) };
  T_v* usm__vec  { sycl::malloc_device<T_v>(dim*n, queue) };

  queue.memcpy(usm__vidx, vidx.data(),  n    * sizeof(T_s));
  queue.memcpy(usm__vec,  vec.data(),   dim*n* sizeof(T_v));

  auto beg { std::chrono::high_resolution_clock::now() };

  queue.parallel_for(sycl::nd_range<1>(global_size, block_size), 
                     [=](sycl::nd_item<1> item) {

    const auto i { item.get_global_id(0) };

    if (i >= n) {
      return; 
    }

    T_v q[dim];
    for (T_s j{0}; j < dim; ++j) {
      q[j] = kdtree::container::id<T_s, dim, maj>(usm__vec, n, i, j);
    }

    usm__vidx[i] = kdtree::nn<float, T_s, dim, maj>(ctx, q, usm__vec, n);

  });

  queue.wait_and_throw();
  auto end { std::chrono::high_resolution_clock::now() };
  auto dur { std::chrono::duration_cast<std::chrono::milliseconds>(end - beg) };

  std::cout << "[kdtree::nn][time]:\t" 
            << dur.count() << " ms\n";

  std::cout << "[kdtree::nn][throughput]:\t" 
            << static_cast<int>((n / (dur.count() * 1e-3) * 1e-6)) << "M\n";

  queue.memcpy(vidx.data(), usm__vidx, n * sizeof(T_s)).wait();

  std::cout << vidx.back() << std::endl;

  sycl::free(usm__vidx, queue);
  sycl::free(usm__vec,  queue);


}
