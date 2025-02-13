cmake_minimum_required(VERSION 3.10)

project(kdtree LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "compiler id: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "c++ standard: c++${CMAKE_CXX_STANDARD}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

include_directories(${CMAKE_SOURCE_DIR}/include)

option(BUILD_TESTS "Build tests" OFF)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")

  #  add_compile_options(-g -Wall -Wextra -Wpedantic -Werror -Wshadow -Wconversion -Wsign-conversion -Wuninitialized -Wmaybe-uninitialized) 

  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")

  add_compile_options(-Wall -Werror -Wfatal-errors)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -xHost -DNDEBUG")

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")

  add_compile_options(/W4 /WX)
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")

endif()


if (CMAKE_CXX_COMPILER MATCHES "icpx$")
  #  add_compile_options(-g -Wall -Wextra -Wpedantic -Werror -Wshadow -Wconversion -Wsign-conversion -Wuninitialized -Wmaybe-uninitialized) 
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 \
                               -march=native  \
                               -funroll-loops \
                               -g3 \
                               -gline-tables-only \
                               -fdebug-info-for-profiling \
                               -debug inline-debug-info \
                               -DNDEBUG")
  add_compile_options(-qopenmp -fsycl)
  add_link_options(-qopenmp -fsycl)
endif()

if (CMAKE_CXX_COMPILER MATCHES "acpp$")
  add_compile_options(-fopenmp -fPIC)
  add_link_options(-fopenmp -fPIC)
endif()

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
else()
  add_subdirectory(src)
endif()

