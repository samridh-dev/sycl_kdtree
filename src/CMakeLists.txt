add_executable(main main.cpp)
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(sycl_nn sycl_nn.cpp)
target_include_directories(sycl_nn PRIVATE ${CMAKE_SOURCE_DIR}/include)

