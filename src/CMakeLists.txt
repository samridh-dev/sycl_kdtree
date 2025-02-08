set(SRC_FILES main.cpp)
add_executable(main ${SRC_FILES})
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/include)
