add_library(pch INTERFACE)
target_precompile_headers(pch INTERFACE pch.h)

if(DEFINED TEST_ONLY)
  message(STATUS "Building only test: ${TEST_ONLY}.cpp")
  set(TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_ONLY}.cpp")
else()
  file(GLOB TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
endif()

unset(TEST_ONLY CACHE)

foreach(test_file ${TEST_FILES})
  get_filename_component(test_name ${test_file} NAME_WE)
  add_executable(${test_name} ${test_file})
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

