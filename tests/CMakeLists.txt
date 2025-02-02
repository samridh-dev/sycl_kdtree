file(GLOB TEST_SOURCES "*.cpp")

add_library(pch INTERFACE)
target_precompile_headers(pch INTERFACE pch.h)

foreach(TEST_FILE ${TEST_SOURCES})

  get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)

  set(TEST_EXEC test_${TEST_NAME})

  add_executable(${TEST_EXEC} ${TEST_FILE})

  target_link_libraries(${TEST_EXEC} PRIVATE pch)
  target_compile_options(${TEST_EXEC} PRIVATE -Wfatal-errors)

  add_test(NAME ${TEST_NAME} COMMAND ${TEST_EXEC})

endforeach()
