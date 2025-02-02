set(SRC_FILES main.cpp)
add_executable(main ${SRC_FILES})
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/include)

message(STATUS "[compiler] ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "[mode] ${CMAKE_BUILD_TYPE}")

target_compile_options(main PRIVATE -Wfatal-errors)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(main PRIVATE
      -O0 -g -ggdb3 -pg -fno-omit-frame-pointer --coverage -fPIC
      -Wall -Wextra -Werror -Wpedantic -Wshadow -Wconversion -Wsign-conversion 
      -Wnull-dereference -Wdouble-promotion -Wformat=2 -Wstrict-aliasing=2
      -Wcast-align=strict -Woverloaded-virtual -Wduplicated-branches 
      -Wduplicated-cond -Wlogical-op -Wuseless-cast
    )
    target_link_options(main PRIVATE --coverage -fprofile-arcs -ftest-coverage)
  elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(main PRIVATE
      -O3 -march=native -flto -fno-plt -fomit-frame-pointer -DNDEBUG -fPIC
    )
    target_link_options(main PRIVATE -flto -fuse-linker-plugin)
  endif()

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(main PRIVATE
      -O0 -g -pg -fno-omit-frame-pointer -fsanitize=undefined,address -fPIC
      -Wall -Wextra -Werror -Wpedantic -Wshadow -Wconversion -Wsign-conversion
      -Wnull-dereference -Wdouble-promotion -Wformat=2 -Wcomma
      -Wcast-align=strict -Woverloaded-virtual -Wstrict-aliasing=2
      -Wduplicated-branches -Wduplicated-cond -Wlogical-op -Wuseless-cast
    )
    target_link_options(main PRIVATE -fsanitize=undefined,address)
  elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(main PRIVATE
      -O3 -march=native -flto -fno-plt -fomit-frame-pointer -DNDEBUG -fPIC
    )
    target_link_options(main PRIVATE -flto -fuse-ld=lld)
  endif()

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(main PRIVATE
      /Od /DEBUG /Zi /GL /Gy /permissive- /w14242 /w14254 /w14263
      /w14265 /w14287 /we4289 /w14296 /w14311 /w14545 /w14546
      /w14547 /w14549 /w14555 /w14619 /w14640 /w14826 /w14905
      /w14906 /w14928 /W4
    )
  elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(main PRIVATE
      /O2 /GL /Gy /fp:fast /arch:AVX2 /DNDEBUG
    )
    target_link_options(main PRIVATE /LTCG)
  endif()
endif()
