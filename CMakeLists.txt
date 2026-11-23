cmake_minimum_required(VERSION 3.20)
project(minicore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(minicore src/matrix.cpp src/main.cpp)
target_include_directories(minicore PRIVATE src)

if(MSVC)
    target_compile_options(minicore PRIVATE /O2 /arch:AVX2 /openmp /EHsc /utf-8 /W3)
    target_link_libraries(minicore PRIVATE psapi)
else()
    target_compile_options(minicore PRIVATE -O3 -mavx2 -mfma -Wall -Wextra)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(minicore PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
