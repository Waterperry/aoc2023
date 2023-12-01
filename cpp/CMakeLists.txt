cmake_minimum_required(VERSION 3.26)
project(aoc23 CUDA)

set(CMAKE_CUDA_STANDARD 17)


add_executable(aoc23 main.cu
        src/day1.cu
)

target_include_directories(aoc23 PUBLIC lib)
set_target_properties(aoc23 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
