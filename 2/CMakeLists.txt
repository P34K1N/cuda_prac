cmake_minimum_required(VERSION 3.5)
project(main LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

set(SOURCE_FILES
        Image.cpp
        main.cu
        clock.cpp 
        memman.cu)

set (CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}")

include_directories(${ADDITIONAL_INCLUDE_DIRS})

add_executable(main ${SOURCE_FILES})

