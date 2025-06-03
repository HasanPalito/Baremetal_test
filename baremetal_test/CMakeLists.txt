cmake_minimum_required(VERSION 3.10)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g")

project(Baremetal_test)

include_directories(${CMAKE_SOURCE_DIR}/../include) 

link_directories(${CMAKE_SOURCE_DIR}/../build/lib) 
link_directories(${CMAKE_SOURCE_DIR}/../build/src)   

find_package(OpenMP)

if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "No OpenMP support found")
endif()

# Set Intel MKL include and lib paths (modify according to your setup)
set(MKL_INCLUDE_PATH "/usr/include/mkl")
set(MKL_LIB_PATH "/usr/lib/x86_64-linux-gnu")

include_directories(${MKL_INCLUDE_PATH})
link_directories(${MKL_LIB_PATH})

# Specify MKL libraries to link with
set(MKL_LIBRARIES
    mkl_intel_ilp64
    mkl_intel_thread
    mkl_core
    iomp5
    pthread
    m
    dl
)

# Find Boost package (required by DiskANN)
find_package(Boost REQUIRED COMPONENTS program_options)

if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Boost not found")
endif()

# Specify the test source file (main.cpp)
add_executable(main ${CMAKE_SOURCE_DIR}/main.cpp)

# Link against the necessary libraries (DiskANN, Boost, MKL, OpenMP, etc.)
target_link_libraries(main
    diskann               # Link with the local diskann library
    Boost::program_options
    ${MKL_LIBRARIES}    
    OpenMP::OpenMP_CXX
    aio    # Ensure OpenMP is linked correctly
)

add_executable(bench ${CMAKE_SOURCE_DIR}/bench.cpp)

# Link against the necessary libraries (DiskANN, Boost, MKL, OpenMP, etc.)
target_link_libraries(bench
    diskann               # Link with the local diskann library
    Boost::program_options
    ${MKL_LIBRARIES}    
    OpenMP::OpenMP_CXX
    aio    # Ensure OpenMP is linked correctly
)
