cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pv-finder)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find conda environment
if(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
else()
    message(FATAL_ERROR "Conda environment not activated. Please activate pvfinderME environment.")
endif()

# Find PyTorch
execute_process(
    COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_PREFIX_PATH}")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Find Eigen
find_package(Eigen3 REQUIRED)

# Add executable
add_executable(pv-finder pv-finder.cpp)

# Link libraries
target_link_libraries(pv-finder ${TORCH_LIBRARIES} Eigen3::Eigen)

# Include directories
target_include_directories(pv-finder PRIVATE ${TORCH_INCLUDE_DIRS})
