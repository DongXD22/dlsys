cmake_minimum_required(VERSION 3.18)
project(needle LANGUAGES C CXX CUDA)
cmake_policy(SET CMP0146 OLD)

# 生成 compile_commands.json，供 VS Code IntelliSense 使用
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find correct version of Python
execute_process(COMMAND python3-config --prefix
  OUTPUT_VARIABLE Python_ROOT_DIR)
find_package(Python COMPONENTS Development Interpreter REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# find pybind
execute_process(COMMAND python3 -m pybind11 --cmakedir
  RESULT_VARIABLE __pybind_exit_code
  OUTPUT_VARIABLE __pybind_path
  OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 PATHS ${__pybind_path})


if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
else()
  set(CMAKE_CXX_FLAGS "/std:c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
endif()

include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${pybind11_LIBRARIES})


###################
### CPU BACKEND ###
###################
add_library(ndarray_backend_cpu MODULE src/ndarray_backend_cpu.cc)
target_link_libraries(ndarray_backend_cpu PUBLIC ${LINKER_LIBS})
pybind11_extension(ndarray_backend_cpu)
if(NOT CMAKE_BUILD_TYPE MATCHES Debug)
  pybind11_strip(ndarray_backend_cpu)
endif()


# directly output to ffi folder
set_target_properties(ndarray_backend_cpu
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray
  CXX_VISIBILITY_PRESET "hidden"
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set_property(TARGET ndarray_backend_cpu PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
endif()



####################
### CUDA BACKEND ###
####################
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  message(STATUS "Found cuda, building cuda backend")

  # 自动检测 GPU 架构
  execute_process(COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE GPU_COMPUTE_CAP
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(GPU_COMPUTE_CAP)
    string(REPLACE "." "" CUDA_ARCH "${GPU_COMPUTE_CAP}")
    message(STATUS "Detected CUDA architecture: ${GPU_COMPUTE_CAP}")
  else()
    set(CUDA_ARCH "52")
    message(STATUS "Could not detect GPU, defaulting to sm_52")
  endif()
  set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})

  add_library(ndarray_backend_cuda MODULE src/ndarray_backend_cuda.cu)
  target_link_libraries(ndarray_backend_cuda ${LINKER_LIBS})
  pybind11_extension(ndarray_backend_cuda)
  if(NOT CMAKE_BUILD_TYPE MATCHES Debug)
    pybind11_strip(ndarray_backend_cuda)
  endif()

  # directly output to ffi folder
  set_target_properties(ndarray_backend_cuda
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden"
  )

endif()
