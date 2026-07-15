# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hongpei Li

cmake_minimum_required(VERSION 3.20)
option(ENABLE_MPI            "Enable distributed computing with MPI"  ON)
option(CARDAL_BUILD_PYTHON   "Build the CARDAL pybind11 module"       OFF)
option(CARDAL_BUILD_CLI      "Build the cardal CLI"                    ON)
option(CARDAL_BUILD_QUBO     "Build the QUBO auxiliary library + CLI" ON)

project(cardal LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(ZLIB REQUIRED)

set(CUDA_LIBS 
    CUDA::cudart 
    CUDA::cublas 
    CUDA::cusparse
    CUDA::curand
)

set(CORE_LINK_LIBS 
    ZLIB::ZLIB
)
option(ENABLE_MATIO "Enable .mat file parsing via libmatio" ON)

if(ENABLE_MATIO)
    find_path(MATIO_INCLUDE_DIR NAMES matio.h)
    find_library(MATIO_LIBRARY NAMES matio)

    if(MATIO_INCLUDE_DIR AND MATIO_LIBRARY)
        message(STATUS "Found system matio header: ${MATIO_INCLUDE_DIR}")
        message(STATUS "Found system matio library: ${MATIO_LIBRARY}")
        list(APPEND CORE_LINK_LIBS ${MATIO_LIBRARY})
        
        set(MATIO_EXTRA_DEFS USE_MATIO)
        set(MATIO_EXTRA_INCLUDES ${MATIO_INCLUDE_DIR})
    else()
        message(STATUS "System matio not found. Auto-downloading and building libmatio from source...")
        
        include(FetchContent)
        set(MATIO_SHARED OFF CACHE BOOL "Build static libmatio" FORCE)
        set(MATIO_MAT73 OFF CACHE BOOL "Disable MAT73" FORCE)
        set(MATIO_WITH_ZLIB ON CACHE BOOL "Enable ZLIB" FORCE)
        set(MATIO_PIC ON CACHE BOOL "Enable PIC" FORCE)
        
        FetchContent_Declare(
            matio
            GIT_REPOSITORY https://github.com/tbeu/matio.git
            GIT_TAG v1.5.23
            GIT_SHALLOW TRUE
            GIT_SUBMODULES ""
            GIT_SUBMODULES_RECURSE FALSE
        )
        FetchContent_MakeAvailable(matio)
        
        list(APPEND CORE_LINK_LIBS matio)
        
        set(MATIO_EXTRA_DEFS USE_MATIO)
        set(MATIO_EXTRA_INCLUDES ${matio_SOURCE_DIR}/src ${matio_BINARY_DIR})
    endif()
else()
    message(STATUS "matio is DISABLED.")
endif()

if(ENABLE_MPI)
    find_package(MPI REQUIRED)
    message(STATUS "MPI is ENABLED")
    list(APPEND CORE_LINK_LIBS MPI::MPI_C)
    
    list(APPEND CORE_LINK_LIBS MPI::MPI_CXX)
    
    find_library(NCCL_LIB NAMES nccl 
                 PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
                       $ENV{NCCL_ROOT}/lib
                       /usr/local/cuda/lib64
                       /usr/lib/x86_64-linux-gnu)

    if(NCCL_LIB)
        message(STATUS "Found NCCL: ${NCCL_LIB}")
        list(APPEND CORE_LINK_LIBS ${NCCL_LIB})
    else()
        message(WARNING "NCCL library not found explicitly, falling back to -lnccl")
        list(APPEND CORE_LINK_LIBS nccl)
    endif()
else()
    message(STATUS "MPI is DISABLED (Single Node Mode)")
endif()

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

list(FILTER SRC_FILES EXCLUDE REGEX "src/(main|cli)\\.c$")

# Distributed (MPI+NCCL) source files — exclude when MPI is off so the
# non-distributed build produces a lean, single-GPU-only library.
if(NOT ENABLE_MPI)
    list(FILTER SRC_FILES EXCLUDE REGEX "src/distribution_(solver|utils)\\.cu$")
endif()

add_library(sdp_core STATIC ${SRC_FILES})

target_link_libraries(sdp_core 
    PRIVATE ${CUDA_LIBS} 
    PUBLIC ${CORE_LINK_LIBS}
)

target_include_directories(sdp_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal>
    ${CUDAToolkit_INCLUDE_DIRS}
)

if(ENABLE_MPI)
    target_compile_definitions(sdp_core PUBLIC USE_MPI)
endif()

if(MATIO_EXTRA_DEFS)
    target_compile_definitions(sdp_core PUBLIC ${MATIO_EXTRA_DEFS})
endif()
if(MATIO_EXTRA_INCLUDES)
    target_include_directories(sdp_core PUBLIC ${MATIO_EXTRA_INCLUDES})
endif()

target_compile_options(sdp_core PRIVATE
    $<$<COMPILE_LANGUAGE:C,CXX>:-O3 -Wall -Wextra -fPIC>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-Wall,-Wextra,-fPIC>
)

set_target_properties(sdp_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

file(GLOB_RECURSE QUBO_SRC_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/qubo/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/qubo/*.cu"
)
list(FILTER QUBO_SRC_FILES EXCLUDE REGEX "qubo/qubo_cli\\.c$")

if(CARDAL_BUILD_QUBO AND QUBO_SRC_FILES)
    add_library(sdp_qubo STATIC ${QUBO_SRC_FILES})
    target_link_libraries(sdp_qubo
        PUBLIC sdp_core
        PRIVATE ${CUDA_LIBS}
    )
    target_include_directories(sdp_qubo PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/qubo>
    )
    target_compile_options(sdp_qubo PRIVATE
        $<$<COMPILE_LANGUAGE:C,CXX>:-O3 -Wall -Wextra -fPIC>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-Wall,-Wextra,-fPIC>
    )
    set_target_properties(sdp_qubo PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

if(CARDAL_BUILD_CLI AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/cli.c")
    add_executable(cardal src/cli.c)
    target_link_libraries(cardal PRIVATE sdp_core)
endif()

if(CARDAL_BUILD_QUBO AND TARGET sdp_qubo AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/qubo/qubo_cli.c")
    add_executable(cardal_qubo qubo/qubo_cli.c)
    target_link_libraries(cardal_qubo PRIVATE sdp_qubo)
endif()

if(CARDAL_BUILD_PYTHON)
    add_subdirectory(python_bindings)
endif()

if(TARGET cardal)
    install(TARGETS cardal RUNTIME DESTINATION bin)
endif()
if(TARGET cardal_qubo)
    install(TARGETS cardal_qubo RUNTIME DESTINATION bin)
endif()
install(TARGETS sdp_core ARCHIVE DESTINATION lib)
if(TARGET sdp_qubo)
    install(TARGETS sdp_qubo ARCHIVE DESTINATION lib)
endif()
install(DIRECTORY include/ DESTINATION include)
