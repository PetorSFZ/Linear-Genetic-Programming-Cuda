# Functions
# ------------------------------------------------------------------------------------------------

# Automatically set source group to correspond to directory structure.
# From Stack Overflow post by Florian:
# http://stackoverflow.com/a/31423421
function(set_directory_source_group)
	foreach(_source IN ITEMS ${ARGN})
		if (IS_ABSOLUTE "${_source}")
			file(RELATIVE_PATH _source_rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_source}")
		else()
			set(source_rel "${_source}")
		endif()
		get_filename_component(_source_path "${_source_rel}" PATH)
		string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
		source_group("${_source_path_msvc}" FILES "${_source}")
	endforeach()
endfunction(set_directory_source_group)

# Compiler flags & directories
# ------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project("LinearGeneticProgramming CUDA" CXX)

# Sets build type to release if no build type is specified in a single-configuration generator.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

# Require 64bit compiler
if(NOT (CMAKE_SIZEOF_VOID_P EQUAL 8))
	message(FATAL_ERROR "Can only compile in 64bit mode")
endif()

# Compiler flags
if(MSVC)
	# MSVC flags
	# /W4 = Warning level 4 (/Wall is too picky and has annoying warnings in standard headers)
	# /wd4201 = Disable warning 4201 (nonstandard extension used : nameless struct/union)
	# /Zi = Produce .pdb debug information. Does not affect optimizations, but does imply /debug.
	# /arch:AVX = Enable (require) Intel AVX instruction for code generation
	# /D_CRT_SECURE_NO_WARNINGS = Removes annyoing warning when using c standard library
	# /MD use MD_DynamicRelease runtime library
	set(CMAKE_CXX_FLAGS "/W4 /wd4201 /Zi /EHsc /arch:AVX /D_CRT_SECURE_NO_WARNINGS /MD")
	# /O2 = Optimize code for fastest speed
	# /fp:fast = "optimize floating-point code for speed at the expense of accuracy and correctness"
	# /DSFZ_NO_DEBUG = defines the "SFZ_NO_DEBUG" macro, which disables sfz_assert_debug()
	set(CMAKE_CXX_FLAGS_RELEASE "/O2 /fp:fast /DSFZ_NO_DEBUG")
	# /DEBUG = "creates debugging information for the .exe file or DLL"
	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/O2 /fp:fast /DEBUG /DSFZ_NO_DEBUG")
	# /Od = "disables optimization, speeding compilation and simplifying debugging"
	set(CMAKE_CXX_FLAGS_DEBUG "/Od /DEBUG")
else()
	# GCC / Clang flags
	set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++14 -march=sandybridge -pthread -fno-strict-aliasing")
	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -ffast-math -g -DSFZ_NO_DEBUG")
	set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -DSFZ_NO_DEBUG")
	set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
endif()

# Directories
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(EXTERNALS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/externals)

# Dependencies
# ------------------------------------------------------------------------------------------------

# sfzCore
set(SFZ_CORE_OPENGL TRUE)
add_subdirectory(${EXTERNALS_DIR}/sfzCore)

# CUDA
find_package(CUDA REQUIRED)

# CUDA compiler flags
# compute_30 <==> Equivalent to Kepler cards, (GTX 680M, etc)
# sm_50 <==> Maxwell cards (low-end), (940M, etc)
# sm_52 <==> Maxwell cards (high end), (GTX 970M, GTX 980 Ti, etc)
# sm_61 <==> Pascal cards (first gen), (1070, 1080, Titan X, etc)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -expt-relaxed-constexpr -gencode arch=compute_30,code=sm_52)
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} -O0 -g -lineinfo) # -G for extreme debugging
set(CUDA_NVCC_FLAGS_RELWITHDEBINFO ${CUDA_NVCC_FLAGS_RELWITHDEBINFO} -O3 -use_fast_math -g -lineinfo)
set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} -O3 -use_fast_math)

# Executable
# ------------------------------------------------------------------------------------------------

include_directories(
	${SOURCE_DIR}
	${SFZ_CORE_INCLUDE_DIRS}
)

set(LGP_CUDA_FILES
	${SOURCE_DIR}/CudaBuffer.hpp
	${SOURCE_DIR}/CudaGLInterop.hpp
	${SOURCE_DIR}/CudaGLInterop.cpp
	${SOURCE_DIR}/CudaHelpers.hpp
	${SOURCE_DIR}/LGP.cu
	${SOURCE_DIR}/ReferenceFunctionData.hpp
	${SOURCE_DIR}/ReferenceFunctionData.cpp
)
set_directory_source_group(${LGP_CUDA_FILES})

cuda_add_executable(LGP ${LGP_CUDA_FILES})
target_link_libraries(
	LGP
	${SFZ_CORE_LIBRARIES}
)

# Copy DLLs
# ------------------------------------------------------------------------------------------------

# Only copy DLLs if MSVC
if (MSVC)
	file(COPY ${SFZ_CORE_DLLS} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
	file(COPY ${SFZ_CORE_DLLS} DESTINATION ${CMAKE_BINARY_DIR}/RelWithDebInfo)
	file(COPY ${SFZ_CORE_DLLS} DESTINATION ${CMAKE_BINARY_DIR}/Release)
endif()
