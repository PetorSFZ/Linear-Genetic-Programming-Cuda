// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cuda_runtime.h>

#include <sfz/Assert.hpp>
#include <sfz/CudaCompatibility.hpp>

// Macros
// ------------------------------------------------------------------------------------------------

#define CHECK_CUDA_ERROR(error) (phe::checkCudaError(__FILE__, __LINE__, error))

namespace phe {

// Error checking
// ------------------------------------------------------------------------------------------------

inline cudaError_t checkCudaError(const char* file, int line, cudaError_t error) noexcept
{
	if (error == cudaSuccess) return error;
	sfz::printErrorMessage("%s:%i: CUDA error: %s\n", file, line, cudaGetErrorString(error));
	return error;
}

} // namespace phe
