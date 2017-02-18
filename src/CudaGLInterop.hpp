// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <sfz/math/Vector.hpp>

namespace phe {

using sfz::vec2i;

// CudaGLTexture
// ------------------------------------------------------------------------------------------------

// OpenGL texture that is read/writeable in Cuda.
// Always float rgba
class CudaGLTexture final {
public:
	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	CudaGLTexture() noexcept = default;
	CudaGLTexture(const CudaGLTexture&) = delete;
	CudaGLTexture& operator= (const CudaGLTexture&) = delete;
	
	CudaGLTexture(vec2i resolution) noexcept;
	CudaGLTexture(CudaGLTexture&& other) noexcept;
	CudaGLTexture& operator= (CudaGLTexture&& other) noexcept;
	~CudaGLTexture() noexcept;

	// Getters
	// --------------------------------------------------------------------------------------------

	inline vec2i resolution() const noexcept { return mRes; }
	inline uint32_t glTexture() const noexcept { return mGLTex; }
	inline cudaSurfaceObject_t cudaSurface() const noexcept { return mCudaSurface; }

private:
	// Private members
	// --------------------------------------------------------------------------------------------

	vec2i mRes = vec2i(0);
	uint32_t mGLTex = 0;
	cudaGraphicsResource_t mCudaResource = 0;
	cudaSurfaceObject_t mCudaSurface = 0;
};

} // namespace phe
