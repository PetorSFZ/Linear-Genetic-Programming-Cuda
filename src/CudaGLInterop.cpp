// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "CudaGLInterop.hpp"

#include <algorithm>

#include <sfz/gl/IncludeOpenGL.hpp>
#include <cuda_gl_interop.h>

#include "CudaHelpers.hpp"

namespace phe {

// CudaGLTexture: Constructors & destructors
// ------------------------------------------------------------------------------------------------

CudaGLTexture::CudaGLTexture(vec2i resolution) noexcept
{
	this->mRes = resolution;

	// Creates OpenGL texture
	glGenTextures(1, &mGLTex);
	glBindTexture(GL_TEXTURE_2D, mGLTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mRes.x, mRes.y, 0, GL_RGBA, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	// https://github.com/nvpro-samples/gl_cuda_interop_pingpong_st
	CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&mCudaResource, mGLTex, GL_TEXTURE_2D,
	                 cudaGraphicsRegisterFlagsSurfaceLoadStore));
	CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &mCudaResource, 0));
	cudaArray_t cudaArray = 0;
	CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&cudaArray, mCudaResource, 0, 0));
	CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &mCudaResource, 0));

	// Create cuda surface object from binding
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(cudaResourceDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cudaArray;
	CHECK_CUDA_ERROR(cudaCreateSurfaceObject(&mCudaSurface, &resDesc));
}

CudaGLTexture::CudaGLTexture(CudaGLTexture&& other) noexcept
{
	std::swap(this->mRes, other.mRes);
	std::swap(this->mGLTex, other.mGLTex);
	std::swap(this->mCudaResource, other.mCudaResource);
	std::swap(this->mCudaSurface, other.mCudaSurface);
}

CudaGLTexture& CudaGLTexture::operator= (CudaGLTexture&& other) noexcept
{
	std::swap(this->mRes, other.mRes);
	std::swap(this->mGLTex, other.mGLTex);
	std::swap(this->mCudaResource, other.mCudaResource);
	std::swap(this->mCudaSurface, other.mCudaSurface);
	return *this;
}

CudaGLTexture::~CudaGLTexture() noexcept
{
	if (mCudaSurface != 0) {
		CHECK_CUDA_ERROR(cudaDestroySurfaceObject(mCudaSurface));
	}
	if (mCudaResource != 0) {
		CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(mCudaResource));
	}
	glDeleteTextures(1, &mGLTex);

	mRes = vec2i(0);
	mGLTex = 0;
	mCudaResource = 0;
	mCudaSurface = 0;
}

} // namespace phe
