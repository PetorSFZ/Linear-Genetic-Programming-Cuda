#pragma once

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "cuda_runtime.h"

#include <sfz/Assert.hpp>
#include <sfz/containers/DynArray.hpp>

#include "CudaHelpers.hpp"

namespace sfz {

using std::uint32_t;
using std::uint64_t;

// CudaBuffer
// ------------------------------------------------------------------------------------------------

template<typename T>
class CudaBuffer final {
public:
	static_assert(std::is_trivial<T>::value, "T is not a trivial type");

	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	CudaBuffer() noexcept = default;
	CudaBuffer(const CudaBuffer&) = delete;
	CudaBuffer& operator= (const CudaBuffer&) = delete;

	explicit CudaBuffer(uint32_t capacity) noexcept;
	explicit CudaBuffer(const DynArray<T>& dynArray) noexcept;
	CudaBuffer(const T* dataPtr, uint32_t numElements) noexcept;
	
	CudaBuffer(CudaBuffer&& other) noexcept;
	CudaBuffer& operator= (CudaBuffer&& other) noexcept;
	~CudaBuffer() noexcept;

	// Methods
	// --------------------------------------------------------------------------------------------

	void create(uint32_t capacity) noexcept;
	void destroy() noexcept;
	void swap(CudaBuffer& other) noexcept;

	void upload(const T* srcPtr, uint32_t dstLocation, uint32_t numElements) noexcept;
	void upload(const DynArray<T>& src) noexcept;
	void download(T* dstPtr, uint32_t dstLocation, uint32_t numElements) noexcept;
	void download(T* dstPtr) noexcept;
	void download(DynArray<T>& dst) noexcept;

	void uploadElement(const T& element, uint32_t dstLocation) noexcept;
	void downloadElement(T& element, uint32_t dstLocation) noexcept;
	T downloadElement(uint32_t dstLocation) noexcept;

	void copyTo(CudaBuffer& dstBuffer, uint32_t dstLocation, uint32_t srcLocation,
	            uint32_t numElements) noexcept;
	void copyTo(CudaBuffer& dstBuffer, uint32_t numElements) noexcept;
	void copyTo(CudaBuffer& dstBuffer) noexcept;

	// Getters
	// --------------------------------------------------------------------------------------------

	inline T* data() noexcept { return mDataPtr; }
	inline const T* data() const noexcept { return mDataPtr; }
	inline uint32_t capacity() const noexcept { return mCapacity; }
	
private:
	// Private members
	// --------------------------------------------------------------------------------------------

	T* mDataPtr = nullptr;
	uint32_t mCapacity = 0u;
};

// CudaBuffer implementation: Constructors & destructors
// ------------------------------------------------------------------------------------------------

template<typename T>
CudaBuffer<T>::CudaBuffer(uint32_t capacity) noexcept
{
	this->create(capacity);
}

template<typename T>
CudaBuffer<T>::CudaBuffer(const DynArray<T>& dynArray) noexcept
{
	this->create(dynArray.size());
	this->upload(dynArray.data(), 0u, dynArray.size());
}

template<typename T>
CudaBuffer<T>::CudaBuffer(const T* srcPtr, uint32_t numElements) noexcept
{
	this->create(numElements);
	this->upload(srcPtr, 0u, numElements);
}

template<typename T>
CudaBuffer<T>::CudaBuffer(CudaBuffer&& other) noexcept
{
	this->swap(other);
}

template<typename T>
CudaBuffer<T>& CudaBuffer<T>::operator= (CudaBuffer&& other) noexcept
{
	this->swap(other);
	return *this;
}

template<typename T>
CudaBuffer<T>::~CudaBuffer() noexcept
{
	this->destroy();
}

// CudaBuffer implementation: Methods
// ------------------------------------------------------------------------------------------------

template<typename T>
void CudaBuffer<T>::create(uint32_t capacity) noexcept
{
	if (mCapacity != 0u) this->destroy();
	uint64_t numBytes = capacity * sizeof(T);
	CHECK_CUDA_ERROR(cudaMalloc(&mDataPtr, numBytes));
	mCapacity = capacity;
}

template<typename T>
void CudaBuffer<T>::destroy() noexcept
{
	CHECK_CUDA_ERROR(cudaFree(mDataPtr));
	mDataPtr = nullptr;
	mCapacity = 0u;
}

template<typename T>
void CudaBuffer<T>::swap(CudaBuffer& other) noexcept
{
	std::swap(this->mDataPtr, other.mDataPtr);
	std::swap(this->mCapacity, other.mCapacity);
}

template<typename T>
void CudaBuffer<T>::upload(const T* dataPtr, uint32_t dstLocation, uint32_t numElements) noexcept
{
	sfz_assert_debug((dstLocation + numElements) <= mCapacity);
	uint64_t numBytes = numElements * sizeof(T);
	CHECK_CUDA_ERROR(cudaMemcpy(mDataPtr + dstLocation, dataPtr, numBytes, cudaMemcpyHostToDevice));
}

template<typename T>
void CudaBuffer<T>::upload(const DynArray<T>& src) noexcept
{
	this->upload(src.data(), 0u, src.size());
}

template<typename T>
void CudaBuffer<T>::download(T* dstPtr, uint32_t dstLocation, uint32_t numElements) noexcept
{
	sfz_assert_debug((dstLocation + numElements) <= mCapacity);
	uint64_t numBytes = numElements * sizeof(T);
	CHECK_CUDA_ERROR(cudaMemcpy(dstPtr, mDataPtr + dstLocation, numBytes, cudaMemcpyDeviceToHost));
}

template<typename T>
void CudaBuffer<T>::download(T* dstPtr) noexcept
{
	this->download(dstPtr, 0u, mCapacity);
}

template<typename T>
void CudaBuffer<T>::download(DynArray<T>& dst) noexcept
{
	dst.ensureCapacity(mCapacity);
	dst.clear();
	this->download(dst.data(), 0u, mCapacity);
	dst.setSize(mCapacity);
}

template<typename T>
void CudaBuffer<T>::uploadElement(const T& element, uint32_t dstLocation) noexcept
{
	this->upload(&element, dstLocation, 1u);
}

template<typename T>
void CudaBuffer<T>::downloadElement(T& element, uint32_t dstLocation) noexcept
{
	this->download(&element, dstLocation, 1u);
}

template<typename T>
T CudaBuffer<T>::downloadElement(uint32_t dstLocation) noexcept
{
	T tmp;
	this->downloadElement(tmp, dstLocation);
	return tmp;
}

template<typename T>
void CudaBuffer<T>::copyTo(CudaBuffer& dstBuffer, uint32_t dstLocation, uint32_t srcLocation,
                           uint32_t numElements) noexcept
{
	sfz_assert_debug(dstBuffer.capacity() >= (dstLocation + numElements));
	uint64_t numBytes = numElements * sizeof(T);
	CHECK_CUDA_ERROR(cudaMemcpy(dstBuffer.mDataPtr + dstLocation, this->mDataPtr + srcLocation,
	                 numBytes, cudaMemcpyDeviceToDevice));
}

template<typename T>
void CudaBuffer<T>::copyTo(CudaBuffer& dstBuffer, uint32_t numElements) noexcept
{
	this->copyTo(dstBuffer, 0u, 0u, numElements);
}

template<typename T>
void CudaBuffer<T>::copyTo(CudaBuffer& dstBuffer) noexcept
{
	this->copyTo(dstBuffer, mCapacity);
}

} // namespace sfz
