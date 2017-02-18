#pragma once

#include <sfz/math/Vector.hpp>

#include "CudaBuffer.hpp"

using sfz::vec2;
using sfz::CudaBuffer;

CudaBuffer<vec2> referenceFunctionData() noexcept;
