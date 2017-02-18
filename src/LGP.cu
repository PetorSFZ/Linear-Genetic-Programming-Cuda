#include <random>

#ifdef _WIN32
#include <direct.h>
#endif

#include <SDL.h>

#include <sfz/gl/Context.hpp>
#include <sfz/gl/FullscreenQuad.hpp>
#include <sfz/gl/GLUtils.hpp>
#include <sfz/gl/IncludeOpenGL.hpp>
#include <sfz/gl/Program.hpp>
#include <sfz/math/MathSupport.hpp>
#include <sfz/sdl/Session.hpp>
#include <sfz/sdl/Window.hpp>
#include <sfz/strings/DynString.hpp>
#include <sfz/strings/StackString.hpp>
#include <sfz/util/IO.hpp>

#include "CudaBuffer.hpp"
#include "CudaGLInterop.hpp"
#include "CudaHelpers.hpp"
#include "ReferenceFunctionData.hpp"

using namespace sfz;
using std::mt19937_64;

// Compile-time parameters
// ------------------------------------------------------------------------------------------------

constexpr uint32_t MAX_NUM_REGISTERS_AND_CONSTANTS = 16; // Hardcoded (4 bits), don't change
constexpr uint32_t MAX_NUM_OPS = 32; //128;
constexpr uint32_t NUM_OP_TYPES = 4; // +, -, *, /

// Structs
// ------------------------------------------------------------------------------------------------

struct Operation final {
	uint16_t data; // 4 bits per thing
	SFZ_CUDA_CALL void set(uint8_t type, uint8_t dst, uint8_t src1, uint8_t src2)
	{
		data = type;
		data |= (dst << 4u);
		data |= (src1 << 8u);
		data |= (src2 << 12u);
	}
	SFZ_CUDA_CALL uint8_t type() const noexcept { return data & 0x000Fu; }
	SFZ_CUDA_CALL uint8_t dst() const noexcept { return (data >> 4u) & 0x000Fu; }
	SFZ_CUDA_CALL uint8_t src1() const noexcept { return (data >> 8u) & 0x000Fu; }
	SFZ_CUDA_CALL uint8_t src2() const noexcept { return (data >> 12u) & 0x000Fu; }
};
static_assert(sizeof(Operation) == 2, "Operation is padded");

struct Chromosome final {
	uint32_t numOps;
	Operation ops[MAX_NUM_OPS];
};

// Host helper functions
// ------------------------------------------------------------------------------------------------

static DynArray<Chromosome> initChromosomes(mt19937_64& rngGen,
                                            uint32_t numChromosomes, uint32_t numInitialOps,
                                            uint32_t numRegisters, uint32_t numConstants,
                                            uint32_t numOpTypes) noexcept
{
	// RNG distributions
	std::uniform_int_distribution<uint32_t> opTypeDistr(0, numOpTypes - 1);
	std::uniform_int_distribution<uint32_t> dstDistr(0, numRegisters - 1);
	std::uniform_int_distribution<uint32_t> srcDistr(0, numRegisters + numConstants - 1);

	// Allocate CPU memory for chromosomes
	DynArray<Chromosome> tmp(numChromosomes);
	tmp.setSize(numChromosomes);

	// Create random chromosomes
	for (uint32_t iChromo = 0; iChromo < numChromosomes; iChromo++) {
		tmp[iChromo].numOps = numInitialOps;
		for (uint32_t iOp = 0; iOp < numInitialOps; iOp++) {
			tmp[iChromo].ops[iOp].set(opTypeDistr(rngGen), dstDistr(rngGen),
			                          srcDistr(rngGen), srcDistr(rngGen));
		}
	}

	return tmp;
}

static void mutateChromosomes(mt19937_64& rngGen,
                              Chromosome* chromosomes, uint32_t numChromosomes,
                              uint32_t numRegisters, uint32_t numConstants,
                              uint32_t numOpTypes, float mutationProbability) noexcept
{
	// RNG distributions
	std::uniform_int_distribution<uint32_t> opTypeDistr(0, numOpTypes - 1);
	std::uniform_int_distribution<uint32_t> dstDistr(0, numRegisters - 1);
	std::uniform_int_distribution<uint32_t> srcDistr(0, numRegisters + numConstants - 1);
	std::uniform_real_distribution<float> mutationDistr(0.0f, 1.0f);

	// Mutate chromosomes
	for (uint32_t iChromo = 0; iChromo < numChromosomes; iChromo++) {
		Chromosome& c = chromosomes[iChromo];

		for (uint32_t iOp = 0; iOp < c.numOps; iOp++) {
			Operation& op = c.ops[iOp];
			uint8_t opType = op.type();
			uint8_t dst = op.dst();
			uint8_t src1 = op.src1();
			uint8_t src2 = op.src2();

			if (mutationDistr(rngGen) > mutationProbability) {
				opType = opTypeDistr(rngGen);
			}
			if (mutationDistr(rngGen) > mutationProbability) {
				dst = dstDistr(rngGen);
			}
			if (mutationDistr(rngGen) > mutationProbability) {
				src1 = srcDistr(rngGen);
			}
			if (mutationDistr(rngGen) > mutationProbability) {
				src2 = srcDistr(rngGen);
			}

			op.set(opType, dst, src1, src2);
		}
	}
}

static uint32_t tournamentSelection(const float* fitness, uint32_t i1, uint32_t i2, float r, float tournamentSelectionParam) noexcept
{
	float fitness1 = fitness[i1];
	float fitness2 = fitness[i2];

	uint32_t chosen;
	if (r < tournamentSelectionParam) {
		if (fitness1 > fitness2) chosen = i1;
		else chosen = i2;
	}
	else {
		if (fitness1 > fitness2) chosen = i2;
		else chosen = i1;
	}

	return chosen;
}

static void crossChromosomes(mt19937_64& rngGen, Chromosome& c1, Chromosome& c2) noexcept
{
	std::uniform_int_distribution<uint32_t> c1NumOpsDistr(0, c1.numOps - 1);
	std::uniform_int_distribution<uint32_t> c2NumOpsDistr(0, c2.numOps - 1);

	uint32_t c1First = c1NumOpsDistr(rngGen);
	uint32_t c1Last = c1NumOpsDistr(rngGen);
	if (c1First > c1Last) std::swap(c1First, c1Last);
	uint32_t c1Len = c1Last - c1First + 1;
	
	uint32_t c2First = c2NumOpsDistr(rngGen);
	uint32_t c2Last = c2NumOpsDistr(rngGen);
	if (c2First > c2Last) std::swap(c2First, c2Last);
	uint32_t c2Len = c2Last - c2First + 1;

	Chromosome tmp1;
	tmp1.numOps = 0;
	for (uint32_t i = 0; i < MAX_NUM_OPS; i++) {
		if (i < c1First) {
			tmp1.ops[i] = c1.ops[i];
		}
		else if ((i - c1First) < c2Len) {
			tmp1.ops[i] = c2.ops[c2First + (i - c1First)];
		}
		else if ((i - c1First - c2Len) < (c1.numOps - c1Last)) {
			tmp1.ops[i] = c1.ops[c1Last + 1 + (i - c1First - c2Len)];
		}
		else {
			break;
		}
		tmp1.numOps = i + 1;
	}

	Chromosome tmp2;
	tmp2.numOps = 0;
	for (uint32_t i = 0; i < MAX_NUM_OPS; i++) {
		if (i < c2First) {
			tmp2.ops[i] = c2.ops[i];
		}
		else if ((i - c2First) < c1Len) {
			tmp2.ops[i] = c1.ops[c1First + (i - c2First)];
		}
		else if ((i - c2First - c1Len) < (c2.numOps - c2Last)) {
			tmp2.ops[i] = c2.ops[c2Last + 1 + (i - c2First - c1Len)];
		}
		else {
			break;
		}
		tmp2.numOps = i + 1;
	}

	c1 = tmp1;
	c2 = tmp2;
}

static void crossoverSelection(mt19937_64& rngGen, Chromosome* chromosomes, const Chromosome* original,
                               const float* fitness, uint32_t numChromosomes,
                               float tournamentSelectionParam, float crossoverProbability) noexcept
{
	// RNG distributions
	std::uniform_int_distribution<uint32_t> indexDistr(0, numChromosomes - 1);
	std::uniform_real_distribution<float> floatDistr(0.0f, 1.0f);

	for (uint32_t iChromo = 0; iChromo < numChromosomes; iChromo += 2) {
		
		// Perform tournament selection
		uint32_t selected1 = tournamentSelection(fitness, indexDistr(rngGen), indexDistr(rngGen),
		                                         floatDistr(rngGen), tournamentSelectionParam);
		uint32_t selected2 = tournamentSelection(fitness, indexDistr(rngGen), indexDistr(rngGen),
		                                         floatDistr(rngGen), tournamentSelectionParam);
		
		// Maybe perform crossover
		if (floatDistr(rngGen) < crossoverProbability) {
			chromosomes[iChromo] = original[selected1];
			chromosomes[iChromo + 1] = original[selected2];
			crossChromosomes(rngGen, chromosomes[iChromo], chromosomes[iChromo + 1]);
		}
		else {
			chromosomes[iChromo] = original[selected1];
			chromosomes[iChromo + 1] = original[selected2];
		}
	}
}

static StackString256 createSymbolicRepresentation(const Chromosome& c, uint32_t numRegisters,
                                              uint32_t numConstants, const float* constants) noexcept
{
	// Create and initialize registers
	StackString256 registers[MAX_NUM_REGISTERS_AND_CONSTANTS];
	registers[0].printf("x");
	for (uint32_t i = 1; i < numRegisters; i++) {
		registers[i].printf("0");
	}
	for (uint32_t i = 0; i < numConstants; i++) {
		registers[i + numRegisters].printf("%.1f", constants[i]);
	}

	// Run program
	StackString256 tmp;
	const uint32_t numOps = c.numOps;
	for (uint32_t i = 0; i < numOps; i++) {
		Operation op = c.ops[i];
		switch (op.type()) {
		case 0: { // +
			tmp.printf("(%s + %s)", registers[op.src1()], registers[op.src2()]);
			} break;
		case 1: { // -
			tmp.printf("(%s - %s)", registers[op.src1()], registers[op.src2()]);
			} break;
		case 2: { // *
			tmp.printf("(%s * %s)", registers[op.src1()], registers[op.src2()]);
			} break;
		case 3: { // /
			tmp.printf("(%s / %s)", registers[op.src1()], registers[op.src2()]);
			} break;
		}
		registers[op.dst()].printf("%s", tmp.str);
	}

	return registers[0];
}

// Cuda kernel help functions
// ------------------------------------------------------------------------------------------------

static __device__ void writeSurface(cudaSurfaceObject_t surface, vec2u coord, vec4 value) noexcept
{
	float4 tmp;
	tmp.x = value.x;
	tmp.y = value.y;
	tmp.z = value.z;
	tmp.w = value.w;
	surf2Dwrite(tmp, surface, coord.x * sizeof(float4), coord.y);
}

// Shared kernel globals
// ------------------------------------------------------------------------------------------------

// __managed__ to access from both host and device (but not at same time obviously)
//static __managed__ __device__ float bestFitness = 0.0f;

// Cuda Kernels
// ------------------------------------------------------------------------------------------------

static __global__ void evaluateChromosomes(const Chromosome* __restrict__ chromosomes,
                                           uint32_t numChromosomes, uint32_t numRegisters,
                                           uint32_t numConstants, const float* __restrict__ constants,
                                           uint32_t numPoints, const vec2* __restrict__ refPoints,
                                           vec2* __restrict__ globalPoints,
                                           float* __restrict__ globalErrors)
{
	const uint32_t chromosomeIdx = blockIdx.x; // Each block represents one chromosome
	const uint32_t refPointIdx = threadIdx.x;
	const uint32_t globalPointsIdx = chromosomeIdx * numPoints + refPointIdx;
	if (refPointIdx >= numPoints) return;

	const Chromosome& chromosome = chromosomes[chromosomeIdx];
	vec2 refPoint = refPoints[refPointIdx];
	const float x = refPoint.x;

	// Create and initialize registers
	float registers[MAX_NUM_REGISTERS_AND_CONSTANTS]; 
	registers[0] = x;
	for (uint32_t i = 1; i < numRegisters; i++) {
		registers[i] = 0.0f;
	}
	for (uint32_t i = 0; i < numConstants; i++) {
		registers[i + numRegisters] = constants[i];
	}
	
	// Run program
	const uint32_t numOps = chromosome.numOps;
	for (uint32_t i = 0; i < numOps; i++) {
		Operation op = chromosome.ops[i];
		switch (op.type()) {
		case 0: { // +
			registers[op.dst()] = registers[op.src1()] + registers[op.src2()];
			} break;
		case 1: { // -
			registers[op.dst()] = registers[op.src1()] - registers[op.src2()];
			} break;
		case 2: { // *
			registers[op.dst()] = registers[op.src1()] * registers[op.src2()];
			} break;
		case 3: { // /
			registers[op.dst()] = registers[op.src1()] / registers[op.src2()];
			} break;
		}
	}
	const float y = registers[0];

	// Calculate error
	float error = (y - refPoint.y);
	error *= error;

	// Write global point with evaluated answer and error
	globalPoints[globalPointsIdx] = vec2(x, y);
	globalErrors[globalPointsIdx] = error;
}

static __global__ void stupidFitnessReduce(uint32_t numPoints, uint32_t numChromsomes,
                                           const float* __restrict__ globalErrors,
                                           float* __restrict__ fitness)
{
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= (numPoints * numChromsomes)) return;
	const uint32_t baseIdx = idx * numPoints;

	float errorSum = 0.0f;
	for (uint32_t i = 0; i < numPoints; i++) {
		errorSum += globalErrors[baseIdx + i];
	}

	fitness[idx] = 1.0f / (sqrt(errorSum / float(numPoints)));
}

static __global__ void plotCurve(cudaSurfaceObject_t surface, vec2u res,
                                 vec2 graphMin, vec2 graphMax, uint32_t numPoints,
                                 const vec2* __restrict__ refPoints, 
                                 const vec2* __restrict__ bestPoints)
{
	// Calculate surface coordinates
	vec2u loc = vec2u(blockIdx.x * blockDim.x + threadIdx.x,
	                  blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= res.x || loc.y >= res.y) return;

	// Calculate x and y coordinates
	vec2 coord = ((vec2(loc) * (1.0f / vec2(res))) * (graphMax - graphMin)) + graphMin;

	// Initalize color to background
	vec4 color = vec4(1.0f);

	// Set color to black if close to axis lines
	if (sfz::abs(coord.x) < 0.00075f) color = vec4(0.0f);
	if (sfz::abs(coord.y) < 0.00075f) color = vec4(0.0f);

	// Go through reference points and set color
	for (uint32_t i = 0; i < numPoints; i++) {
		vec2 refPoint = refPoints[i];
		float squaredDist = dot(coord - refPoint, coord - refPoint);
		if (squaredDist < 0.00075f) {
			float scale = (squaredDist * (1.0f / 0.00075f));
			color = sfz::lerp(vec4(1.0f, 0.0f, 0.0f, 1.0f), color, scale);
		}
	}

	// Go through best points and set color
	for (uint32_t i = 0; i < numPoints; i++) {
		vec2 bestPoint = bestPoints[i];
		float squaredDist = dot(coord - bestPoint, coord - bestPoint);
		if (squaredDist < 0.00050f) {
			float scale = (squaredDist * (1.0f / 0.00050f));
			color = sfz::lerp(vec4(0.0f, 0.0f, 1.0f, 1.0f), color, scale);
		}
	}

	// Write color to surface
	writeSurface(surface, loc, color);
}

// Main entry point
// ------------------------------------------------------------------------------------------------

int main(int arg, char** argv)
{
	// Boring SDL + OpenGL init stuff
	// --------------------------------------------------------------------------------------------

#ifdef _WIN32
	// Enable hi-dpi awareness
	SetProcessDPIAware();

	// Set current working directory to SDL_BasePath()
	_chdir(sfz::basePath());
#endif

	// SDL
	sdl::Session sdlSession = sdl::Session({sdl::SDLInitFlags::EVENTS, sdl::SDLInitFlags::VIDEO});
	sdl::Window window("LGP CUDA", 1280, 720, {sdl::WindowFlags::ALLOW_HIGHDPI, sdl::WindowFlags::OPENGL, sdl::WindowFlags::RESIZABLE});

	// OpenGL
	gl::Context glContext(window.ptr(), 4, 5, gl::GLContextProfile::COMPATIBILITY,
#ifdef SFZ_NO_DEBUG
	false
#else
	true
#endif
	);
	glewExperimental = GL_TRUE;
	GLenum glewError = glewInit();
	if (glewError != GLEW_OK) sfz::error("GLEW init failure: %s", glewGetErrorString(glewError));
	gl::printSystemGLInfo();

	// Create OpenGL/CUDA interop texture
	phe::CudaGLTexture glCudaTex(window.drawableDimensions());

	// GL stuff
	gl::FullscreenQuad quad;
	gl::Program shader = gl::Program::postProcessFromSource(
R"(
	#version 450

	in vec2 uvCoord;
	layout(location = 0) uniform sampler2D uTexture;
	out vec4 outFragColor;
	
	void main()
	{
		outFragColor = texture(uTexture, uvCoord);
	}
)");

	// LGP Cuda stuff starts here
	// --------------------------------------------------------------------------------------------

	// Run-time parameters (could be loaded from file in future version)
	const uint32_t numIterations = 10000000;
	const uint32_t numChromosomes = 60;
	const uint32_t numInitialOps = 10;
	const uint32_t numRegisters = 4;
	DynArray<float> constantsCpu;
	constantsCpu.add(1.0f);
	constantsCpu.add(-1.0f);
	constantsCpu.add(2.0f);
	const CudaBuffer<float> constants(constantsCpu);
	sfz_assert_release((numRegisters + constantsCpu.size()) <= MAX_NUM_REGISTERS_AND_CONSTANTS);
	const float tournamentSelectionParam = 0.8f;
	const float crossoverProbability = 0.2f;
	const float mutationProbability = 0.025f;
	const uint32_t elitismNumBestPreviousInsertions = 2;

	// RNG generator

	std::random_device rd;
	mt19937_64 rngGen(rd());

	// Load reference data
	const CudaBuffer<vec2> refData = referenceFunctionData();
	const uint32_t numPoints = refData.capacity();

	// Initialize chromosomes
	DynArray<Chromosome> chromosomesCpu = initChromosomes(rngGen, numChromosomes, numInitialOps,
	                                                      numRegisters, constants.capacity(),
	                                                      NUM_OP_TYPES);
	CudaBuffer<Chromosome> chromosomes(chromosomesCpu);

	// Compact array with evaluated points
	// Chromosome 'i' starts at data() + i * numPoints
	CudaBuffer<vec2> evaluatedPoints(numPoints * numChromosomes);
	CudaBuffer<float> pointErrors(numPoints * numChromosomes); // The error for each point

	// Initialize fitness to 0 for each chromosome
	DynArray<float> fitnessCpu;
	fitnessCpu.addMany(numChromosomes, 0.0f);
	CudaBuffer<float> fitness(fitnessCpu);

	Chromosome bestChromosome;
	float bestFitness = 0.0f;
	CudaBuffer<vec2> bestChromosomePoints(numPoints);
	bool bestChromosomeUpdated = true;
	DynArray<Chromosome> chromosomesCpuCopy = chromosomesCpu;

	for (uint32_t iItr = 0; iItr < numIterations; iItr++) {
		
		// Evaluate chromsomes
		evaluateChromosomes<<<chromosomes.capacity(), numPoints>>>(chromosomes.data(), chromosomes.capacity(),
		                                                           numRegisters, constants.capacity(),
		                                                           constants.data(), numPoints,
		                                                           refData.data(), evaluatedPoints.data(),
		                                                           pointErrors.data());
		CHECK_CUDA_ERROR(cudaGetLastError());
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());

		// Calculate fitness
		stupidFitnessReduce<<<(chromosomes.capacity()/32) + 1, 32>>>(numPoints, chromosomes.capacity(),
		                                                             pointErrors.data(), fitness.data());
		CHECK_CUDA_ERROR(cudaGetLastError());
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());

		// Download fitness to CPU
		fitness.download(fitnessCpu);

		// Find best chromosome (and fitness)
		for (uint32_t i = 0; i < fitnessCpu.size(); i++) {
			float f = fitnessCpu[i];
			if (f > bestFitness) {
				bestFitness = f;
				bestChromosome = chromosomes.downloadElement(i);
				evaluatedPoints.copyTo(bestChromosomePoints, 0, numPoints * i, numPoints);
				bestChromosomeUpdated = true;
			}
		}

		// Download chromosomes back to cpu and copy them
		chromosomes.download(chromosomesCpu);
		chromosomesCpuCopy = chromosomesCpu;

		// Selection & crossover
		crossoverSelection(rngGen, chromosomesCpu.data(), chromosomesCpuCopy.data(), fitnessCpu.data(),
		                   numChromosomes, tournamentSelectionParam, crossoverProbability);

		// Mutation
		mutateChromosomes(rngGen, chromosomesCpu.data(), numChromosomes, numRegisters, constants.capacity(),
		                  NUM_OP_TYPES, mutationProbability);

		// Elitism
		for (uint32_t i = 0; i < elitismNumBestPreviousInsertions; i++) {
			chromosomesCpu[i] = bestChromosome;
		}

		// Upload chromosomes to gpu
		chromosomes.upload(chromosomesCpu);

		// Render graph if better chromosome found
		if (bestChromosomeUpdated) {

			auto symbolicStr = createSymbolicRepresentation(bestChromosome, numRegisters,
			                                                constantsCpu.size(), constantsCpu.data());

			printf("Iteration %u, fitness %.3f, numOps %u\nf(x) = %s\n\n", iItr, bestFitness, bestChromosome.numOps, symbolicStr.str);

			// Plot reference points
			vec2u res = vec2u(glCudaTex.resolution());
			dim3 threadsPerBlock(8, 8);
			dim3 numBlocks((res.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
						   (res.y + threadsPerBlock.y - 1) / threadsPerBlock.y);
			plotCurve<<<numBlocks, threadsPerBlock>>>(glCudaTex.cudaSurface(), res,
			                                          vec2(-5.5f, -1.5f), vec2(5.5f, 1.5f), numPoints,
			                                          refData.data(), bestChromosomePoints.data());
			CHECK_CUDA_ERROR(cudaGetLastError());
			CHECK_CUDA_ERROR(cudaDeviceSynchronize());

			// Render glCudaFB to window fb
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			glViewport(0, 0, window.drawableWidth(), window.drawableHeight());
			glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			shader.useProgram();
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, glCudaTex.glTexture());
			quad.render();
			SDL_GL_SwapWindow(window.ptr());

			bestChromosomeUpdated = false;
		}
	}

	system("pause");
	return 0;
}