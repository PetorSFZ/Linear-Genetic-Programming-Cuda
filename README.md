# Linear Genetic Programming Cuda

A simple curve-fitting program optimized with CUDA. This is something I threw together and is still early progress. The evaluation of the programs are done in CUDA, but most other stuff is done on the CPU in C++. I have not yet profiled or optimized the code, and I also knew of quite a lot of things that are done inefficiently because I'm lazy. So there should be quite a lot of room for improvement if I ever feel up to it.

# Building

Currently only tested on Windows, will hopefully build on other platforms as well.

* Have Visual Studio 2015 (or newer) installed
* Have CUDA 8 SDK (or newer) installed
* Have CMake installed
* Create a directory called `build` in the project root
* Inside the `build` directory run the following command: `cmake .. -G "Visual Studio 14 2015 Win64"`

# License

Currently not licensed under anything because it's not good enough for that. If you want to play around with it feel free, but please drop me a message if you plan to release it so I can license it under an appropriate license (for both us, likely zlib) first.
