# GSParLib

GSParLib is a C++ object-oriented multi-level API for GPU programming that allows code portability between different GPU platforms and targets stream and data parallelism.

The scientific article presenting GSParLib is currently under review.

## Compilation

- `make` builds the library
- `make examples` builds the examples for both Pattern and Driver API, as well as the sequential versions. To compile just a specific set of examples, use one of:
  - `make examples_driver_api`
  - `make examples_pattern_api`
  - `make examples_sequential`
- Alternatively, it is possible to compile individual examples by referring directly to their compiled names (the `cuda`/`opencl` suffix may be ommited). Ex.: `make bin/ex_driverapi_gpuinfo` compiles both CUDA and OpenCL versions of the [gpuinfo.cpp](examples/driver_api/gpuinfo.cpp) example.

To compile with debugging enabled, use `DEBUG=1 make` (both when compiling the library and the examples). This automatically enables debugging flags, so that GSParLib print various debugging information during execution.

## Run examples

After building the library it is necessary to make it available at runtime.
To do this, execute `export LD_LIBRARY_PATH=<path>/bin:$LD_LIBRARY_PATH`, replacing `<path>` with the path to the repository's root folder.

After this, just execute any example under the path `bin/ex_`
