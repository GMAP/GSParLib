# GSParLib: A Multi-Level Programming Interface Unifying OpenCL and CUDA for Expressing Stream and Data Parallelism

GSParLib is a C++ object-oriented multi-level API unifying OpenCL and CUDA for GPU programming that allows code portability between different GPU platforms and targets stream and data parallelism.

Contributors and role:
- Dinei A. Rockenbach [[ORCID](https://orcid.org/0000-0002-2091-9626)]: architecture design, interface definition, and development.
- Gabriell Alves de Araujo [[ORCID](https://orcid.org/0000-0001-8179-2318)]: development of new features and most optimizations.
- Dalvan Griebler [[ORCID](https://orcid.org/0000-0002-4690-3964)]: coordination, funds acquisition, and project management.

## How to cite
Rockenbach, D. A., Araujo, G., Griebler, D., & Fernandes, L. G. (2025). **GSParLib: A multi-level programming interface unifying OpenCL and CUDA for expressing stream and data parallelism**. *Computer Standards & Interfaces*, 92, 103922. https://doi.org/10.1016/j.csi.2024.103922

```bibtex
@article{ROCKENBACH:CSI:25,
	title={{GSParLib}: A multi-level programming interface unifying {OpenCL} and {CUDA} for expressing stream and data parallelism},
	author={Dinei A. Rockenbach and Gabriell Araujo and Dalvan Griebler and Luiz Gustavo Fernandes},
	journal={Computer Standards \& Interfaces},
	volume={92},
	pages={103922},
	publisher={Elsevier},
	month=mar,
	year={2025},
	doi={10.1016/j.csi.2024.103922},
	abstract={The evolution of Graphics Processing Units (GPUs) has allowed the industry to overcome long-lasting problems and challenges. Many belong to the stream processing domain, whose central aspect is continuously receiving and processing data from streaming data producers such as cameras and sensors. Nonetheless, programming GPUs is challenging because it requires deep knowledge of many-core programming, mechanisms and optimizations for GPUs. Current GPU programming standards do not target stream processing and present programmability and code portability limitations. Among our main scientific contributions resides GSParLib, a C++ multi-level programming interface unifying CUDA and OpenCL for GPU processing on stream and data parallelism with negligible performance losses compared to manual implementations; GSParLib is organized in two layers: one for general-purpose computing and another for high-level structured programming based on parallel patterns; a methodology to provide unified and driver agnostic interfaces minimizing performance losses; a set of parallelism strategies and optimizations for GPU processing targeting stream and data parallelism; and new experiments covering GPU performance on applications exposing stream and data parallelism.},
}
```

## Usage

A simple Vector Sum function using GSParLib Pattern API is shown below. Please refer to the [examples](examples/) folder to find complete examples using both Driver API and Pattern API, along with sequential versions of the algorithms for comparison.

```c++
float* vector_sum(const float max, const float* a, const float* b) {
  float* result = new float[max];
  try {
    // Create a new Map object and defined the
    // function to be executed for each data item
    auto pattern = new GSPar::Pattern::Map(GSPAR_STRINGIZE_SOURCE(
        result[x] = a[x] + b[x];
    ));
    // Sets the Map parameters, which will be translated
    // by GSParLib into GPU Kernel parameters
    pattern->setParameter("a", sizeof(float) * max, a)
        .setParameter("b", sizeof(float) * max, b)
        .setParameter("result", sizeof(float) * max, result, GSPAR_PARAM_OUT);
    // Runs the pattern synchronously on the GPU
    // using GSPar::Driver::CUDA or GSPar::Driver::OpenCL
    pattern->run<GSPar::Driver::CUDA::Instance>({max, 0});
    // Since the "result" is marked as an output parameter in setParameter,
    // it is automatically copied from the GPU memory after running the kernel.
    return result;
  } catch (GSPar::GSParException &ex) {
    // Any errors occurring during the execution will trigger an exception
    std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
    exit(-1);
  }
}
```

## Compilation

- `make` builds the library
- `make examples` builds the examples for both Pattern and Driver API, as well as the sequential versions. To compile just a specific set of examples, use one of:
  - `make examples_driver_api`
  - `make examples_pattern_api`
  - `make examples_sequential`
- Alternatively, it is possible to compile individual examples by referring directly to their compiled names (the `cuda`/`opencl` suffix may be ommited). Ex.: `make bin/ex_driverapi_gpuinfo` compiles both CUDA and OpenCL versions of the [gpuinfo.cpp](examples/driver_api/gpuinfo.cpp) example.

To compile with debugging enabled, use `DEBUG=1 make` (both when compiling the library and the examples). This enables debugging code paths, so that GSParLib prints various debugging information during execution.

## Run examples

After building the library it is necessary to make it available at runtime.
To do this, execute `export LD_LIBRARY_PATH=<path>/bin:$LD_LIBRARY_PATH`, replacing `<path>` with the path to the repository's root folder. If you are on the right folder, just execute `export LD_LIBRARY_PATH=$(pwd)/bin:$LD_LIBRARY_PATH`.

After this, just execute any example under `bin/ex_`

## Documentation

Detailed documentation of the library is available at the [Wiki](https://github.com/GMAP/GSParLib/wiki).
