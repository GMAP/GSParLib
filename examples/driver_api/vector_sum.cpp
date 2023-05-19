#include <iostream>
#include <chrono>

#ifdef GSPARDRIVER_OPENCL

    #include "GSPar_OpenCL.hpp"
    using namespace GSPar::Driver::OpenCL;

#else

    #include "GSPar_CUDA.hpp"
    using namespace GSPar::Driver::CUDA;

#endif

const char* kernelSource = GSPAR_STRINGIZE_SOURCE(
    GSPAR_DEVICE_KERNEL void vector_sum_kernel(const int max,
            GSPAR_DEVICE_GLOBAL_MEMORY const unsigned int *a,
            GSPAR_DEVICE_GLOBAL_MEMORY const unsigned int *b,
            GSPAR_DEVICE_GLOBAL_MEMORY unsigned int *result) {
        size_t gid = gspar_get_global_id(0);
        if (gid <= max) {
            result[gid] = a[gid] + b[gid];
        }
    }
);

void vector_sum(const unsigned int max, const unsigned int* a, const unsigned int* b, unsigned int* result) {

    try {

        auto driver = Instance::getInstance();
        driver->init();

        int numGpus = driver->getGpuCount();
        if (numGpus == 0) {
            std::cout << "No GPU found, interrupting test" << std::endl;
            exit(-1);
        }

        // Get the first GPU
        auto gpu = driver->getGpu(0);
        
        // MemoryObject* a_dev = new MemoryObject(gpu, sizeof(unsigned int) * max, a);
        auto a_dev = gpu->malloc(sizeof(unsigned int) * max, a);
        auto b_dev = gpu->malloc(sizeof(unsigned int) * max, b);
        // Async copy
        a_dev->copyInAsync();
        b_dev->copyInAsync();
        AsyncExecutionSupport::waitAllAsync({ a_dev, b_dev });
        // Sync copy
        // a_dev->copyIn();
        // b_dev->copyIn();

        auto result_dev = gpu->malloc(sizeof(unsigned int) * max, result);

        // auto kernel = new Kernel(gpu, kernelSource, "vector_sum_kernel");
        auto kernel = gpu->prepareKernel(kernelSource, "vector_sum_kernel");

        // Set a fixed number of threads per block for the X dimension
        kernel->setNumThreadsPerBlockForX(5);
        kernel->setParameter(sizeof(max), &max);
        kernel->setParameter(a_dev);
        kernel->setParameter(b_dev);
        kernel->setParameter(result_dev);

        kernel->runAsync({max, 0});
        kernel->waitAsync();

        result_dev->copyOut();

        delete kernel;
        delete a_dev;
        delete b_dev;
        delete result_dev;

    } catch (GSPar::GSParException &ex) {
        std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
        exit(-1);
    }
}

void print_vector(unsigned int size, const unsigned int* vector, bool compact = false) {
    if (compact || size > 100) {
        std::cout << vector[0] << "..." << vector[size-1];
    } else {
        for (unsigned int i = 0; i < size; i++) {
            std::cout << vector[i] << " ";
        }
    }
    std::cout << std::endl;
}

int main(int argc, const char * argv[]) {
    if (argc < 2) {
        std::cerr << "Use: " << argv[0] << " <vector_size>" << std::endl;
        exit(-1);
    }

    const unsigned int VECTOR_SIZE = std::stoi(argv[1]);

    // Create memory objects
    unsigned int* result = new unsigned int[VECTOR_SIZE];
    unsigned int* a = new unsigned int[VECTOR_SIZE];
    unsigned int* b = new unsigned int[VECTOR_SIZE];
    for (unsigned int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = (unsigned int)i;
        b[i] = (unsigned int)i + 1;
        result[i] = 0;
    }

    std::cout << "Summing vectors:" << std::endl;
    std::cout << "Vector A: ";
    print_vector(VECTOR_SIZE, a);
    std::cout << "Vector B: ";
    print_vector(VECTOR_SIZE, b);

    auto t_start = std::chrono::steady_clock::now();

    vector_sum(VECTOR_SIZE, a, b, result);

    auto t_end = std::chrono::steady_clock::now();

    // Output the result buffer
    std::cout << "Result:   ";
    print_vector(VECTOR_SIZE, result);

    delete result;
    delete a;
    delete b;

    std::cout << "Test finished succesfully in " << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() << " ms " << std::endl;

    return 0;
}
