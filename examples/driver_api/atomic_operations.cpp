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
    GSPAR_DEVICE_KERNEL void atomicops_kernel(const int max,
            GSPAR_DEVICE_GLOBAL_MEMORY const int *vector,
            GSPAR_DEVICE_GLOBAL_MEMORY int *result) {
        size_t gid = gspar_get_global_id(0);
        if (gid <= max) {
            gspar_atomic_add_int(result, vector[gid]);
        }
    }
);

void print_vector(int size, const int* vector, bool compact = false) {
    if (compact || size > 100) {
        std::cout << vector[0] << "..." << vector[size-1];
    } else {
        for (int i = 0; i < size; i++) {
            std::cout << vector[i] << " ";
        }
    }
    std::cout << std::endl;
}

int main(int argc, const char * argv[]) {

    std::cout << "Testing atomic operations in GSParLib Driver API" << std::endl;

    const int VECTOR_SIZE = 20;

    // Create memory objects
    int correctResult = 0;
    int* result = new int;
    int* vector = new int[VECTOR_SIZE];
    for (int i = 0; i < VECTOR_SIZE; i++) {
        vector[i] = (int)i;
        correctResult += i;
    }

    std::cout << "Vector with " << VECTOR_SIZE << " elements:" << std::endl;
    print_vector(VECTOR_SIZE, vector);

    try {

        auto t_start = std::chrono::steady_clock::now();

        auto driver = Instance::getInstance();
        driver->init();

        int numGpus = driver->getGpuCount();
        if (numGpus == 0) {
            std::cout << "No GPU found, interrupting test" << std::endl;
            exit(-1);
        }

        // Get the first GPU
        auto gpu = driver->getGpu(0);
        
        auto vector_dev = gpu->malloc(sizeof(int) * VECTOR_SIZE, vector);
        // Async copy
        // vector_dev->copyInAsync();
        // vector_dev->waitAsync();
        // Sync copy
        vector_dev->copyIn();

        auto result_dev = gpu->malloc(sizeof(int), result);

        auto kernel = gpu->prepareKernel(kernelSource, "atomicops_kernel");
        // auto kernel = new Kernel(gpu, kernelSource, "atomicops_kernel");

        // Set a fixed number of threads per block for the X dimension
        // kernel->setNumThreadsPerBlockForX(5);
        kernel->setParameter(sizeof(VECTOR_SIZE), &VECTOR_SIZE);
        kernel->setParameter(vector_dev);
        kernel->setParameter(result_dev);

        kernel->runAsync({VECTOR_SIZE, 0});
        kernel->waitAsync();

        result_dev->copyOut();

        delete kernel;
        delete vector_dev;
        delete result_dev;

        auto t_end = std::chrono::steady_clock::now();

        // Output the result buffer
        std::cout << "Expected result: " << correctResult << std::endl;
        std::cout << "Actual result:   " << *result << std::endl;

        delete vector;
        delete result;

        std::cout << "Test finished succesfully in " << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() << " ms " << std::endl;

        return 0;

    } catch (GSPar::GSParException &ex) {
        std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
        exit(-1);
    }

}
