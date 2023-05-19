#include <iostream>
#include <chrono>

#ifdef GSPARDRIVER_OPENCL
    #include "GSPar_OpenCL.hpp"
    using namespace GSPar::Driver::OpenCL;
#else
    #include "GSPar_CUDA.hpp"
    using namespace GSPar::Driver::CUDA;
#endif

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

    std::cout << "Testing shared memory in GSParLib Driver API" << std::endl;

    std::string kernelSource = ""
    "GSPAR_DEVICE_KERNEL void sharedmem_kernel(const int max, \n"
    "    GSPAR_DEVICE_GLOBAL_MEMORY const unsigned int *vector, \n"
    "    GSPAR_DEVICE_GLOBAL_MEMORY unsigned int *result";
    #ifdef GSPARDRIVER_OPENCL // OpenCL requires declaring shared memory after all the parameters
        kernelSource += ", GSPAR_DEVICE_SHARED_MEMORY unsigned int* sharedMem) { \n";
    #else // CUDA requires declaring shared memory inside kernel's body
        kernelSource += ") { \n GSPAR_DEVICE_SHARED_MEMORY unsigned int sharedMem[];\n";
    #endif
    kernelSource += 
    "    size_t gid = gspar_get_global_id(0); \n"
    "    if (gid <= max) { \n"
    "        sharedMem[gid] = vector[gid]; \n"
    "    } \n"
    "    gspar_synchronize_local_threads(); \n"
    "    if (gid == 0) { \n"
    "        for (size_t i = 0; i < max; i++) { \n"
    "            *result += sharedMem[i]; \n"
    "        } \n"
    "    } \n"
    "} \n";

    const unsigned int VECTOR_SIZE = 20;

    // Create memory objects
    unsigned int correctResult = 0;
    unsigned int* result = new unsigned int;
    unsigned int* vector = new unsigned int[VECTOR_SIZE];
    for (unsigned int i = 0; i < VECTOR_SIZE; i++) {
        vector[i] = (unsigned int)i;
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
        
        auto vector_dev = gpu->malloc(sizeof(unsigned int) * VECTOR_SIZE, vector);
        // Async copy
        // vector_dev->copyInAsync();
        // vector_dev->waitAsync();
        // Sync copy
        vector_dev->copyIn();

        auto result_dev = gpu->malloc(sizeof(unsigned int), result);

        auto kernel = gpu->prepareKernel(kernelSource, "sharedmem_kernel");
        // auto kernel = new Kernel(gpu, kernelSource, "sharedmem_kernel");

        kernel->setSharedMemoryAllocation(sizeof(unsigned int) * VECTOR_SIZE);

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
