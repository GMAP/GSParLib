#include <iostream>
#include <chrono>
#include <vector>

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

void vector_sum(const std::vector<unsigned int> &a,
                const std::vector<unsigned int> &b,
                std::vector<unsigned int> &result) {

    try {

        unsigned int max = result.size();

        auto driver = Instance::getInstance();
        driver->init();

        int numGpus = driver->getGpuCount();
        if (numGpus == 0) {
            std::cout << "No GPU found, interrupting test" << std::endl;
            exit(-1);
        }

        // Get the first GPU
        auto gpu = driver->getGpu(0);

        // MemoryObject* a_dev = new MemoryObject(gpu, a);
        auto a_dev = gpu->malloc(a);
        auto b_dev = gpu->malloc(b);
        // Async copy
        a_dev->copyInAsync();
        b_dev->copyInAsync();
        AsyncExecutionSupport::waitAllAsync({ a_dev, b_dev });
        // Sync copy
        // a_dev->copyIn();
        // b_dev->copyIn();

        auto result_dev = gpu->malloc(result);

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

void print_vector(const std::vector<unsigned int> &vector, bool compact = false) {
    if (compact || vector.size() > 100) {
        std::cout << vector.front() << "..." << vector.back();
    } else {
        for (const unsigned int& i : vector) {
            std::cout << i << " ";
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
    std::vector<unsigned int> result = std::vector<unsigned int>(VECTOR_SIZE);
    std::vector<unsigned int> a = std::vector<unsigned int>(VECTOR_SIZE);
    std::vector<unsigned int> b = std::vector<unsigned int>(VECTOR_SIZE);
    for (unsigned int i = 0; i < VECTOR_SIZE; i++) {
        a.at(i) = i;
        b.at(i) = i + 1;
        result.at(i) = 0;
    }

    std::cout << "Summing vectors:" << std::endl;
    std::cout << "Vector A: ";
    print_vector(a);
    std::cout << "Vector B: ";
    print_vector(b);

    auto t_start = std::chrono::steady_clock::now();

    vector_sum(a, b, result);

    auto t_end = std::chrono::steady_clock::now();

    // Output the result buffer
    std::cout << "Result:   ";
    print_vector(result);

    result = std::vector<unsigned int>();
    a = std::vector<unsigned int>();
    b = std::vector<unsigned int>();

    std::cout << "Test finished succesfully in " << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() << " ms " << std::endl;

    return 0;
}
