#include <iostream>
#include <chrono>

#ifdef GSPARDRIVER_CUDA

    #include "GSPar_CUDA.hpp"
    using namespace GSPar::Driver::CUDA;

// #elif GSPARDRIVER_OPENCL
#else // This way my IDE doesn't complain

    #include "GSPar_OpenCL.hpp"
    using namespace GSPar::Driver::OpenCL;

#endif

const char* kernelSource = GSPAR_STRINGIZE_SOURCE(
    GSPAR_DEVICE_KERNEL void matrix_multi(long MX,
            GSPAR_DEVICE_GLOBAL_MEMORY const long *a,
            GSPAR_DEVICE_GLOBAL_MEMORY const long *b,
            GSPAR_DEVICE_GLOBAL_MEMORY long *result) {
        long i = gspar_get_global_id(0);
        long j = gspar_get_global_id(1);
        if (i < MX && j < MX) {
            for (long k = 0; k<MX; k++) {
                result[i*MX+j] += a[i*MX+k] * b[k*MX+j];
            }
        }
    }
);

void matrix_multi(const long max, const long* a, const long* b, long* result) {
    try {

        Instance* driver = Instance::getInstance();
        driver->init();

        int numGpus = driver->getGpuCount();
        if (numGpus == 0) {
            std::cerr << "No GPU found, interrupting test" << std::endl;
            exit(-1);
        }

        auto gpus = driver->getGpuList();

        // Get the first GPU
        Device* gpu = gpus.front();
        MemoryObject* a_dev = gpu->malloc(sizeof(long) * max * max, a);
        MemoryObject* b_dev = gpu->malloc(sizeof(long) * max * max, b);
        // Async copy
        // a_dev->copyInAsync();
        // b_dev->copyInAsync();
        // AsyncExecutionSupport::waitAllAsync({ a_dev->getBaseAsyncObject(), b_dev->getBaseAsyncObject() });
        // Sync copy
        a_dev->copyIn();
        b_dev->copyIn();

        MemoryObject* result_dev = gpu->malloc(sizeof(long) * max * max, result);
        result_dev->copyIn();

        // Kernel* kernel = gpu->prepareKernel(kernelSource, "matrix_multi");
        Kernel* kernel = new Kernel(gpu, kernelSource, "matrix_multi");
        
        kernel->setParameter(sizeof(max), &max);
        kernel->setParameter(a_dev);
        kernel->setParameter(b_dev);
        kernel->setParameter(result_dev);

        unsigned long dimensions[3] = {(unsigned long)max, (unsigned long)max, 0};
        kernel->runAsync(dimensions);
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

void print_matrix(long max, const long* matrix, bool compact = false) {
    if (compact || max > 100) {
        std::cout << matrix[0] << "..." << matrix[(max * max)-1];
    } else {
        for (long i = 0; i < max; i++) {
            std::cout << std::endl;
            for (long j = 0; j < max; j++) {
                std::cout << matrix[i * max + j] << " ";
            }
        }
    }
    std::cout << std::endl;
}

int main(int argc, const char * argv[]) {
    if (argc < 2) {
        std::cerr << "Use: " << argv[0] << " <matrix_size>" << std::endl;
        exit(-1);
    }
    
    const long MX = std::stoi(argv[1]);
    std::cout << "Multiplying matrixes of " << MX << " x " << MX << std::endl;
    
    // Create memory objects
    long* matrix_a = new long[MX * MX];
    long* matrix_b = new long[MX * MX];
    long* result = new long[MX * MX];
    for (long i = 0; i < MX; i++) {
        for (long j = 0; j < MX; j++) {
            matrix_a[j * MX + i] = 4;
            matrix_b[j * MX + i] = 5;
            result[j * MX + i] = 0;
        }
    }

    std::cout << "Matrix A: ";
    print_matrix(MX, matrix_a, true);
    std::cout << "Matrix B: ";
    print_matrix(MX, matrix_b, true);

    auto t_start = std::chrono::steady_clock::now();

    matrix_multi(MX, matrix_a, matrix_b, result);

    auto t_end = std::chrono::steady_clock::now();

    // Output the result buffer
    std::cout << "Result:   ";
    print_matrix(MX, result);

    delete matrix_a;
    delete matrix_b;
    delete result;

    std::cout << "Test finished succesfully in " << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() << " ms " << std::endl;

    return 0;
}
