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
    GSPAR_DEVICE_KERNEL void vector_sum(const int max,
            GSPAR_DEVICE_GLOBAL_MEMORY const float *a,
            GSPAR_DEVICE_GLOBAL_MEMORY const float *b,
            GSPAR_DEVICE_GLOBAL_MEMORY float *result) {
        size_t gid = gspar_get_global_id(0);
        if (gid <= max) {
            result[gid] = a[gid] + b[gid];
        }
    }
);

void vector_sum(const unsigned int max, const unsigned int chunks, const float* a, const float* b, float* result) {

    try {

        Instance* driver = Instance::getInstance();
        driver->init();

        int numGpus = driver->getGpuCount();
        if (numGpus == 0) {
            std::cout << "No GPU found, interrupting test" << std::endl;
            exit(-1);
        }

        // Get the first GPU
        auto gpu = driver->getGpu(0);
        
        // Separate memory in chunks to simulate real-world chunked data
        const unsigned int itemsInEachChunk = max/chunks;

        const void** a_chunked = new const void*[chunks];
        const void** b_chunked = new const void*[chunks];
        for (unsigned int chunk = 0; chunk < chunks; chunk++) {
            a_chunked[chunk] = &a[chunk*itemsInEachChunk];
            b_chunked[chunk] = &b[chunk*itemsInEachChunk];
            // std::cout << "a_chunked[" << chunk << "] starts on " << ((float*)a_chunked[chunk])[0] << std::endl;
            // std::cout << "b_chunked[" << chunk << "] starts on " << ((float*)b_chunked[chunk])[0] << std::endl;
        }

        ChunkedMemoryObject* a_dev = gpu->mallocChunked(chunks, sizeof(float) * itemsInEachChunk, a_chunked);
        ChunkedMemoryObject* b_dev = gpu->mallocChunked(chunks, sizeof(float) * itemsInEachChunk, b_chunked);

        // Async copy
        a_dev->copyInAsync();
        b_dev->copyInAsync();
        AsyncExecutionSupport::waitAllAsync({ a_dev, b_dev });
        // Sync copy
        // a_dev->copyIn();
        // b_dev->copyIn();

        MemoryObject* result_dev = gpu->malloc(sizeof(float) * max, result);

        // Kernel* kernel = gpu->prepareKernel(kernelSource, "vector_sum");
        Kernel* kernel = new Kernel(gpu, kernelSource, "vector_sum");

        kernel->setParameter(sizeof(max), &max);
        kernel->setParameter(a_dev);
        kernel->setParameter(b_dev);
        kernel->setParameter(result_dev);

        unsigned long dimensions[3] = {max, 0, 0};
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

void print_vector(unsigned int size, const float* vector, unsigned int itemsInEachChunk = 0, bool compact = false) {
    if (compact || size > 100) {
        std::cout << vector[0] << "..." << vector[size-1];
    } else {
        for (unsigned int i = 0; i < size; i++) {
            std::cout << vector[i] << " ";
            if (itemsInEachChunk && ((i+1) % itemsInEachChunk == 0)) std::cout << "| ";
        }
    }
    std::cout << std::endl;
}

int main(int argc, const char * argv[]) {
    if (argc < 3) {
        std::cerr << "Use: " << argv[0] << " <vector_size> <chunks>" << std::endl;
        std::cerr << " <vector_size> should be divisible by <chunks>" << std::endl;
        exit(-1);
    }

    const unsigned int VECTOR_SIZE = std::stoi(argv[1]);
    const unsigned int CHUNKS = std::stoi(argv[2]);

    const unsigned int itemsInEachChunk = VECTOR_SIZE/CHUNKS;


    // Create memory objects
    float* result = new float[VECTOR_SIZE];
    float* a = new float[VECTOR_SIZE];
    float* b = new float[VECTOR_SIZE];
    for (unsigned int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
        result[i] = 0;
    }

    std::cout << "Summing vectors:" << std::endl;
    std::cout << "Vector A: ";
    print_vector(VECTOR_SIZE, a, itemsInEachChunk);
    std::cout << "Vector B: ";
    print_vector(VECTOR_SIZE, b, itemsInEachChunk);

    auto t_start = std::chrono::steady_clock::now();

    vector_sum(VECTOR_SIZE, CHUNKS, a, b, result);

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
