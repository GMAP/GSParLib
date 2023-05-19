#include <iostream>
#include <chrono>

#ifdef GSPARDRIVER_OPENCL

    const char* nameOfGSParDriver = "OpenCL";

    #include "GSPar_OpenCL.hpp"
    using namespace GSPar::Driver::OpenCL;

#else

    const char* nameOfGSParDriver = "CUDA";

    #include "GSPar_CUDA.hpp"
    using namespace GSPar::Driver::CUDA;

#endif

const char* kernelSource = GSPAR_STRINGIZE_SOURCE(
    GSPAR_DEVICE_MACRO_BEGIN CONSTANT_N 42 GSPAR_DEVICE_MACRO_END
    GSPAR_DEVICE_KERNEL void info_kernel(int N) {
        unsigned int idx_x = gspar_get_global_id(0);
        unsigned int idx_y = gspar_get_global_id(1);
        unsigned int blk_x = gspar_get_block_size(0);
        unsigned int blk_y = gspar_get_block_size(1);
        unsigned int blkid_x = gspar_get_block_id(0);
        unsigned int blkid_y = gspar_get_block_id(1);
        unsigned int thr_x = gspar_get_thread_id(0);
        unsigned int thr_y = gspar_get_thread_id(1);
        gspar_synchronize_local_threads(); // Unnecessary, just for show
        printf("Thread [%u,%u]: Dim (%u, %u), Block (%u, %u), thread (%u, %u), constant N: %d, parameter N: %d\n",
            idx_x, idx_y, blk_x, blk_y, blkid_x, blkid_y, thr_x, thr_y, CONSTANT_N, N);
    }
);

int main(int argc, const char * argv[]) {

    std::cout << "Testing GSPar Driver: " << nameOfGSParDriver << std::endl;

    try {

        auto t_start = std::chrono::steady_clock::now();

        Instance* driver = Instance::getInstance();
        driver->init();

        int numGpus = driver->getGpuCount();
        if (numGpus == 0) {
            std::cout << "No GPU found, interrupting test" << std::endl;
            exit(-1);
        }

        auto gpus = driver->getGpuList();

        std::cout << "Found " << numGpus << " GPU devices:" << std::endl;
        int d = 0;
        for (auto const& gpu : gpus) {
            std::cout << "Device #" << ++d << ": \"" << gpu->getName() << "\"";
            std::cout << " (" << (gpu->isIntegratedMainMemory() ? "integrated" : "dedicated") << ")" << std::endl;
            std::cout << "    Memory:" << std::endl;
            std::cout << "      Total global memory:        " << gpu->getGlobalMemorySizeBytes()/(1024 * 1024) << " MB" << std::endl;
            std::cout << "      Total local memory:         " << gpu->getLocalMemorySizeBytes()/1024 << " KB" << std::endl;
            std::cout << "      Total shared memory per CU: " << gpu->getSharedMemoryPerComputeUnitSizeBytes()/1024 << " KB" << std::endl;
            std::cout << "    Number of compute units (CU): " << gpu->getComputeUnitsCount() << std::endl;
            std::cout << "    Maximum threads per block:    " << gpu->getMaxThreadsPerBlock() << std::endl;
            std::cout << "    Device clock rate:            " << gpu->getClockRateMHz() << " MHz" << std::endl;
        }

        auto gpu = gpus.front();
        std::cout << "Running test kernel in the first GPU (" << gpu->getName() << ")" << std::endl;

        auto kernel = gpu->prepareKernel(kernelSource, "info_kernel");
        // auto kernel = new Kernel(gpu, kernelSource, "info_kernel");

        int N = 12;
        kernel->setParameter(sizeof(N), &N);

        kernel->runAsync({2, 3});
        // kernel->waitAsync();

        delete kernel;

        auto t_end = std::chrono::steady_clock::now();

        std::cout << "Test finished succesfully in " << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() << " ms " << std::endl;

    } catch (GSPar::GSParException &ex) {
        std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
    }

    return 0;
}
