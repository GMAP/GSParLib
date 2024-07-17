
#ifndef __GSPAR_CUDA_INCLUDED__
#define __GSPAR_CUDA_INCLUDED__

#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <cuda.h>
#include <nvrtc.h>

///// Forward declarations /////

namespace GSPar {
    namespace Driver {
        namespace CUDA {
            class Exception;
            class ExecutionFlow;
            class AsyncExecutionSupport;
            class Instance;
            class Device;
            class Kernel;
            class MemoryObject;
            class ChunkedMemoryObject;
            class StreamElement;
            class KernelGenerator;
        }
    }
}

#include "GSPar_BaseGPUDriver.hpp"

namespace GSPar {
    namespace Driver {
        namespace CUDA {

            ///// Exception /////

            class Exception :
                public BaseException<CUresult> {
            protected:
                std::string getErrorString(CUresult code) override;

            public:
                explicit Exception(std::string msg, std::string details = "");
                explicit Exception(CUresult code, std::string details = "");

                static Exception* checkError(CUresult code, std::string details = "");
                static void throwIfFailed(CUresult code, std::string details = "");
            };

            #define throwCompilationExceptionIfFailed( code, cudaProgram ) CompilationException::throwIfFailed( code, cudaProgram, defaultExceptionDetails() )

            class CompilationException :
                public BaseException<nvrtcResult> {
            protected:
                std::string getErrorString(nvrtcResult code) override;

            public:
                explicit CompilationException(std::string msg, std::string details = "");
                explicit CompilationException(nvrtcResult code, std::string details = "");

                static CompilationException* checkError(nvrtcResult code, std::string details = "");
                static void throwIfFailed(nvrtcResult code, std::string details = "");
                static void throwIfFailed(nvrtcResult code, nvrtcProgram cudaProgram, std::string details = "");
            };

            ///// ExecutionFlow /////

            class ExecutionFlow :
                virtual public BaseExecutionFlow<ExecutionFlow, Device, CUstream> {
            public:
                ExecutionFlow();
                explicit ExecutionFlow(Device* device);
                virtual ~ExecutionFlow();
                CUstream start() override;
                void synchronize() override;

                static CUstream checkAndStartFlow(Device* device, ExecutionFlow* executionFlow = NULL);
            };

            ///// AsyncExecutionSupport /////

            class AsyncExecutionSupport :
                virtual public BaseAsyncExecutionSupport<CUstream> {
            public:
                AsyncExecutionSupport(CUstream asyncObj = NULL);
                void waitAsync() override;

                static void waitAllAsync(std::initializer_list<AsyncExecutionSupport*> asyncs);
            };

            ///// Instance /////

            class Instance :
                public BaseInstance<ExecutionFlow, Device, Kernel, MemoryObject, ChunkedMemoryObject, KernelGenerator> {
            protected:
                static Instance *instance;
                void loadGpuList() override;

            public:
                Instance();
                virtual ~Instance();
                void init() override;
                unsigned int getGpuCount() override;

                static Instance* getInstance();
            };

            ///// Device /////

            class Device :
                public BaseDevice<ExecutionFlow, Kernel, MemoryObject, ChunkedMemoryObject, CUcontext, CUdevice*, CUstream> {
            private:
                mutable std::mutex attributeCacheMutex;
                std::map<CUdevice_attribute, int> attributeCache;
                int deviceId;

            public:
                using BaseDevice<ExecutionFlow, Kernel, MemoryObject, ChunkedMemoryObject, CUcontext, CUdevice*, CUstream>::malloc;
                
                Device();
                explicit Device(int ordinal);
                virtual ~Device();
                ExecutionFlow* getDefaultExecutionFlow() override;
                CUcontext getContext() override;
                CUstream startDefaultExecutionFlow() override;
                unsigned int getDeviceId();
                const std::string getName() override;
                unsigned int getComputeUnitsCount() override;
                unsigned int getWarpSize() override;
                unsigned int getMaxThreadsPerBlock() override;
                unsigned long getGlobalMemorySizeBytes() override;
                unsigned long getLocalMemorySizeBytes() override;
                unsigned long getSharedMemoryPerComputeUnitSizeBytes() override;
                unsigned int getClockRateMHz() override;
                bool isIntegratedMainMemory() override;
                MemoryObject* malloc(long size, void* hostPtr = nullptr, bool readOnly = false, bool writeOnly = false) override;
                MemoryObject* malloc(long size, const void* hostPtr = nullptr) override;
                ChunkedMemoryObject* mallocChunked(unsigned int chunks, long chunkSize, void** hostPtr = nullptr, bool readOnly = false, bool writeOnly = false) override;
                ChunkedMemoryObject* mallocChunked(unsigned int chunks, long chunkSize, const void** hostPtr = nullptr) override;
                Kernel* prepareKernel(const std::string kernelSource, const std::string kernelName) override;
                std::vector<Kernel*> prepareKernels(const std::string kernelSource, const std::vector<std::string> kernelNames) override;

                // const char* queryInfoText(cl_device_info paramName);
                const int queryInfoNumeric(CUdevice_attribute paramName, bool cacheable = true);
                std::tuple<nvrtcProgram, CUmodule> compileCudaProgramAndLoadModule(std::string source, const std::string programName);
            };

            ///// Kernel /////

            class Kernel :
                public BaseKernel<ExecutionFlow, Device, MemoryObject, ChunkedMemoryObject, CUstream>,
                public AsyncExecutionSupport {
            private:
                nvrtcProgram cudaProgram = NULL;
                CUmodule cudaModule = NULL;
                CUfunction cudaFunction = NULL;
                std::vector<void*> kernelParams;
                bool isPrecompiled;
                std::map<CUfunction_attribute, int> attributeCache;

                void loadCudaFunction(const std::string kernelName);

            public:
                Kernel();
                Kernel(Device* device, const std::string kernelSource, const std::string kernelName);
                virtual ~Kernel();
                virtual void cloneInto(BaseKernelBase* baseOther) override;
                int setParameter(MemoryObject* memoryObject) override;
                int setParameter(ChunkedMemoryObject* chunkedMemoryObject) override;
                int setParameter(size_t parmSize, void* parm) override;
                int setParameter(size_t parmSize, const void* parm) override;
                void clearParameters() override;
                Dimensions getNumBlocksAndThreadsFor(Dimensions dims) override;
                void runAsync(Dimensions max, ExecutionFlow* executionFlow = NULL) override;

                Kernel(Device* device, nvrtcProgram cudaProgram, CUmodule cudaModule, const std::string kernelName);
                const int queryInfoNumeric(CUfunction_attribute paramName, bool cacheable = true);
            };

            ///// MemoryObject /////

            class MemoryObject :
                public BaseMemoryObject<Exception, ExecutionFlow, Device, CUdeviceptr*, CUstream>,
                public AsyncExecutionSupport {
            private:
                void allocDeviceMemory();
            public:
                MemoryObject(Device* device, size_t size, void* hostPtr, bool readOnly, bool writeOnly);
                MemoryObject(Device* device, size_t size, const void* hostPtr);
                virtual ~MemoryObject();
                virtual void pinHostMemory() override;
                virtual void copyIn() override;
                virtual void copyOut() override;
                virtual void copyInAsync(ExecutionFlow* executionFlow = NULL) override;
                virtual void copyOutAsync(ExecutionFlow* executionFlow = NULL) override;
            };

            ///// ChunkedMemoryObject /////

            class ChunkedMemoryObject :
                public BaseChunkedMemoryObject<Exception, ExecutionFlow, Device, CUdeviceptr*, CUstream>,
                public AsyncExecutionSupport {
            private:
                void allocDeviceMemory();

            public:
                ChunkedMemoryObject(Device* device, unsigned int chunks, size_t chunkSize, void** hostPointers, bool readOnly, bool writeOnly);
                ChunkedMemoryObject(Device* device, unsigned int chunks, size_t chunkSize, const void** hostPointers);
                virtual ~ChunkedMemoryObject();
                virtual void pinHostMemory() override;
                // Copy all chunks
                virtual void copyIn() override;
                virtual void copyOut() override;
                virtual void copyInAsync(ExecutionFlow* executionFlow = NULL) override;
                virtual void copyOutAsync(ExecutionFlow* executionFlow = NULL) override;
                // Copy specific chunks of memory. We can't use function overloading due to the override.
                virtual void copyIn(unsigned int chunk);
                virtual void copyOut(unsigned int chunk);
                virtual void copyInAsync(unsigned int chunk, ExecutionFlow* executionFlow = NULL);
                virtual void copyOutAsync(unsigned int chunk, ExecutionFlow* executionFlow = NULL);
            };

            ///// StreamElement /////

            class StreamElement :
                public BaseStreamElement<ExecutionFlow, Device, CUstream, CUstream>,
                public AsyncExecutionSupport,
                public ExecutionFlow {
            private:
                Kernel* kernel;

            public:
                explicit StreamElement(Device* device);
                ~StreamElement();
            };

            ///// KernelGenerator /////

            class KernelGenerator :
                public BaseKernelGenerator {
            public:
                static const std::string KERNEL_PREFIX;
                static const std::string GLOBAL_MEMORY_PREFIX;
                static const std::string SHARED_MEMORY_PREFIX;
                static const std::string CONSTANT_PREFIX;
                static const std::string DEVICE_FUNCTION_PREFIX;
                static const std::string ATOMIC_ADD_POLYFILL;
                const std::string getKernelPrefix() override;
                std::string generateStdFunctions() override;
                std::string replaceMacroKeywords(std::string kernelSource) override;
                std::string generateInitKernel(Pattern::BaseParallelPattern* pattern, Dimensions dims) override;
                std::string generateParams(Pattern::BaseParallelPattern* pattern, Dimensions dims) override;
                std::string generateStdVariables(Pattern::BaseParallelPattern* pattern, Dimensions dims) override;
                std::string generateBatchedParametersInitialization(Pattern::BaseParallelPattern* pattern, Dimensions dims) override;

            };

        }
    }
}

#endif
