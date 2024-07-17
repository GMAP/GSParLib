
#ifndef __GSPAR_OPENCL_INCLUDED__
#define __GSPAR_OPENCL_INCLUDED__

#include <string>
#include <map>
#include <mutex>
#include <CL/opencl.h>

///// Forward declarations /////

namespace GSPar {
    namespace Driver {
        namespace OpenCL {
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
        namespace OpenCL {

            ///// Exception /////

            class Exception :
                public BaseException<cl_int> {
            protected:
                std::string getErrorString(cl_int code) override;

            public:
                explicit Exception(std::string msg, std::string details = "");
                explicit Exception(cl_int code, std::string details = "");

                static Exception* checkError(cl_int code, std::string details = "");
                static void throwIfFailed(cl_int code, std::string details = "");

                explicit Exception(cl_int code, cl_program program, cl_device_id device);
                static Exception* checkError(cl_int code, cl_program program, cl_device_id device);
                static void throwIfFailed(cl_int code, cl_program program, cl_device_id device);
            };

            ///// ExecutionFlow /////

            class ExecutionFlow :
                virtual public BaseExecutionFlow<ExecutionFlow, Device, cl_command_queue> {
            public:
                ExecutionFlow();
                explicit ExecutionFlow(Device* device);
                virtual ~ExecutionFlow();
                cl_command_queue start() override;
                void synchronize() override;

                static cl_command_queue checkAndStartFlow(Device* device, ExecutionFlow* executionFlow = NULL);
            };

            ///// AsyncExecutionSupport /////

            class AsyncExecutionSupport :
                virtual public BaseAsyncExecutionSupport<cl_event*> {
            protected:
                unsigned int numAsyncEvents = 0;
                /// OpenCL sometimes simply hangs on clWaitForEvents
                /// I've seen it happen when using multithread and 3 kernels (pattern->run) called sequentially by each thread
                /// The internet are full of people complaining over similar issues, and one of them used clFinish instead of clWaitForEvents, so that's what we're gonna do
                /// https://github.com/fangq/mcxcl/commit/135dc825e2905253ab0626a2b335dfee8b6e741e
                /// https://community.intel.com/t5/OpenCL/Is-there-a-driver-watchdog-time-limit-for-Intel-GPU-on-Linux/td-p/1108291
                /// Whenever an Execution Flow is filled here, we'll synchronize it instead of waiting for the event
                ExecutionFlow *executionFlow = nullptr;
            public:
                AsyncExecutionSupport(cl_event *asyncObjs = NULL, unsigned int numAsyncEvents = 0);
                virtual ~AsyncExecutionSupport();
                void setBaseAsyncObject(cl_event *asyncObject) override;
                void waitAsync() override;

                void releaseBaseAsyncObject();
                void setBaseAsyncObject(cl_event *asyncObject, unsigned int numAsyncEvents);
                void setExecutionFlowToSynchronize(ExecutionFlow *flow) {
                    this->executionFlow = flow;
                }
                static void waitAllAsync(std::initializer_list<AsyncExecutionSupport*> asyncs);
            };

            ///// Instance /////

            class Instance : public BaseInstance<ExecutionFlow, Device, Kernel, MemoryObject, ChunkedMemoryObject, KernelGenerator> {
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
                public BaseDevice<ExecutionFlow, Kernel, MemoryObject, ChunkedMemoryObject, cl_context, cl_device_id, cl_command_queue> {
            private:
                mutable std::mutex attributeCacheMutex;
                std::map<cl_device_info, void*> attributeCache;

            public:
                using BaseDevice<ExecutionFlow, Kernel, MemoryObject, ChunkedMemoryObject, cl_context, cl_device_id, cl_command_queue>::malloc;

                Device();
                explicit Device(cl_device_id deviceId);
                virtual ~Device();
                ExecutionFlow* getDefaultExecutionFlow() override;
                cl_context getContext() override;
                cl_command_queue startDefaultExecutionFlow() override;
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

                template<class T>
                const T* queryInfoDevice(cl_device_info paramName, bool cacheable = true);
                cl_program compileOCLProgram(std::string source);
            };

            ///// Kernel /////

            class Kernel :
                public BaseKernel<ExecutionFlow, Device, MemoryObject, ChunkedMemoryObject, cl_event*>,
                public AsyncExecutionSupport {
            private:
                cl_program oclProgram;
                cl_kernel oclKernel;
                bool isPrecompiled;
                std::map<cl_kernel_work_group_info, void*> attributeCache;

                void loadOclKernel(const std::string kernelName);

            public:
                Kernel();
                Kernel(Device* device, const std::string kernelSource, const std::string kernelName);
                virtual ~Kernel();
                virtual void cloneInto(BaseKernelBase* baseOther) override;
                int setParameter(MemoryObject* memoryObject) override;
                int setParameter(ChunkedMemoryObject* chunkedMemoryObject) override;
                int setParameter(size_t parmSize, void* parm) override;
                int setParameter(size_t parmSize, const void* parm) override;
                Dimensions getNumBlocksAndThreadsFor(Dimensions dims) override;
                void runAsync(Dimensions max, ExecutionFlow* executionFlow = NULL) override;

                template<class T>
                T* queryInfo(cl_kernel_work_group_info param, bool cacheable = true);
                Kernel(Device* device, cl_program oclProgram, const std::string kernelName);
            };

            ///// MemoryObject /////

            class MemoryObject :
                public BaseMemoryObject<Exception, ExecutionFlow, Device, cl_mem, cl_event*>,
                public AsyncExecutionSupport {
            private:
                void copy(bool in, bool async, ExecutionFlow* executionFlow = NULL);
                void allocDeviceMemory();

            public:
                MemoryObject(Device* device, size_t size, void* hostPtr, bool readOnly, bool writeOnly);
                MemoryObject(Device* device, size_t size, const void* hostPtr);
                virtual ~MemoryObject();
                void copyIn() override;
                void copyOut() override;
                void copyInAsync(ExecutionFlow* executionFlow = NULL) override;
                void copyOutAsync(ExecutionFlow* executionFlow = NULL) override;
            };

            ///// ChunkedMemoryObject /////

            class ChunkedMemoryObject :
                public BaseChunkedMemoryObject<Exception, ExecutionFlow, Device, cl_mem, cl_event*>,
                public AsyncExecutionSupport {
            private:
                void copy(bool in, bool async, unsigned int chunkFrom, unsigned int chunkTo, ExecutionFlow* executionFlow = NULL);
                void allocDeviceMemory();

            public:
                ChunkedMemoryObject(Device* device, unsigned int chunks, size_t chunkSize, void** hostPointers, bool readOnly, bool writeOnly);
                ChunkedMemoryObject(Device* device, unsigned int chunks, size_t chunkSize, const void** hostPointers);
                virtual ~ChunkedMemoryObject();
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
                public BaseStreamElement<ExecutionFlow, Device, cl_event*, cl_command_queue>,
                public AsyncExecutionSupport,
                public ExecutionFlow {
            private:
                Kernel* kernel;
                cl_kernel oclKernel = NULL;

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
