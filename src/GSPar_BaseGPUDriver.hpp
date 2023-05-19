
#ifndef __GSPAR_BASEGPUDRIVER_INCLUDED__
#define __GSPAR_BASEGPUDRIVER_INCLUDED__

#define SUPPORTED_DIMS 3

#include <string>
#include <list>
#include <iosfwd>
#include <ostream>
#include <mutex>
#include <math.h>
#ifdef GSPAR_DEBUG
#include <iostream> //std::cout and std::cerr
#endif

///// Forward declarations /////

namespace GSPar {
    namespace Driver {

        enum Runtime {
            GSPAR_RT_NONE,
            GSPAR_RT_CUDA,
            GSPAR_RT_OPENCL
        };

        struct SingleDimension {
            unsigned long max;
            unsigned long min;
            // TODO step
            // unsigned long step;

            SingleDimension() : SingleDimension(0, 0) { }
            SingleDimension(unsigned long max) : SingleDimension(max, 0) { }
            SingleDimension(unsigned long max, unsigned long min) : max(max), min(min) { }

            unsigned long delta() { return this->max - this->min; }

            std::string toString() {
                std::string out;
                if (this->min) {
                    out += std::to_string(this->min) + " to ";
                }
                out += std::to_string(this->max);
                return out;
            }

            SingleDimension& operator= (SingleDimension other) { // https://en.cppreference.com/w/cpp/language/copy_assignment
                if (&other == this) return *this;
                this->max = other.max;
                this->min = other.min;
                return *this;
            }
            explicit operator bool() const { return this->max > 0; }
            bool operator==(SingleDimension& other) {
                return this->max == other.max && this->min == other.min;
            }
            bool operator!=(SingleDimension& other) { return !(*this == other); }
            SingleDimension& operator*=(unsigned int number) {
                this->max *= number;
                this->min *= number;
                return *this;
            }
            SingleDimension operator*(unsigned int number) { return SingleDimension(this->max*number, this->min*number); }
        };

        struct Dimensions {
            // TODO remove this crap
            SingleDimension _empty;
            SingleDimension x;
            SingleDimension y;
            SingleDimension z;

            Dimensions() : _empty(0), x(0), y(0), z(0) { };
            Dimensions(SingleDimension x, SingleDimension y) : Dimensions() {
                this->x = x;
                this->y = y;
            }
            Dimensions(SingleDimension x, SingleDimension y, SingleDimension z) : Dimensions(x, y) {
                this->z = z;
            }
            Dimensions(unsigned long maxX, unsigned long maxY, unsigned long maxZ) : Dimensions(SingleDimension(maxX), SingleDimension(maxY), SingleDimension(maxZ)) { };
            /**
             * Creates a 3-Dimensions with specified max values and min=0
             * @param max Max values for the 3 dimensions
             */
            Dimensions(unsigned long max[3]) : Dimensions(max[0], max[1], max[2]) { };
            /**
             * Created a 3-Dimensions with specified max and min values.
             * Eg.: dims[0][0] is max value for X dim, dims[0][1] is min value for X dim, dims[1] is Y, dims[2] is Z
             * @param dims Max and min values for dimensions
             */
            Dimensions(unsigned long dims[3][2]) : Dimensions(SingleDimension(dims[0][0], dims[0][1]), SingleDimension(dims[1][0], dims[1][1]), SingleDimension(dims[2][0], dims[2][1])) { };
            // This constructor gets called instead of copy assignment when assignin directly or passing values to function
            Dimensions(const Dimensions &other) : Dimensions(other.x, other.y, other.z) { };

            bool is(unsigned int dimension) { return (bool)((*this)[dimension]); };
            int getCount() const { return (bool)this->x + (bool)this->y + (bool)this->z; }

            std::string getName(unsigned int dimension) {
                if (this->is(dimension)) {
                    switch (dimension) {
                    case 0: return "x";
                    case 1: return "y";
                    case 2: return "z";
                    }
                }
                return NULL;
            }

            std::string toString() {
                std::string out;
                out += "[dim" + std::to_string(this->getCount()) + ":";
                for (int d = 0; d < this->getCount(); d++) {
                    out += (*this)[d].toString() + "x";
                }
                out.pop_back();
                out += "]";
                return out;
            }

            // https://en.cppreference.com/w/cpp/language/operators
            SingleDimension& operator[] (const int index) {
                if (index == 0) return this->x;
                if (index == 1) return this->y;
                if (index == 2) return this->z;
                return this->_empty; // TODO Should we throw an exception?
            }
            Dimensions& operator= (Dimensions& other) { // https://en.cppreference.com/w/cpp/language/copy_assignment
                if (&other == this) return *this;
                this->_empty = other._empty;
                this->x = other.x;
                this->y = other.y;
                this->z = other.z;
                return *this;
            }
            bool operator==(Dimensions& other) {
                bool ret = this->getCount() == other.getCount();
                for (int d = 0; ret && d < 3; d++) {
                    ret = ret && (*this)[d] == other[d];
                }
                return ret;
            }
            bool operator!=(Dimensions& other) { return !(*this == other); }
            Dimensions& operator*=(unsigned int number) {
                for (int d = 0; d < this->getCount(); d++) {
                    (*this)[d] *= number;
                }
                return *this;
            }
            Dimensions operator*(unsigned int number) { return Dimensions(
                this->x ? this->x*number : 0,
                this->y ? this->y*number : 0,
                this->z ? this->z*number : 0
            ); }
            explicit operator bool() const { return this->getCount() > 0 && (bool)this->x; }
        };

        template <class TLibCode>
        class BaseException;

        class BaseExecutionFlowBase;

        template <class TExecutionFlow, class TDevice, class TLibFlowObject>
        class BaseExecutionFlow;

        template <class TLibAsyncObj>
        class BaseAsyncExecutionSupport;

        class BaseInstanceBase;

        template <class TExecutionFlow, class TDevice, class TKernel, class TMemoryObject, class TChunkedMemoryObject, class TKernelGenerator>
        class BaseInstance;

        class BaseDeviceBase;

        template <class TExecutionFlow, class TKernel, class TMemoryObject, class TChunkedMemoryObject, class TLibContext, class TLibDevice, class TLibFlowObject>
        class BaseDevice;

        class BaseKernelBase;

        template <class TExecutionFlow, class TDevice, class TMemoryObject, class TChunkedMemoryObject, class TLibAsyncObj>
        class BaseKernel;

        /**
         * Class to allow storing pointers to BaseMemoryObject without templates.
         */
        class BaseMemoryObjectBase {
        protected:
            size_t size;
            void* hostPtr = NULL;
        public:
            BaseMemoryObjectBase() {}
            virtual ~BaseMemoryObjectBase() {}

            size_t getSize() { return this->size; }
            void* getHostPointer() { return this->hostPtr; }
        };

        template <class TException, class TExecutionFlow, class TDevice, class TLibMemoryObject, class TLibAsyncObj>
        class BaseMemoryObject;

        template <class TException, class TExecutionFlow, class TDevice, class TLibMemoryObject, class TLibAsyncObj>
        class BaseChunkedMemoryObject;
        
        template <class TExecutionFlow, class TDevice, class TLibAsyncObj, class TLibFlowObject>
        class BaseStreamElement;

        class BaseKernelGeneration;

    }
}

#include "GSPar_Base.hpp"
#include "GSPar_BaseParallelPattern.hpp"

namespace GSPar {
    namespace Driver {

        #define defaultExceptionDetails() std::string(__func__) + " in " + std::string(__FILE__) + ":" + std::to_string(__LINE__)
        #define throwExceptionIfFailed( code ) Exception::throwIfFailed( code, defaultExceptionDetails() )

        /**
         * Base class for exceptions
         * 
         * @param <TLibCode> Type of the (lib-specific) error code
         */
        template <class TLibCode>
        class BaseException : public GSParException {
        protected:
            TLibCode code;

            virtual std::string getErrorString(TLibCode code) = 0;

            template <class TChildException>
            static TChildException* checkError(TLibCode code, TLibCode successCode, std::string details = "") {
                if (code != successCode) {
                    return new TChildException(code, details);
                }
                return nullptr;
            }

            template <class TChildException>
            static void throwIfFailed(TLibCode code, TLibCode sucessCode, std::string details = "") {
                TChildException* ex = BaseException::checkError<TChildException>(code, sucessCode, details);
                if (ex != nullptr) {
                    throw *ex;
                }
            }

        public:
            BaseException() : GSParException() { }
            explicit BaseException(std::string msg, std::string details = "") : GSParException(msg, details) { }
            explicit BaseException(TLibCode code, std::string details = "") : GSParException("", details) {
                this->code = code;
                this->details = details;
                // This virtual method call must be placed in child's implementation
                // this->msg = this->getErrorString(code);
            }
            TLibCode getCode() {
                return this->code;
            }
        };

        /**
         * Class to allow storing pointers to BaseExecutionFlow without templates.
         */
        class BaseExecutionFlowBase {
        public:
            BaseExecutionFlowBase() {}
            virtual ~BaseExecutionFlowBase() {}
        };

        /**
         * Classes that manage an execution flow should inherit from this class.
         * 
         * @param <TExecutionFlow> Type of the specialized BaseExecutionFlow class
         * @param <TDevice> Type of the specialized BaseDevice class
         * @param <TLibFlowObject> Type of the (lib-specific) underlying flow control object
         */
        template <class TExecutionFlow, class TDevice, class TLibFlowObject>
        class BaseExecutionFlow : public BaseExecutionFlowBase {
        protected:
            TDevice* device = NULL;
            TLibFlowObject flowObject = NULL;

        public:
            BaseExecutionFlow() : BaseExecutionFlowBase() { }
            explicit BaseExecutionFlow(TDevice* device) {
                this->device = device;
            }
            virtual ~BaseExecutionFlow() { }
            virtual void setBaseFlowObject(TLibFlowObject flowObject) { this->flowObject = flowObject; }
            virtual TLibFlowObject getBaseFlowObject() { return this->flowObject; }
            virtual void setDevice(TDevice* device) { this->device = device; }
            virtual TDevice* getDevice() { return this->device; }

            /**
             * Start the execution flow if it hasn't been started yet.
             * Can be safely called multiple times.
             */
            virtual TLibFlowObject start() = 0;
            /**
             * Wait for the operations in this execution flow to complete.
             */
            virtual void synchronize() = 0;

            /**
             * Check if the execution flow was provided and get the device's default execution flow otherwise.
             * Start the execution flow and returns
             * 
             * @param device The device from which get the default execution flow if the executionFlow is NULL
             * @param executionFlow The execution flow to start
             */
            static TLibFlowObject checkAndStartFlow(TDevice* device, TExecutionFlow* executionFlow = NULL) {
                if (executionFlow) {
                    return executionFlow->start();
                } else {
                    return device->startDefaultExecutionFlow();
                }
            }
        };

        /**
         * Classes that support asynchronous execution should inherit from this class.
         * 
         * @param <TLibAsyncObj> Type of the (lib-specific) underlying async object
         */
        template <class TLibAsyncObj>
        class BaseAsyncExecutionSupport {
        protected:
            TLibAsyncObj asyncObject = NULL;
            bool runningAsync = false;

            virtual void clearRunningAsync() {
                this->runningAsync = false;
            }

        public:
            BaseAsyncExecutionSupport(TLibAsyncObj asyncObj = NULL) {
                if (asyncObj) this->asyncObject = asyncObj;
            }
            virtual ~BaseAsyncExecutionSupport() { }
            virtual void setBaseAsyncObject(TLibAsyncObj asyncObject) { this->asyncObject = asyncObject; }
            virtual TLibAsyncObj getBaseAsyncObject() { return this->asyncObject; }
            virtual bool isRunningAsync() { return this->runningAsync; }

            /**
             * Wait for the async operations represented by this async object to complete
             */
            virtual void waitAsync() = 0;
        };

        /**
         * Class to allow references to BaseInstance without templates.
         */
        class BaseInstanceBase {
        protected:
            Runtime runtime;
            BaseInstanceBase(Runtime rt) : runtime(rt) { }
        public:
            BaseInstanceBase() { }
            virtual ~BaseInstanceBase() { }
        };

        /**
         * This class represents the entry point of the API.
         * 
         * @param <TExecutionFlow> Type of the specialized BaseExecutionFlow class
         * @param <TDevice> Type of the specialized BaseDevice class
         * @param <TKernel> Type of the specialized BaseKernel class
         * @param <TMemoryObject> Type of the specialized BaseMemoryObject class
         * @param <TChunkedMemoryObject> Type of the specialized BaseChunkedMemoryObject class
         * @param <TKernelGenerator> Type of the specialized BaseKernelGenerator class
         */
        template <class TExecutionFlow, class TDevice, class TKernel, class TMemoryObject, class TChunkedMemoryObject, class TKernelGenerator>
        class BaseInstance :
            public BaseInstanceBase {
        private:
            TKernelGenerator* kernelGenerator = nullptr;

        protected:
            bool instanceInitiated = false;
            std::vector<TDevice*> devices;
            virtual void loadGpuList() = 0;
            virtual void clearGpuList() {
                for (size_t i = 0; i < this->devices.size(); i++) {
                    delete this->devices[i];
                }
                this->devices.clear();
            }
            
            BaseInstance(Runtime rt) : BaseInstanceBase(rt) { }

        public:
            BaseInstance() {}
            virtual ~BaseInstance() {
                if (!this->devices.empty()) {
                    this->clearGpuList();
                }
            }
            virtual void init() = 0;
            virtual unsigned int getGpuCount() = 0;
            virtual std::vector<TDevice*> getGpuList() {
                if (this->devices.empty()) {
                    this->loadGpuList();
                }
                return this->devices;
            }
            virtual TDevice* getGpu(unsigned int index) {
                std::vector<TDevice*> gpus = this->getGpuList();
                if (gpus.size() > index) {
                    return gpus.at(index);
                }
                return nullptr;
            }
            virtual TKernelGenerator* getKernelGenerator() {
                // TODO implement thread safety
                if (!this->kernelGenerator) {
                    this->kernelGenerator = new TKernelGenerator();
                }
                return this->kernelGenerator;
            }

            static TExecutionFlow getExecutionFlowType() { return TExecutionFlow(); }
            static TDevice getDeviceType() { return TDevice(); }
            static TKernel getKernelType() { return TKernel(); }
            static TMemoryObject getMemoryObjectType() { return TMemoryObject(); }
            static TChunkedMemoryObject getChunkedMemoryObjectType() { return TChunkedMemoryObject(); }
        };

        /**
         * Class to allow references to BaseDevice without templates.
         */
        class BaseDeviceBase {
        public:
            BaseDeviceBase() { }
            virtual ~BaseDeviceBase() { }
        };

        /**
         * Class that represent a single GPU device
         * 
         * @param <TExecutionFlow> Type of the specialized BaseExecutionFlow class
         * @param <TKernel> Type of the specialized BaseKernel class
         * @param <TMemoryObject> Type of the specialized BaseMemoryObject class
         * @param <TChunkedMemoryObject> Type of the specialized BaseChunkedMemoryObject class
         * @param <TLibContext> Type of the (lib-specific) underlying context object
         * @param <TLibDevice> Type of the (lib-specific) underlying device object
         * @param <TLibFlowObject> Type of the (lib-specific) underlying async execution flow object (the same used when inheriting BaseAsyncExecutionSupport)
         */
        template <class TExecutionFlow, class TKernel, class TMemoryObject, class TChunkedMemoryObject, class TLibContext, class TLibDevice, class TLibFlowObject>
        class BaseDevice :
            public BaseDeviceBase {
        protected:
            mutable std::mutex libContextMutex;
            TLibContext libContext = NULL;
            TLibDevice libDevice = NULL;
            mutable std::mutex defaultExecutionFlowMutex;
            TExecutionFlow* defaultExecutionFlow = NULL; //TODO use a smart pointer

        public:
            BaseDevice() { }
            virtual ~BaseDevice() { }
            virtual TExecutionFlow* getDefaultExecutionFlow() = 0;
            virtual void setBaseDeviceObject(TLibDevice device) { this->libDevice = device; }
            virtual TLibDevice getBaseDeviceObject() { return this->libDevice; }
            virtual void setContext(TLibContext context) { this->libContext = context; }
            virtual TLibContext getContext() { return this->libContext; }

            virtual TLibFlowObject startDefaultExecutionFlow() = 0;
            virtual const std::string getName() = 0;
            virtual unsigned int getComputeUnitsCount() = 0; // Number of multiprocessors
            virtual unsigned int getWarpSize() = 0;
            virtual unsigned int getMaxThreadsPerBlock() = 0;
            /**
             * Device's global memory size
             */
            virtual unsigned long getGlobalMemorySizeBytes() = 0;
            /**
             * Device's local (block-shared) memory size
             */
            virtual unsigned long getLocalMemorySizeBytes() = 0;
            /**
             * Device's amount of shared memory per compute unit
             */
            virtual unsigned long getSharedMemoryPerComputeUnitSizeBytes() = 0;
            virtual unsigned int getClockRateMHz() = 0;
            virtual bool isIntegratedMainMemory() = 0;
            // virtual bool supportUnifiedMemory() = 0; //CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING
            virtual TMemoryObject* malloc(long size, void* hostPtr = nullptr, bool readOnly = false, bool writeOnly = false) = 0;
            virtual TMemoryObject* malloc(long size, const void* hostPtr = nullptr) = 0;
            virtual TChunkedMemoryObject* mallocChunked(unsigned int chunks, long chunkSize, void** hostPointers = nullptr, bool readOnly = false, bool writeOnly = false) = 0;
            virtual TChunkedMemoryObject* mallocChunked(unsigned int chunks, long chunkSize, const void** hostPointers = nullptr) = 0;
            // Can't convert this BaseGPUDriver instance to child Driver instance
            // virtual TMemoryObject* malloc(long size, void* hostPtr = NULL) {
            //     return new TMemoryObject(this, size, hostPtr, false, false);
            // }
            virtual TKernel* prepareKernel(const std::string kernelSource, const std::string kernelName) = 0;
            virtual std::vector<TKernel*> prepareKernels(const std::string kernelSource, const std::vector<std::string> kernelNames) = 0;
        };

        /**
         * Class to allow storing pointers to BaseKernel without templates.
         */
        class BaseKernelBase {
        public:
            BaseKernelBase() {}
            virtual ~BaseKernelBase() {}

            virtual void cloneInto(BaseKernelBase* other) { }
            virtual Dimensions getNumBlocksAndThreads(Dimensions dims, const unsigned int maxThreadsPerBlock, size_t* maxThreadsDimension) { return dims; }
            virtual Dimensions getNumBlocksAndThreadsFor(Dimensions dims) { return dims; }
        };

        /**
         * Class that represent a single GPU kernel, which can be invoked multiple times.
         * 
         * @param <TExecutionFlow> Type of the specialized BaseExecutionFlow class
         * @param <TDevice> Type of the specialized BaseDevice class
         * @param <TMemoryObject> Type of the specialized BaseMemoryObject class
         * @param <TChunkedMemoryObject> Type of the specialized BaseChunkedMemoryObject class
         * @param <TLibAsyncObj> Type of the (lib-specific) underlying async object (the same used when inheriting BaseAsyncExecutionSupport)
         */
        template <class TExecutionFlow, class TDevice, class TMemoryObject, class TChunkedMemoryObject, class TLibAsyncObj>
        class BaseKernel :
            public BaseKernelBase,
            virtual public BaseAsyncExecutionSupport<TLibAsyncObj> {
        protected:
            std::string kernelName;
            TDevice* device;
            unsigned int parameterCount = 0;
            unsigned int sharedMemoryBytes = 0;
            Dimensions numThreadsPerBlock = {0, 0, 0};

            BaseKernel(TDevice* device) : BaseKernel() {
                this->device = device;
            }
            virtual Dimensions getNumBlocksAndThreads(Dimensions dims, const unsigned int maxThreadsPerBlock, size_t* maxThreadsDimension) override {
                #ifdef GSPAR_DEBUG
                    std::stringstream ss; // Using stringstream eases multi-threaded debugging
                    ss.str("");
                #endif
                // maxThreadsDimension is unsigned int[SUPPORTED_DIMS]
                // Max is threads, min is blocks
                Dimensions blocksAndThreads = {
                    {1, 1}, // X
                    {1, 1}, // Y
                    {1, 1}  // Z
                };

                if (dims.y) {
                    if (dims.z) {

                        // TODO support 3D kernels
                        throw GSParException("3-dimensional kernels not supported");
                        
                    } else {
                        if ((dims.x.max * dims.y.max) > maxThreadsPerBlock) {
                            int maxThreads2D = sqrt(maxThreadsPerBlock);
                            maxThreadsDimension[0] = maxThreads2D;
                            maxThreadsDimension[1] = maxThreads2D;
                        }
                    }
                }
                
                #ifdef GSPAR_DEBUG
                    if (this->numThreadsPerBlock) {
                        ss << "[GSPar Kernel " << this << "] Configured num of threads per block is " << this->numThreadsPerBlock.toString() << std::endl;
                        std::cout << ss.str();
                        ss.str("");
                    }
                #endif

                for (int d = 0; d < SUPPORTED_DIMS; d++) {
                    if (dims[d]) {
                        if (numThreadsPerBlock[d] && numThreadsPerBlock[d].max < maxThreadsDimension[d]) {
                            maxThreadsDimension[d] = numThreadsPerBlock[d].max;
                        }
                        if (dims[d].delta() <= maxThreadsDimension[d]) {
                            blocksAndThreads[d].min = 1; // Blocks
                            blocksAndThreads[d].max = dims[d].delta(); // Threads
                        } else {
                            blocksAndThreads[d].min = ceil((double)dims[d].delta()/maxThreadsDimension[d]); // Blocks
                            blocksAndThreads[d].max = maxThreadsDimension[d]; // Threads
                        }
                    }
                }

                return blocksAndThreads;
            }

        public:
            BaseKernel() { }
            BaseKernel(TDevice* device, const std::string kernelSource, const std::string kernelName) : BaseKernel(device) {
                this->kernelName = kernelName;
            }
            virtual ~BaseKernel() { }
            virtual void cloneInto(BaseKernelBase* baseOther) override {
                BaseKernelBase::cloneInto(baseOther);
                BaseKernel* other = static_cast<BaseKernel*>(baseOther);
                other->kernelName = this->kernelName;
                other->device = this->device;
                other->parameterCount = this->parameterCount;
                other->sharedMemoryBytes = this->sharedMemoryBytes;
            }
            virtual void setSharedMemoryAllocation(unsigned int sharedMemoryBytes) {
                this->sharedMemoryBytes = sharedMemoryBytes;
            }
            virtual BaseKernel& setNumThreadsPerBlockForX(unsigned long num) { this->numThreadsPerBlock[0] = num; return *this; }
            virtual BaseKernel& setNumThreadsPerBlockForY(unsigned long num) { this->numThreadsPerBlock[1] = num; return *this; }
            virtual BaseKernel& setNumThreadsPerBlockForZ(unsigned long num) { this->numThreadsPerBlock[2] = num; return *this; }
            virtual BaseKernel& setNumThreadsPerBlockFor(int dim, unsigned long num) {
                this->numThreadsPerBlock[dim] = num;
                return *this;
            }
            virtual BaseKernel& setNumThreadsPerBlock(unsigned long numX, unsigned long numY, unsigned long numZ) {
                this->numThreadsPerBlock[0] = numX;
                this->numThreadsPerBlock[1] = numY;
                this->numThreadsPerBlock[2] = numZ;
                return *this;
            }
            // TODO setParameter should return the Kernel object itself to allow fluent programming, such as BaseParallelPattern
            virtual int setParameter(TMemoryObject* memoryObject) = 0;
            virtual int setParameter(TChunkedMemoryObject* chunkedMemoryObject) = 0;
            virtual int setParameter(size_t parmSize, void* parm) = 0;
            virtual int setParameter(size_t parmSize, const void* parm) = 0;
            virtual void clearParameters() {
                this->parameterCount = 0;
            }
            virtual void runAsync(unsigned long max[3], TExecutionFlow* executionFlow = NULL) {
                this->runAsync(Dimensions(max), executionFlow);
            }
            virtual void runAsync(Dimensions max, TExecutionFlow* executionFlow = NULL) = 0;
        };

        /**
         * Class that represent a single memory object.
         * It is bound to a device. It holds a (optional) host and a device pointer.
         * 
         * @param <TException> Type of the specialized BaseException class
         * @param <TExecutionFlow> Type of the specialized BaseExecutionFlow class
         * @param <TDevice> Type of the specialized BaseDevice class
         * @param <TLibMemoryObject> Type of the (lib-specific) underlying error code
         * @param <TLibAsyncObj> Type of the (lib-specific) underlying async object (the same used when inheriting BaseAsyncExecutionSupport)
         */
        template <class TException, class TExecutionFlow, class TDevice, class TLibMemoryObject, class TLibAsyncObj>
        class BaseMemoryObject :
            virtual public BaseMemoryObjectBase,
            virtual public BaseAsyncExecutionSupport<TLibAsyncObj> {
        protected:
            // https://www.learncpp.com/cpp-tutorial/3-8a-bit-flags-and-bit-masks/
            static const unsigned char CAN_READ_FLAG = 1 << 0;
            static const unsigned char CAN_WRITE_FLAG = 1 << 1;

            TDevice* device;
            TLibMemoryObject devicePtr = NULL;
            unsigned char flags = CAN_READ_FLAG | CAN_WRITE_FLAG;
            bool _isPinnedHostMemory = false;

            /**
             * @param readOnly identify that this memory object can only be read inside kernel
             * @param writeOnly identify that this memory object can only be written inside kernel
             */
            explicit BaseMemoryObject(bool readOnly, bool writeOnly) {
                if (readOnly && writeOnly) {
                    throw TException("A memory object can't be read-only and write-only at the same time");
                } else if (readOnly) {
                    this->flags &= ~CAN_WRITE_FLAG;
                } else if (writeOnly) {
                    this->flags &= ~CAN_READ_FLAG;
                }
            }
            explicit BaseMemoryObject(TDevice* device, size_t size, void* hostPtr, bool readOnly, bool writeOnly) : BaseMemoryObject(readOnly, writeOnly) {
                this->device = device;
                this->hostPtr = hostPtr;
                this->size = size;
            }
            explicit BaseMemoryObject(TDevice* device, size_t size, const void* hostPtr) :
                // const pointer must be read-only
                BaseMemoryObject(device, size, const_cast<void*>(hostPtr), true, false) { }

        public:
            BaseMemoryObject() {}
            virtual ~BaseMemoryObject() {}
            TLibMemoryObject getBaseMemoryObject() { return this->devicePtr; }
            bool isReadOnly() { return !(this->flags & CAN_WRITE_FLAG); }
            bool isWriteOnly() { return !(this->flags & CAN_READ_FLAG); }
            void bindTo(void* hostPtr) { this->hostPtr = hostPtr; }
            void bindTo(void* hostPtr, size_t size) {
                this->bindTo(hostPtr);
                this->size = size;
            }
            virtual void pinHostMemory() { this->setPinnedHostMemory(true); }
            virtual void setPinnedHostMemory(bool pinned) { this->_isPinnedHostMemory = pinned; }
            virtual bool isPinnedHostMemory() { return this->_isPinnedHostMemory; }
            virtual void copyIn() = 0;
            virtual void copyOut() = 0;
            virtual void copyInAsync(TExecutionFlow* executionFlow = NULL) = 0;
            virtual void copyOutAsync(TExecutionFlow* executionFlow = NULL) = 0;
        };

        /**
         * Class that represent a chunked memory object.
         * It is bound to a device. It holds a bunch of host pointers (the chunks) and a single device pointer.
         * 
         * @param <TException> Type of the specialized BaseException class
         * @param <TExecutionFlow> Type of the specialized BaseExecutionFlow class
         * @param <TDevice> Type of the specialized BaseDevice class
         * @param <TLibMemoryObject> Type of the (lib-specific) underlying error code
         * @param <TLibAsyncObj> Type of the (lib-specific) underlying async object (the same used when inheriting BaseAsyncExecutionSupport)
         */
        template <class TException, class TExecutionFlow, class TDevice, class TLibMemoryObject, class TLibAsyncObj>
        class BaseChunkedMemoryObject :
            virtual public BaseMemoryObjectBase,
            virtual public BaseMemoryObject<TException, TExecutionFlow, TDevice, TLibMemoryObject, TLibAsyncObj> {
        protected:
            void** hostPointers = NULL;
            unsigned int chunks = 0;
            // We use the base property size for the chunkSize (size of each data chunk)

            // TODO shouldn't we call the base constructor?
            explicit BaseChunkedMemoryObject(TDevice* device, unsigned int chunks, size_t chunkSize, void** hostPointers, bool readOnly, bool writeOnly) {
                this->device = device;
                this->hostPtr = NULL;
                this->size = chunkSize;
                this->hostPointers = hostPointers;
                this->chunks = chunks;
            }
            explicit BaseChunkedMemoryObject(TDevice* device, unsigned int chunks, size_t chunkSize, const void** hostPointers) :
                // const pointer must be read-only
                BaseChunkedMemoryObject(device, chunks, chunkSize, const_cast<void**>(hostPointers), true, false) { }

        public:
            BaseChunkedMemoryObject() : BaseMemoryObject<TException, TExecutionFlow, TDevice, TLibMemoryObject, TLibAsyncObj>() { }
            virtual ~BaseChunkedMemoryObject() { }
            size_t getChunkSize() { return this->size; }
            unsigned int getChunkCount() { return this->chunks; }
        };

        /**
         * Class that will end up being part of the stream elements
         * 
         * @param <TExecutionFlow> Type of the specialized BaseExecutionFlow class
         * @param <TDevice> Type of the specialized BaseDevice class
         * @param <TLibAsyncObj> Type of the (lib-specific) underlying async object (the same used when inheriting BaseAsyncExecutionSupport)
         * @param <TLibFlowObject> Type of the (lib-specific) underlying async execution flow object (the same used when inheriting BaseAsyncExecutionSupport)
         */
        template <class TExecutionFlow, class TDevice, class TLibAsyncObj, class TLibFlowObject>
        class BaseStreamElement :
            virtual public BaseAsyncExecutionSupport<TLibAsyncObj>,
            virtual public BaseExecutionFlow<TExecutionFlow, TDevice, TLibFlowObject> {
        public:
            explicit BaseStreamElement(TDevice* device) {
                // We should extend BaseExecutionFlow::constructor(device)
                this->device = device;
            }
            virtual ~BaseStreamElement() {}

        };

        /**
         * Base class for kernel code generation
         */
        class BaseKernelGenerator {
        protected:
            std::array<std::string, 3> defaultStdVarNames = {"x", "y", "z"};

        public:
            virtual const std::string getKernelPrefix() = 0;
            virtual std::string generateStdFunctions() = 0;
            virtual std::string replaceMacroKeywords(std::string kernelSource) = 0;
            virtual std::string generateInitKernel(Pattern::BaseParallelPattern* pattern, Dimensions dims) = 0;
            virtual std::string generateParams(Pattern::BaseParallelPattern* pattern, Dimensions dims) = 0;
            virtual std::string generateStdVariables(Pattern::BaseParallelPattern* pattern, Dimensions dims) = 0;
            virtual std::string generateBatchedParametersInitialization(Pattern::BaseParallelPattern* pattern, Dimensions dims) = 0;
            virtual std::string getStdVarNameForDimension(std::array<std::string, 3>& patternNames, int dimension) {
                if (patternNames[dimension].empty()) {
                    return this->defaultStdVarNames[dimension];
                }
                return patternNames[dimension];
            }
            virtual std::array<std::string, 3> getStdVarNames(std::array<std::string, 3>& patternNames) {
                return {
                    this->getStdVarNameForDimension(patternNames, 0),
                    this->getStdVarNameForDimension(patternNames, 1),
                    this->getStdVarNameForDimension(patternNames, 2)
                };
            }
        };

    }
}

#endif
