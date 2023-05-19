
#ifndef __GSPAR_BASEPARALLELPATTERN_INCLUDED__
#define __GSPAR_BASEPARALLELPATTERN_INCLUDED__

#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <iostream> //std::cout and std::cerr
#include <chrono>
#include <algorithm> //std::generate_n
#ifdef GSPAR_DEBUG
#include <sstream>
#include <thread>
#endif

// Includes for getTypeName
#include <type_traits>
#include <typeinfo>
#ifndef _MSC_VER
#   include <cxxabi.h>
#endif
#include <memory>
#include <cstdlib>

///// Forward declarations /////

namespace GSPar {
    namespace Pattern {

        enum ParameterValueType {
            GSPAR_PARAM_VALUE,
            GSPAR_PARAM_POINTER
        };

        enum ParameterDirection {
            GSPAR_PARAM_NONE,
            GSPAR_PARAM_IN,
            GSPAR_PARAM_OUT,
            GSPAR_PARAM_INOUT,
            GSPAR_PARAM_PRESENT // It avoids memory transfers when using a MemoryObject from user
        };

        struct VarType {
            std::string name;
            bool isPointer; //std::is_pointer
            // Remember that struct are classes also
            bool isClass; //std::is_class
            bool isConst; //std::is_const
            bool isVolatile; //std::is_volatile
            bool isLValueRef; //std::is_lvalue_reference
            bool isRValueRef; //std::is_rvalue_reference

            std::string getDeclarationName() {
                return std::string("")
                    // Classes are not supported in OpenCL C99, so we assume the class is a struct
                    + (isClass ? "struct " : "")
                    + (isConst ? "const " : "")
                    + (isVolatile ? "volatile " : "")
                    + this->getFullName();
            }

            std::string getFullName() {
                return std::string("")
                    + (isLValueRef ? "&" : "")
                    + (isRValueRef ? "&&" : "")
                    + name;
            }

            std::string toString() {
                return getFullName()
                    + (isPointer ? "*" : "");
            }
        };

        /**
         * Base class for pattern parameters
         */
        class BaseParameter {
        protected:
            bool complete = true; // Placeholder parameters are not complete
            bool batched = false; // If the parameter is part of the batch
        public:
            std::string name;
            VarType type;
            size_t size;
            ParameterValueType paramValueType;
            ParameterDirection direction;

            BaseParameter() { }
            BaseParameter(std::string name, VarType type, size_t size, ParameterValueType paramValueType, ParameterDirection direction = GSPAR_PARAM_IN, bool batched = false) :
                    name(name), type(type), size(size), paramValueType(paramValueType), direction(direction), batched(batched) {
                    //  std::cout << "Creating parameter " << type.name << " " << name << " of " << size << " bytes" << (batched ? " [batched]" : "") << std::endl;
                    };
            virtual ~BaseParameter() { }

            virtual std::string toString() {
                return this->type.getFullName() + " " + name;
            }
            virtual std::string getNonPointerTypeName() {
                auto type = this->type.getFullName();
                if (type.back() == '*') { // Should we check isPointer instead?
                    type.pop_back();
                }
                return type;
            }
            virtual bool isComplete() {
                return this->complete;
            }
            virtual void setComplete(bool complete) {
                this->complete = complete;
            }
            virtual bool isBatched() {
                return this->batched;
            }
            virtual bool isConstant() {
                return type.isConst;
            }
            virtual bool isIn() {
                return this->direction == GSPAR_PARAM_IN || this->direction == GSPAR_PARAM_INOUT;
            }
            virtual bool isOut() {
                return this->direction == GSPAR_PARAM_OUT || this->direction == GSPAR_PARAM_INOUT;
            }
            /**
             * Returns the parameter type for use inside the kernel
             */
            virtual std::string toKernelParameter() {
                std::string type = this->type.getFullName();
                if (this->isBatched() && paramValueType == GSPAR_PARAM_VALUE) {
                    // A batched parameter is a pointer of values.
                    // If it's a PointerParameter, we already ripped off the extra * and will flatten the pointers.
                    // If it's a ValueParameter, we need to add an extra * (we will use a pointer of values)
                    type += "*";
                }
                return type + " " + this->getKernelParameterName();
            }
            virtual std::string getKernelParameterName() {
                return (this->isBatched() ? "gspar_batched_" : "") + this->name;
            }
            virtual bool isValueTyped() = 0;
        };

        template<class T>
        class TypedParameter;

        class ValueParameter;

        class PointerParameter;

        class BaseParallelPattern;

    }
}

#include "GSPar_Base.hpp"
#include "GSPar_BaseGPUDriver.hpp"

namespace GSPar {
    namespace Pattern {

        // TODO this specialized classes are completely useless. We could work all out with only BaseParameter and it would be far simpler

        /**
         * A pattern typed parameter
         */
        template<class T>
        class TypedParameter
            : public BaseParameter {
        protected:
            T value;
            std::unique_ptr<Driver::BaseMemoryObjectBase> memoryObject;
            Driver::BaseMemoryObjectBase* userMemoryObject = nullptr; // MemoryObject from user
        public:
            size_t numberOfElements;

            TypedParameter() { }
            TypedParameter(std::string name, VarType type, size_t size, T value,
                    ParameterValueType paramValueType, ParameterDirection direction = GSPAR_PARAM_IN, bool batched = false) :
                    BaseParameter(name, type, size, paramValueType, direction, batched), value(value) { };
            virtual ~TypedParameter() { }

            virtual Driver::BaseMemoryObjectBase *getMemoryObject() {
                if(userMemoryObject != nullptr){
                    return this->userMemoryObject;
                }
                return this->memoryObject.get();
            }

            virtual void setUserMemoryObject(Driver::BaseMemoryObjectBase* memoryObjectFromUser) {
                this->userMemoryObject = memoryObjectFromUser;
            }

            // virtual T getValue() { return this->value; }
        };

        /**
         * A value parameter for pattern
         */
        class ValueParameter
            : public TypedParameter<void*> {
        public:
            ValueParameter() : TypedParameter() { }
            ValueParameter(std::string name, VarType type, size_t size, void *value, ParameterDirection direction = GSPAR_PARAM_IN, bool batched = false) :
                TypedParameter(name, type, size, value, ParameterValueType::GSPAR_PARAM_VALUE, direction, batched) {
                    if (value == nullptr) { // It may be just a placeholder
                        this->complete = false;
                    }
                };
            virtual ~ValueParameter() { }

            virtual bool isValueTyped() override { return true; }
            virtual void* getPointer() { return this->value; }

            template <class TDevice>
            Driver::BaseMemoryObjectBase *malloc(TDevice gpu, unsigned int batchSize) {
                if (this->isBatched()) {
                    // By default, it is a read-only parameter
                    this->memoryObject = std::unique_ptr<Driver::BaseMemoryObjectBase>(gpu->malloc(batchSize * this->size, this->getPointer(), true, false));
                }
                // If it is a non-batched ValueParameter, we return a nullptr
                return this->memoryObject.get();
            }
        };

        /**
         * A pointer parameter for pattern
         */
        class PointerParameter
            : public TypedParameter<void*> {
        public:
            PointerParameter() : TypedParameter() { }
            // Constructor with no MemoryObject from user
            PointerParameter(std::string name, VarType type, size_t size, void *value, ParameterDirection direction = GSPAR_PARAM_IN, bool batched = false) :
                    TypedParameter(name, type, size, value, ParameterValueType::GSPAR_PARAM_POINTER, direction, batched) {
                if (!value) { // It is just a placeholder
                    this->complete = false;
                }
            };
            // Constructor with MemoryObject from user
            PointerParameter(std::string name, VarType type, Driver::BaseMemoryObjectBase* userMemoryObject, ParameterDirection direction = GSPAR_PARAM_IN, bool batched =false) :
                    TypedParameter(name, type, userMemoryObject->getSize(), userMemoryObject->getHostPointer(), ParameterValueType::GSPAR_PARAM_POINTER, direction, batched) {
                this->setUserMemoryObject(userMemoryObject);
            };
            virtual ~PointerParameter() { }

            virtual bool isValueTyped() override { return false; }
            virtual void* getPointer() { return this->value; }

            template <class TDevice>
            Driver::BaseMemoryObjectBase *malloc(TDevice gpu, unsigned int batchSize) {
                // If it is only IN, the kernel won't write, if is OUT, the kernel won't read
                bool readOnly = (this->direction == Pattern::ParameterDirection::GSPAR_PARAM_IN);
                bool writeOnly = (this->direction == Pattern::ParameterDirection::GSPAR_PARAM_OUT);
                if (this->isBatched()) {
                    // A batched PointerParameter is conversible to void**
                    this->memoryObject = std::unique_ptr<Driver::BaseMemoryObjectBase>(gpu->mallocChunked(batchSize, this->size, (void**)this->getPointer(), readOnly, writeOnly));
                } else {
                    this->memoryObject = std::unique_ptr<Driver::BaseMemoryObjectBase>(gpu->malloc(this->size, this->getPointer(), readOnly, writeOnly));
                }
                return this->memoryObject.get();
            }
        };

        /**
         * Base class for parallel patterns
         */
        class BaseParallelPattern {
        private:
            unsigned int gpuIndex = 0;
            Driver::BaseDeviceBase* gpuDevice = nullptr;

        protected:
            std::unique_ptr<Driver::BaseExecutionFlowBase> executionFlow;
            bool batched = false;
            unsigned int batchSize = 1; //TODO what if Dimension max is not divisible by batchSize? It actually segfaults
            bool _isKernelCompiled = false;
            bool isKernelStale = false; // Do we need to recompile the kernel?
            mutable std::mutex compiledKernelMutex;
            // Should we use a std::map to support multiple pre-compiled kernels?
            Driver::Dimensions compiledKernelDimension;
            std::shared_ptr<Driver::BaseKernelBase> compiledKernel;
            std::string kernelName;
            std::string userKernel;
            std::string extraKernelCode;
            std::vector<std::string> paramsOrder;
            // Set the thread block size (it is an optional paramenter) #gabriell
            int numThreadsPerBlock[3] = {0, 0, 0};
            /**
             * We use a shared_ptr of parameters, so they can be safely cloned together with the Pattern
             * And they'll be automatically released as soon as all clones are destroyed
             */
            std::map<std::string, std::shared_ptr<BaseParameter>> params;
            std::array<std::string, 3> stdVarNames;
            bool useSharedMemory = false;
            mutable std::mutex sharedMemoryParameterMutex;
            PointerParameter* sharedMemoryParameter = nullptr;

            // Parameters

            /**
             * Get the type (as string) of the template argument
             * from https://stackoverflow.com/a/20170989/
             */
            template <typename T>
            VarType getTemplatedType() {
                typedef typename std::remove_reference<T>::type TR;
                std::unique_ptr<char, void(*)(void*)> own (
                #ifndef _MSC_VER
                    abi::__cxa_demangle(typeid(TR).name(), nullptr, nullptr, nullptr),
                #else
                    nullptr,
                #endif
                    std::free
                );
                VarType varType;
                varType.name = own != nullptr ? own.get() : typeid(TR).name();
                varType.isPointer = std::is_pointer<TR>::value;
                if (varType.isPointer) {
                    typedef typename std::remove_pointer<TR>::type TNoPtr;
                    varType.isClass = std::is_class<TNoPtr>::value;
                } else {
                    varType.isClass = std::is_class<TR>::value;
                }
                varType.isConst = std::is_const<TR>::value;
                varType.isVolatile = std::is_volatile<TR>::value;
                varType.isLValueRef = std::is_lvalue_reference<T>::value;
                if (!varType.isLValueRef) { // Can't be both
                    varType.isRValueRef = std::is_rvalue_reference<T>::value;
                }
                return varType;
            }

            virtual void setPointerParameter(std::string name, VarType type, size_t size, void *value, ParameterDirection direction = GSPAR_PARAM_IN, bool batched = false) {
                // GSPAR_PARAM_PRESENT is incorrect when using a host pointer instead of a MemoryObject 
                if (direction == GSPAR_PARAM_PRESENT) {
                    throw GSParException("Pattern parameter \"" + name + "\": GSPAR_PARAM_PRESENT is only allowed when a MemoryObject is provided");
                }
                std::shared_ptr<BaseParameter> parameter(new PointerParameter(name, type, size, value, direction, batched));
                this->setParameter(parameter);
            }
            // Using MemoryObject from user
            virtual void setPointerParameter(std::string name, VarType type, Driver::BaseMemoryObjectBase* userMemoryObject, ParameterDirection direction = GSPAR_PARAM_IN, bool batched = false) {
                // new PointParameter with MemoryObject from user
                std::shared_ptr<BaseParameter> parameter(new PointerParameter(name, type, userMemoryObject, direction, batched));
                this->setParameter(parameter);
            }
            virtual void setValueParameter(std::string name, VarType type, size_t size, void *value, ParameterDirection direction = GSPAR_PARAM_IN, bool batched = false) {
                std::shared_ptr<BaseParameter> parameter(new ValueParameter(name, type, size, value, direction, batched));
                this->setParameter(parameter);
            }
            virtual void setParameter(std::shared_ptr<BaseParameter> parameter) {
                // std::cout << "Setting BaseParameter " << parameter->type.getFullName() << " " << parameter->name << " of " << parameter->size << " bytes" << (parameter->isBatched() ? " [batched]" : "") << std::endl;
                auto paramName = parameter.get()->name;
                if (std::find(this->paramsOrder.begin(), this->paramsOrder.end(), paramName) == this->paramsOrder.end()) {
                    this->paramsOrder.push_back(paramName);
                    this->isKernelStale = true; // There is a new parameter, we need to recompile the kernel
                }
                this->params[paramName] = parameter;
            }

            template<class TDriverInstance>
            decltype(TDriverInstance::getExecutionFlowType())* getExecutionFlow() {
                return dynamic_cast<decltype(TDriverInstance::getExecutionFlowType())*>(this->executionFlow.get());
            }

            // Main run function for Parallel Pattern
            template<class TDriverInstance>
            void run(Driver::Dimensions pDims, bool useCompiledDim) {
                Driver::Dimensions dimsToUse = useCompiledDim ? this->compiledKernelDimension : pDims;
                if (!dimsToUse.getCount()) {
                    throw GSParException("No dimensions set to run the pattern");
                }
                #ifdef GSPAR_DEBUG
                    std::stringstream ss;
                #endif

                // TODO validade if dimsToUse is valid

                Driver::Dimensions dimsToRun = dimsToUse;
                if (this->isBatched()) {
                    dimsToRun *= this->batchSize;
                    #ifdef GSPAR_DEBUG
                        ss << "[" << std::this_thread::get_id() << " GSPar Pattern "<<this<<"] Batched pattern, asked for " << dimsToUse.toString() << " * ";
                        ss << this->batchSize << " batch size, so we'll run for " << dimsToRun.toString() << std::endl;
                        std::cout << ss.str();
                        ss.str("");
                    #endif
                }

                this->compile<TDriverInstance>(dimsToUse);

                // #ifdef GSPAR_DEBUG
                //     auto gpu = this->getGpu<TDriverInstance>();
                //     ss << "[" << std::this_thread::get_id() << " GSPar Pattern "<<this<<"] Working with GPU " << gpu << " - " << gpu->getName() << std::endl;
                //     std::cout << ss.str();
                //     ss.str("");
                // #endif

                auto kernel = this->getCompiledKernel<TDriverInstance>();
                kernel->clearParameters();

                // Set the thread block size (it is an optional paramenter)
                if (numThreadsPerBlock[0] != 0) {
                    kernel->setNumThreadsPerBlockForX(numThreadsPerBlock[0]);
                }
                if (numThreadsPerBlock[1] != 0) {
                    kernel->setNumThreadsPerBlockForY(numThreadsPerBlock[1]);
                }
                if (numThreadsPerBlock[2] != 0) {
                    kernel->setNumThreadsPerBlockForZ(numThreadsPerBlock[2]);
                }

                this->callbackBeforeAllocatingMemoryOnGpu(dimsToUse, kernel);

                this->mallocParametersInGpu<TDriverInstance>();

                this->copyParametersFromHostToGpuAsync<TDriverInstance>();

                this->setSharedMemoryInKernel<TDriverInstance>(kernel, dimsToUse);

                this->setParametersInKernel<TDriverInstance>(kernel, dimsToUse);

                this->callbackAfterCopyDataFromHostToGpu();
                this->callbackBeforeRunInGpu();

                auto executionFlow = this->getExecutionFlow<TDriverInstance>();

                #ifdef GSPAR_DEBUG
                    ss << "[" << std::this_thread::get_id() << " GSPar Pattern "<<this<<"] Running kernel " << kernel << " for " << dimsToRun.toString() << " in flow " << executionFlow << std::endl;
                    std::cout << ss.str();
                    ss.str("");
                #endif

                kernel->runAsync(dimsToRun, executionFlow);

                #ifdef GSPAR_DEBUG
                    ss << "[" << std::this_thread::get_id() << " GSPar Pattern "<<this<<"] Started running kernel " << kernel << " in flow " << executionFlow << std::endl;
                    std::cout << ss.str();
                    ss.str("");
                #endif

                kernel->waitAsync();

                #ifdef GSPAR_DEBUG
                    ss << "[" << std::this_thread::get_id() << " GSPar Pattern "<<this<<"] Finished running kernel " << kernel << " in flow " << executionFlow << std::endl;
                    std::cout << ss.str();
                    ss.str("");
                #endif

                this->callbackAfterRunInGpu();

                this->copyParametersFromGpuToHostAsync<TDriverInstance>();

                this->callbackAfterCopyDataFromGpuToHost(dimsToUse, kernel);

                #ifdef GSPAR_DEBUG
                    ss << "[" << std::this_thread::get_id() << " GSPar Pattern "<<this<<"] Finished running pattern" << std::endl;
                    std::cout << ss.str();
                    ss.str("");
                #endif
            }

        public:
            BaseParallelPattern() { }
            BaseParallelPattern(std::string kernelSource) : userKernel(kernelSource) { }
            // This constructor gets called instead of copy assignment when assignin directly or passing values to function
            BaseParallelPattern(const BaseParallelPattern &other) {
                other.cloneIntoNonTemplated(this);
            };
            
            virtual ~BaseParallelPattern() { }

            virtual bool isBatched() {
                return this->batched;
            }

            virtual BaseParallelPattern& setBatchSize(unsigned int batchSize) {
                if (!batchSize) { // Set not batched
                    if (this->isBatched()) {
                        // The pattern was batched and now it isn't, we need to recompile the kernel
                        this->isKernelStale = true;
                    }
                    this->batched = false;
                } else {
                    if (!this->isBatched()) {
                        // The pattern wasn't batched and now it is, we need to recompile the kernel
                        this->isKernelStale = true;
                    }
                    this->batched = true;
                }
                this->batchSize = batchSize;
                return *this;
            }

            // TODO support using GPUs based on some scheduler (round-robin, etc)
            virtual void setGpuIndex(unsigned int index) {
                if (this->gpuIndex != index) {
                    this->isKernelStale = true; // If the GPU changed, we need to recompile the kernel
                    this->gpuDevice = nullptr;
                    this->executionFlow.reset();
                    this->gpuIndex = index;
                }
            }
            virtual unsigned int getGpuIndex() {
                return this->gpuIndex;
            }

            template<class TDriverInstance>
            void cloneInto(BaseParallelPattern* other) const {
                this->cloneIntoNonTemplated(other);
                // Clone templated values
                other->setGpu<TDriverInstance>((decltype(TDriverInstance::getDeviceType())*)this->gpuDevice);
                // executionFlow is not copied, each instance uses it's own. setGpu call initializes it also
                
                if (this->_isKernelCompiled && !this->isKernelStale) { // We only copy the kernel if it's a valid (and usable) one
                    std::lock_guard<std::mutex> lock(other->compiledKernelMutex); // Auto-unlock, RAII
                    other->_isKernelCompiled = this->_isKernelCompiled;
                    other->isKernelStale = this->isKernelStale;
                    // compiledKernelMutex is (quite obviously) unique for each instance
                    if (this->compiledKernelDimension.getCount()) {
                        Driver::Dimensions compiledKernelDimension = this->compiledKernelDimension;
                        other->compiledKernelDimension = compiledKernelDimension;
                    }
                    if (this->compiledKernel.get()) {
                        other->compiledKernel = std::shared_ptr<decltype(TDriverInstance::getKernelType())>(new decltype(TDriverInstance::getKernelType())());
                        auto localKernel = this->getCompiledKernel<TDriverInstance>();
                        localKernel->cloneInto(other->compiledKernel.get());
                    }
                }
            }

            void cloneIntoNonTemplated(BaseParallelPattern* other) const {
                // Clone
                other->gpuIndex = this->gpuIndex;
                other->batched = this->batched;
                other->batchSize = this->batchSize;
                other->kernelName = this->kernelName;
                other->userKernel = this->userKernel;
                other->extraKernelCode = this->extraKernelCode;
                other->paramsOrder = this->paramsOrder;
                other->params = this->params;
                other->stdVarNames = this->stdVarNames;
                other->useSharedMemory = this->useSharedMemory;
                other->sharedMemoryParameter = this->sharedMemoryParameter;
            }

            template<class TDriverInstance>
            BaseParallelPattern& setCompiledKernel(decltype(TDriverInstance::getKernelType())* kernel, Driver::Dimensions dims) {
                std::lock_guard<std::mutex> lock(this->compiledKernelMutex); // Auto-unlock, RAII
                this->compiledKernel = std::shared_ptr<Driver::BaseKernelBase>(kernel);
                this->compiledKernelDimension = dims;
                this->_isKernelCompiled = true;
                this->isKernelStale = false;
                return *this;
                // Auto-unlock of compiledKernelMutex, RAII
            }

            template<class TDriverInstance>
            decltype(TDriverInstance::getKernelType())* getCompiledKernel() const {
                return static_cast<decltype(TDriverInstance::getKernelType())*>(this->compiledKernel.get());
            }

            template<class TDriverInstance>
            void setGpu(decltype(TDriverInstance::getDeviceType())* device) {
                if (this->gpuDevice != device) {
                    this->gpuDevice = device;
                    auto executionFlow = new decltype(TDriverInstance::getExecutionFlowType())(device);
                    executionFlow->start();
                    this->executionFlow = std::unique_ptr<decltype(TDriverInstance::getExecutionFlowType())>(executionFlow);
                }
            }
            template<class TDriverInstance>
            decltype(TDriverInstance::getDeviceType())* getGpu() {
            // Driver::BaseDeviceBase* getGpu() {
                if (this->gpuDevice == nullptr) {
                    TDriverInstance* driver = TDriverInstance::getInstance();
                    // Driver::CUDA::Instance driver = TDriverInstance::getInstance(); //Provides autocomplete
                    driver->init();

                    if (driver->getGpuCount() == 0) {
                        return nullptr;
                    }

                    auto gpu = driver->getGpu(this->gpuIndex);
                    this->setGpu<TDriverInstance>(gpu);
                }
                return (decltype(TDriverInstance::getDeviceType())*)this->gpuDevice;
            }

            virtual BaseParallelPattern& addExtraKernelCode(std::string extraKernelCode) {
                this->extraKernelCode += extraKernelCode;
                this->isKernelStale = true; // The kernel code changed, we need to recompile it
                return *this;
            }

            virtual std::pair<std::string, std::string> generateDefaultControlIf(Driver::Dimensions dims, std::array<std::string, 3> stdVarNames) {
                std::string r = "if (";
                for(int d = 0; d < SUPPORTED_DIMS; d++) {
                    if (dims[d]) {
                        if (this->isBatched()) {
                            r += "(gspar_batch_" + stdVarNames[d] + " < gspar_batch_size)&&";
                        }
                        r += "(" + stdVarNames[d] + " < gspar_max_" + stdVarNames[d] + ")&&";
                    }
                }
                // Removes last &&
                r.pop_back();
                r.pop_back();
                r += ") {\n";
                return std::make_pair(r, "}");
            }

            template<class TDriverInstance>
            std::string generateKernelSource(Driver::Dimensions dims) {

                auto codeGenerator = TDriverInstance::getInstance()->getKernelGenerator();
                std::string kernelName = this->getKernelName();

                std::pair<std::string, std::string> ifDimensions = this->generateDefaultControlIf(dims, codeGenerator->getStdVarNames(this->stdVarNames));

                return (!this->extraKernelCode.empty() ? this->extraKernelCode + "\n" : "")
                    + codeGenerator->getKernelPrefix() + " " + kernelName + "("
                    + codeGenerator->generateParams(this, dims) + ") {\n"
                    + codeGenerator->generateInitKernel(this, dims) + "\n"
                    + codeGenerator->generateStdVariables(this, dims)
                    + codeGenerator->generateBatchedParametersInitialization(this, dims) + "\n"
                    + ifDimensions.first
                    + this->getKernelCore(dims, codeGenerator->getStdVarNames(this->stdVarNames))
                    + "\n" + ifDimensions.second + "\n" // if (dims)
                    + "}\n"; // kernel
            }

            virtual std::string getKernelName() {
                if (this->kernelName.empty()) {
                    this->kernelName = "gspar_kernel_" + getRandomString(7);
                }
                return this->kernelName;
            }

            virtual void setKernelName(std::string kernelName) {
                this->kernelName = kernelName;
            }

            std::array<std::string, 3>& getStdVarNames() {
                return this->stdVarNames;
            }
            BaseParallelPattern& setStdVarNames(std::array<std::string, 3> names) {
                this->stdVarNames = names;
                this->isKernelStale = true; // The kernel code changed, we need to recompile it
                // TODO should we check if the names really changed?
                return *this;
            }

            virtual std::string getKernelCore(Driver::Dimensions dims, std::array<std::string, 3> stdVarNames) {
                return std::string(this->getUserKernel());
            }
            std::string getUserKernel() {
                return userKernel;
            }

            bool isUsingSharedMemory() {
                return this->useSharedMemory;
            }
            virtual PointerParameter* generateSharedMemoryParameter(Driver::Dimensions dims, Driver::BaseKernelBase *kernel) {
                return this->getSharedMemoryParameter();
            }
            virtual PointerParameter* getSharedMemoryParameter() {
                return this->sharedMemoryParameter;
            }
            BaseParameter* getParameter(std::string name) {
                auto it = this->params.find(name);
                if (it == this->params.end()) {
                    return nullptr;
                }
                return it->second.get();
            }
            virtual std::vector<BaseParameter*> getParameterList() {
                std::vector<BaseParameter*> paramList;
                for (auto &paramName : this->paramsOrder) {
                    paramList.push_back(this->getParameter(paramName));
                }
                return paramList;
            }
            // Set the thread block size (it is an optional paramenter) #gabriell
            virtual BaseParallelPattern& setNumThreadsPerBlockForX(unsigned long num) {
                return this->setNumThreadsPerBlockFor(0, num);
            }
            virtual BaseParallelPattern& setNumThreadsPerBlockForY(unsigned long num) {
                return this->setNumThreadsPerBlockFor(1, num);
            }
            virtual BaseParallelPattern& setNumThreadsPerBlockForZ(unsigned long num) {
                return this->setNumThreadsPerBlockFor(2, num);
            }
            virtual BaseParallelPattern& setNumThreadsPerBlockFor(int dim, unsigned long num) {
                this->numThreadsPerBlock[dim] = num;
                return *this;
            }
            virtual BaseParallelPattern& setNumThreadsPerBlock(unsigned long numX, unsigned long numY, unsigned long numZ) {
                this->numThreadsPerBlock[0] = numX;
                this->numThreadsPerBlock[1] = numY;
                this->numThreadsPerBlock[2] = numZ;
                return *this;
            }

            /**
             * Parameter placeholder
             */
            template <typename T>
            BaseParallelPattern& setParameterPlaceholder(std::string name, ParameterValueType parameterType = GSPAR_PARAM_POINTER, ParameterDirection direction = GSPAR_PARAM_IN, bool batched = false) {
                VarType varType = getTemplatedType<T>();
                if (parameterType == ParameterValueType::GSPAR_PARAM_POINTER) {
                    this->setPointerParameter(name, varType, 0, nullptr, direction, batched);
                } else if (parameterType == ParameterValueType::GSPAR_PARAM_VALUE) {
                    this->setValueParameter(name, varType, sizeof(T), nullptr, direction, batched);
                }
                if (batched) {
                    this->batched = true;
                }
                return *this;
            }

            /**
             * Pointer parameters
             */
            template <typename T>
            BaseParallelPattern& setParameter(std::string name, size_t size, T* value, ParameterDirection direction = GSPAR_PARAM_IN) {
                VarType varType = getTemplatedType<decltype(value)>();
                this->setPointerParameter(name, varType, size, value, direction);
                return *this;
            }
            template <typename T>
            BaseParallelPattern& setParameter(std::string name, size_t size, const T* value) {
                // Can't call setParameter(non-const T) because getTemplatedType would lost const information
                VarType varType = getTemplatedType<decltype(value)>();
                // A const parameter must be IN, as it can't be modified
                this->setPointerParameter(name, varType, size, const_cast<T*>(value), GSPAR_PARAM_IN);
                return *this;
            }
            // Using MemoryObject from user
            template <typename T>
            BaseParallelPattern& setParameter(std::string name, Driver::BaseMemoryObjectBase* userMemoryObject, ParameterDirection direction = GSPAR_PARAM_IN) {
                VarType varType = getTemplatedType<T>();
                this->setPointerParameter(name, varType, userMemoryObject, direction);
                return *this;
            }

            /**
             * Value parameters
             */
            template <typename T>
            BaseParallelPattern& setParameter(std::string name, T value) {
                VarType varType = getTemplatedType<decltype(value)>();
                // We need a pointer, so we allocate memory and copy the value
                T* value_copy = new T;
                *value_copy = value;
                // A value parameter must be IN, as it can't be modified
                this->setValueParameter(name, varType, sizeof(T), value_copy, GSPAR_PARAM_IN);
                return *this;
            }

            /**
             * Batched (pointer and value) parameters
             */
            template <typename T>
            BaseParallelPattern& setBatchedParameter(std::string name, size_t sizeOfEachBatch, T** value, ParameterDirection direction = GSPAR_PARAM_IN) {
                this->batched = true;
                VarType varType = getTemplatedType<decltype(value)>();
                varType.name.pop_back(); // We receive ** due to the batch. So the kernel type is only * (we flatten the pointers)
                this->setPointerParameter(name, varType, sizeOfEachBatch, value, direction, true);
                return *this;
            }
            template <typename T>
            BaseParallelPattern& setBatchedParameter(std::string name, size_t sizeOfEachBatch, const T** value) {
                // Can't call setBatchedParameter(non-const T) because getTypeName would lost const information
                this->batched = true;
                VarType varType = getTemplatedType<decltype(value)>();
                varType.name.pop_back(); // We receive ** due to the batch. So the kernel type is only * (we flatten the pointers)
                // A const parameter must be IN, as it can't be modified
                this->setPointerParameter(name, varType, sizeOfEachBatch, const_cast<T**>(value), GSPAR_PARAM_IN, true);
                return *this;
            }
            template <typename T>
            BaseParallelPattern& setBatchedParameter(std::string name, const T* value) {
                this->batched = true;
                VarType varType = getTemplatedType<decltype(value)>();
                varType.name.pop_back(); // We receive * due to the batch.
                // The effective kernel type is a pure value, but for the parameters we still need it to be a pointer (check BaseParameter::toKernelParameter).
                this->setValueParameter(name, varType, sizeof(T), const_cast<T*>(value), GSPAR_PARAM_IN, true);
                return *this;
            }

            virtual bool isKernelCompiledFor(Driver::Dimensions dims) {
                // We only compile if the kernel wasn't compiled yet and the configuration didn't change
                return this->_isKernelCompiled && !this->isKernelStale &&
                    // TODO #10 Do we really need the exact same dimension? The sizes are passed in parameters.
                    this->compiledKernelDimension == dims;
            }

            /**
             * Compiles the pattern (including the generation and compilation of the GPU kernel) for the dims Dimensions.
             * 
             * @param <TDriverInstance> Type of the specialized BaseInstance class
             * @param dims The Dimensions for which the pattern should be compiled
             */
            template<class TDriverInstance>
            BaseParallelPattern& compile(Driver::Dimensions dims) {
                // We only compile if the kernel wasn't compiled yet and the configuration didn't change
                if (this->isKernelCompiledFor(dims)) {
                    return *this;
                }
                std::lock_guard<std::mutex> lock(this->compiledKernelMutex); // Auto-unlock, RAII
                #ifdef GSPAR_DEBUG
                    std::stringstream ss;
                    ss << "[" << std::this_thread::get_id() << " GSPar "<<this<<"] Compiling Kernel for ParallelPattern with " << dims.toString() << std::endl;
                    std::cout << ss.str();
                    ss.str("");
                #endif

                auto gpu = this->getGpu<TDriverInstance>();
                if (gpu == nullptr) {
                    throw GSParException("No GPU found for Pattern compilation");
                }

                std::string kernelName = this->getKernelName();

                this->callbackBeforeGeneratingKernelSource();

                std::string kernelSource = this->generateKernelSource<TDriverInstance>(dims);

                #ifdef GSPAR_DEBUG
                    ss << "[" << std::this_thread::get_id() << " GSPar "<<this<<"] Compiling kernel source for " << kernelName << ":" << std::endl;
                    ss << kernelSource << std::endl;
                    std::cout << ss.str();
                    ss.str("");
                #endif
                // this->compiledKernel = std::unique_ptr<void, void(*)(void*)>{
                //     (void*)(gpu->prepareKernel(kernelSource.c_str(), kernelName.c_str())),
                //     [](void *ptr) { delete static_cast<decltype(TDriverInstance::getKernelType())*>(ptr); }
                // };
                auto kernel = gpu->prepareKernel(kernelSource.c_str(), kernelName.c_str());
                this->compiledKernel = std::shared_ptr<Driver::BaseKernelBase>(kernel);
                this->compiledKernelDimension = dims;
                this->_isKernelCompiled = true;
                this->isKernelStale = false;
                return *this;
                // Auto-unlock of compiledKernelMutex, RAII
            }

            // TODO most of the following functions should have protected visibility

            /**
             * Set shared memory allocation in kernel object
             * 
             * @param <TDriverInstance> Type of the specialized BaseInstance class
             * @param kernel The kernel on which the shared memory will be configured
             * @param dims The Dimensions for which the shared memory will be configured
             */
            template<class TDriverInstance>
            void setSharedMemoryInKernel(decltype(TDriverInstance::getKernelType())* kernel, Driver::Dimensions dims) {
                if (!this->isUsingSharedMemory()) {
                    return;
                }
                #ifdef GSPAR_DEBUG
                    std::stringstream ss;
                    ss << "[" << std::this_thread::get_id() << " GSPar Pattern "<<this<<"] Is using shared memory, generating it in kernel " << kernel << std::endl;
                    std::cout << ss.str();
                    ss.str("");
                #endif
                auto shmemParam = this->generateSharedMemoryParameter(dims, kernel);
                kernel->setSharedMemoryAllocation(shmemParam->size);
            }

            /**
             * Allocates memory in GPU device for this pattern's parameters
             * 
             * @param <TDriverInstance> Type of the specialized BaseInstance class
             */
            template<class TDriverInstance>
            void mallocParametersInGpu() {
                auto device = this->getGpu<TDriverInstance>();
                if (device == nullptr) {
                    throw GSParException("No GPU found to allocate memory for parameters for Pattern");
                }
                for (auto &paramName : this->paramsOrder) {
                    auto param = this->getParameter(paramName);
                    if (!param || !param->isComplete()) {
                        throw GSParException("Pattern parameter \"" + param->name + "\" is just a placeholder. The parameter list must be complete to run the parallel pattern.");
                    }
                    if (param->paramValueType == Pattern::ParameterValueType::GSPAR_PARAM_POINTER) { // It is a PointerParameter
                        auto paramPointer = static_cast<Pattern::PointerParameter*>(param);
                        if (paramPointer->getMemoryObject() == nullptr) { // It returns a MemoryObject from user, if available
                            paramPointer->malloc(device, this->batchSize); //TODO check if the batchSize changed since the last parameter allocation
                            #ifndef GSPAR_PATTERN_DISABLE_PINNED_MEMORY
                                // In some cases, copyInAsync fails with CUDA_ERROR_INVALID_VALUE: invalid argument. According to the docs:
                                //   Memory regions requested must be either entirely registered with CUDA, or in the case of host pageable transfers, not registered at all.
                                //   Memory regions spanning over allocations that are both registered and not registered with CUDA are not supported and will return CUDA_ERROR_INVALID_VALUE.
                                // We confirmed that avoiding pinned memory eliminates the failure, but we are still unsure why it happens
                                if (paramPointer->direction == GSPAR_PARAM_INOUT || paramPointer->direction == GSPAR_PARAM_OUT) {
                                    // Pinned memory allows for memory operations overlapping in CUDA
                                    if (paramPointer->isBatched()) {
                                        auto chunkedMemObj = dynamic_cast<decltype(TDriverInstance::getChunkedMemoryObjectType())*>(paramPointer->getMemoryObject());
                                        chunkedMemObj->pinHostMemory();
                                    } else {
                                        auto singleMemObj = dynamic_cast<decltype(TDriverInstance::getMemoryObjectType())*>(paramPointer->getMemoryObject());
                                        singleMemObj->pinHostMemory();
                                    }
                                }
                            #endif
                        }
                    } else if (param->paramValueType == Pattern::ParameterValueType::GSPAR_PARAM_VALUE) {
                        auto paramValue = static_cast<Pattern::ValueParameter*>(param);
                        if (paramValue->getMemoryObject() == nullptr) {
                            paramValue->malloc(device, this->batchSize);
                        }
                    }
                }
            }

            /**
             * Copies IN and INOUT parameters from host to device (asynchronously)
             * 
             * @param <TDriverInstance> Type of the specialized BaseInstance class
             */
            template<class TDriverInstance>
            void copyParametersFromHostToGpuAsync() {
                #ifdef GSPAR_DEBUG
                    std::stringstream ss; ss.str("");
                #endif
                // We use the same execution flow as the kernel itself, so we don't need to wait the async copies to finish
                // Waiting the async copies to finish causes OpenCL to hang (possibly a deadlock?)
                auto executionFlow = this->getExecutionFlow<TDriverInstance>();

                for (auto &paramName : this->paramsOrder) {
                    auto param = this->getParameter(paramName);
                    if (param && param->isIn()) {
                        if (param->paramValueType == Pattern::ParameterValueType::GSPAR_PARAM_POINTER) {
                            auto paramPointer = static_cast<Pattern::PointerParameter*>(param);
                            #ifdef GSPAR_DEBUG
                                ss << "[" << std::this_thread::get_id() << " GSPar Pattern "<<this<<"] Copying " << param->name << " (" << paramPointer->getMemoryObject() << ") to GPU in flow " << executionFlow << std::endl;
                                std::cout << ss.str();
                                ss.str("");
                            #endif
                            if (param->isBatched()) {
                                auto chunkedMemObj = dynamic_cast<decltype(TDriverInstance::getChunkedMemoryObjectType())*>(paramPointer->getMemoryObject());
                                if (this->batchSize != chunkedMemObj->getChunkCount()) {
                                    // The pattern batch size changed from when the parameter was created.
                                    // If it is lower than the parameter batch size, we copy only the related chunks
                                    // TODO what if it is higher?
                                    for (unsigned int c = 0; c < this->batchSize; c++) {
                                        chunkedMemObj->copyInAsync(c, executionFlow);
                                    }
                                } else {
                                    chunkedMemObj->copyInAsync(executionFlow); // Copy all the chunks
                                }
                            } else {
                                auto singleMemObj = dynamic_cast<decltype(TDriverInstance::getMemoryObjectType())*>(paramPointer->getMemoryObject());
                                singleMemObj->copyInAsync(executionFlow);
                            }
                        } else if (param->paramValueType == Pattern::ParameterValueType::GSPAR_PARAM_VALUE) {
                            if (param->isBatched()) {
                                auto paramValue = static_cast<Pattern::ValueParameter*>(param);
                                auto memObj = dynamic_cast<decltype(TDriverInstance::getMemoryObjectType())*>(paramValue->getMemoryObject());
                                memObj->copyInAsync(executionFlow);
                            }
                        }
                    }
                }
            }

            template<class TDriverInstance>
            void copyParametersFromGpuToHostAsync() {
                // #ifdef GSPAR_DEBUG
                //     std::stringstream ss;
                // #endif
                for (auto& paramName : this->paramsOrder) {
                    // #ifdef GSPAR_DEBUG
                    //     ss << "[GSPar Pattern "<<this<<"] Copying parameter " << paramName << " from GPU to host" << std::endl;
                    //     std::cout << ss.str();
                    //     ss.str("");
                    // #endif
                    auto param = this->getParameter(paramName);
                    if (param && param->isOut() && param->paramValueType == Pattern::ParameterValueType::GSPAR_PARAM_POINTER) {
                        auto paramPointer = static_cast<Pattern::PointerParameter*>(param);
                        // TODO copy async
                        // memObj->copyOutAsync();
                        // std::cout << "Asking to copy " << param->name << " back from GPU" << std::endl;
                        if (param->isBatched()) {
                            auto chunkedMemObj = dynamic_cast<decltype(TDriverInstance::getChunkedMemoryObjectType())*>(paramPointer->getMemoryObject());
                            if (this->batchSize != chunkedMemObj->getChunkCount()) {
                                // The pattern batch size changed from when the parameter was created.
                                // If it is lower than the parameter batch size, we copy only the related chunks
                                // TODO what if it is higher?
                                for (unsigned int c = 0; c < this->batchSize; c++) {
                                    chunkedMemObj->copyOut(c);
                                }
                            } else {
                                chunkedMemObj->copyOut(); // Copy all the chunks
                            }
                        } else {
                            auto singleMemObj = dynamic_cast<decltype(TDriverInstance::getMemoryObjectType())*>(paramPointer->getMemoryObject());
                            if (singleMemObj) {
                                singleMemObj->copyOut();
                            }
                        }
                    }
                }
            }

            template<class TDriverInstance>
            void setParametersInKernel(decltype(TDriverInstance::getKernelType())* kernel, Driver::Dimensions dims) {
                this->setDimsParametersInKernel<TDriverInstance>(kernel, dims);
                
                if (this->isBatched()) {
                    kernel->setParameter(sizeof(unsigned int), &this->batchSize);
                }

                // Sets Pattern parameters in Kernel object
                for (auto &paramName : this->paramsOrder) {
                    auto param = this->getParameter(paramName);
                    this->setParameterInKernel<TDriverInstance>(kernel, param);
                }
            }

            template<class TDriverInstance>
            void setDimsParametersInKernel(decltype(TDriverInstance::getKernelType())* kernel, Driver::Dimensions dims) {
                for(int d = 0; d < dims.getCount(); d++) {
                    if (dims.is(d)) {
                        // #ifdef GSPAR_DEBUG
                        //     std::stringstream ss; ss.str("");
                        //     ss << "[" << std::this_thread::get_id() << " GSPar Pattern "<<this<<"] Setting max parameter for dimension " << d << ": " << dims[d].max << " (in kernel " << kernel << ")" << std::endl;
                        //     std::cout << ss.str();
                        //     ss.str("");
                        // #endif
                        kernel->setParameter(sizeof(unsigned long), &(dims[d].max));
                        if (dims[d].min && !this->isBatched()) { // Same check as codeGenerator
                            // TODO Support min in batches
                            // #ifdef GSPAR_DEBUG
                            //     ss << "[" << std::this_thread::get_id() << " GSPar Pattern "<<this<<"] Setting min parameter for dimension " << d << ": " << dims[d].min << " (in kernel " << kernel << ")" << std::endl;
                            //     std::cout << ss.str();
                            //     ss.str("");
                            // #endif
                            kernel->setParameter(sizeof(unsigned long), &(dims[d].min));
                        }
                    }
                }
            }

            template<class TDriverInstance>
            void setParameterInKernel(decltype(TDriverInstance::getKernelType())* kernel, BaseParameter* parameter) {
                if (parameter->direction == Pattern::ParameterDirection::GSPAR_PARAM_NONE) {
                    return; // NONE parameters doesn't go in kernel
                }
                #ifdef GSPAR_DEBUG
                    std::stringstream ss;
                    ss << "[" << std::this_thread::get_id() << " GSPar Pattern "<<this<<"] Setting parameter '" << parameter->name << "' in kernel " << kernel << std::endl;
                    std::cout << ss.str();
                    ss.str("");
                #endif
                if (parameter->paramValueType == Pattern::ParameterValueType::GSPAR_PARAM_POINTER) { // It is a PointerParameter
                    auto paramPointer = static_cast<Pattern::PointerParameter*>(parameter);
                    if (parameter->isBatched()) {
                        auto chunkedMemObj = dynamic_cast<decltype(TDriverInstance::getChunkedMemoryObjectType())*>(paramPointer->getMemoryObject());
                        // We don't need to wait the async copy because they are running in the same execution flow as the kernel itself
                        // if (chunkedMemObj) {
                        //     chunkedMemObj->waitAsync(); // Waits for async copy to finish
                        // }
                        kernel->setParameter(chunkedMemObj); // We can simply set the memory object
                    } else {
                        auto singleMemObj = dynamic_cast<decltype(TDriverInstance::getMemoryObjectType())*>(paramPointer->getMemoryObject());
                        // We don't need to wait the async copy because they are running in the same execution flow as the kernel itself
                        // if (singleMemObj) {
                        //     singleMemObj->waitAsync(); // Waits for async copy to finish
                        // }
                        kernel->setParameter(singleMemObj); // We can simply set the memory object
                    }
                } else if (parameter->paramValueType == Pattern::ParameterValueType::GSPAR_PARAM_VALUE) { // It is a ValueParameter
                    auto paramValue = static_cast<Pattern::ValueParameter*>(parameter);
                    if (parameter->isBatched()) {
                        // Batched ValueParameters are allocated as a single buffer
                        auto singleMemObj = dynamic_cast<decltype(TDriverInstance::getMemoryObjectType())*>(paramValue->getMemoryObject());
                        // We don't need to wait the async copy because they are running in the same execution flow as the kernel itself
                        // if (singleMemObj) {
                        //     singleMemObj->waitAsync(); // Waits for async copy to finish
                        // }
                        kernel->setParameter(singleMemObj); // We can simply set the memory object
                    } else {
                        // We get the pointer directly
                        auto paramValue = static_cast<Pattern::ValueParameter*>(parameter);
                        kernel->setParameter(paramValue->size, paramValue->getPointer());
                    }
                }

            }

            template<class TDriverInstance>
            void run() {
                this->run<TDriverInstance>(Driver::Dimensions(), true);
            }

            template<class TDriverInstance>
            void run(unsigned long dims[3][2]) {
                this->run<TDriverInstance>(Driver::Dimensions(dims), false);
            }

            template<class TDriverInstance>
            void run(unsigned long max[3]) {
                this->run<TDriverInstance>(Driver::Dimensions(max), false);
            }

            template<class TDriverInstance>
            void run(Driver::Dimensions dims) {
                this->run<TDriverInstance>(dims, false);
            }

            // Overridable callbacks
            // TODO these callbacks should have protected visibility
            virtual void callbackBeforeGeneratingKernelSource() { }
            virtual void callbackBeforeAllocatingMemoryOnGpu(Driver::Dimensions dims, Driver::BaseKernelBase *kernel) { }
            virtual void callbackAfterCopyDataFromHostToGpu() { }
            virtual void callbackBeforeRunInGpu() { }
            virtual void callbackAfterRunInGpu() { }
            virtual void callbackAfterCopyDataFromGpuToHost(Driver::Dimensions dims, Driver::BaseKernelBase *kernel) { }
        };
    
    }
}

#endif
