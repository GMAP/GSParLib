
#include <regex>
#include <iostream>
#include <cstring>
#include <vector>
#ifdef GSPAR_DEBUG
#include <sstream>
#endif
#include <string>
#include <typeinfo>

#include "GSPar_CUDA.hpp"

using namespace GSPar::Driver::CUDA;


///// Exception /////

std::string Exception::getErrorString(CUresult code) {
    const char* errName;
    cuGetErrorName(code, &errName);
    const char* errString;
    cuGetErrorString(code, &errString);
    std::string res(errName);
    res.append(": ");
    res.append(errString);
    return res;
}
Exception::Exception(std::string msg, std::string details) : BaseException(msg, details) { }
Exception::Exception(CUresult code, std::string details) : BaseException(code, details) {
    // Can't call this virtual function in the base constructor
    this->msg = this->getErrorString(code);
}
Exception* Exception::checkError(CUresult code, std::string details) {
    return BaseException::checkError<Exception>(code, CUDA_SUCCESS, details);
}
void Exception::throwIfFailed(CUresult code, std::string details) {
    // Exception* ex = Exception::checkError(code, details);
    // if (ex) std::cerr << "Exception: " << ex->what() << " - " << ex->getDetails() << std::endl;
    BaseException::throwIfFailed<Exception>(code, CUDA_SUCCESS, details);
}

std::string CompilationException::getErrorString(nvrtcResult code) {
    const char* errString = nvrtcGetErrorString(code);
    return std::string(errString);
}
CompilationException::CompilationException(std::string msg, std::string details) : BaseException(msg, details) { }
CompilationException::CompilationException(nvrtcResult code, std::string details) : BaseException(code, details) {
    // Can't call this virtual function in the base constructor
    this->msg = this->getErrorString(code);
}
CompilationException* CompilationException::checkError(nvrtcResult code, std::string details) {
    return BaseException::checkError<CompilationException>(code, NVRTC_SUCCESS, details);
}
void CompilationException::throwIfFailed(nvrtcResult code, std::string details) {
    BaseException::throwIfFailed<CompilationException>(code, NVRTC_SUCCESS, details);
}
void CompilationException::throwIfFailed(nvrtcResult code, nvrtcProgram cudaProgram, std::string details) {
    if (code == NVRTC_ERROR_COMPILATION) {
        size_t logSize;
        nvrtcGetProgramLogSize(cudaProgram, &logSize);
        char* log = new char[logSize];
        nvrtcGetProgramLog(cudaProgram, log);
        details += "\n" + std::string(log);
    }
    BaseException::throwIfFailed<CompilationException>(code, NVRTC_SUCCESS, details);
}


///// ExecutionFlow /////

ExecutionFlow::ExecutionFlow() : BaseExecutionFlow() { }
ExecutionFlow::ExecutionFlow(Device* device) : BaseExecutionFlow(device) { }
ExecutionFlow::~ExecutionFlow() {
    // We don't throw exceptions on destructors
    #ifdef GSPAR_DEBUG
        std::stringstream ss; // Using stringstream eases multi-threaded debugging
    #endif
    if (this->flowObject) {
        // In case the device is still doing work in the stream when cuStreamDestroy() is called,
        // the function will return immediately and the resources associated with the stream will
        // be released automatically once the device has completed all work in the stream.
        // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html
        #ifdef GSPAR_DEBUG
            ss << "[GSPar Execution Flow " << this << "] clearing CUstream" << std::endl;
            std::cout << ss.str();
            ss.str("");
        #endif
        Exception* ex = Exception::checkError( cuStreamDestroy(this->flowObject) );
        if (ex != nullptr) {
            std::cerr << "Failed when releasing cuda stream of execution flow: ";
            std::cerr << ex->what() << " - " << ex->getDetails() << std::endl;
            delete ex;
        }

        this->flowObject = NULL;
    }
}
CUstream ExecutionFlow::start() {
    // #ifdef GSPAR_DEBUG
    //     std::stringstream ss; // Using stringstream eases multi-threaded debugging
    //     ss << "[GSPar CUDA "<<this<<"] Starting execution flow " << this << " in device " << this->device << std::endl;
    //     std::cout << ss.str();
    //     ss.str("");
    // #endif

    if (!this->device) {
        // Can't start flow on a NULL device
        throw Exception("A device is required to start an execution flow", defaultExceptionDetails());
    }
    if (!this->flowObject) {
        this->device->getContext(); // There must be a context to create a stream
        CUstream stream;
        throwExceptionIfFailed( cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) );
        this->setBaseFlowObject(stream);
    }
    return this->getBaseFlowObject();
}
void ExecutionFlow::synchronize() {
    throwExceptionIfFailed( cuStreamSynchronize(this->getBaseFlowObject()) );
}
CUstream ExecutionFlow::checkAndStartFlow(Device* device, ExecutionFlow* executionFlow) {
    return BaseExecutionFlow::checkAndStartFlow(device, executionFlow);
}


///// AsyncExecutionSupport /////

AsyncExecutionSupport::AsyncExecutionSupport(CUstream asyncObj) : BaseAsyncExecutionSupport(asyncObj) { }
void AsyncExecutionSupport::waitAsync() {
    if (this->asyncObject) {
        throwExceptionIfFailed( cuStreamSynchronize(this->asyncObject) );
        this->runningAsync = false;
    }
};
// static
void AsyncExecutionSupport::waitAllAsync(std::initializer_list<AsyncExecutionSupport*> asyncs) {
    for (auto async : asyncs) {
        throwExceptionIfFailed( cuStreamSynchronize(async->getBaseAsyncObject()) );
    }
}


///// Instance /////

Instance *Instance::instance = nullptr;

void Instance::loadGpuList() {
    this->init();
    this->clearGpuList();

    unsigned int gpuCount = this->getGpuCount();
    for (unsigned int i = 0; i < gpuCount; ++i) {
        this->devices.push_back(new Device(i));
    }
}

Instance::Instance() : BaseInstance(Runtime::GSPAR_RT_CUDA) { }
Instance::~Instance() {
    Instance::instance = nullptr;
}
Instance* Instance::getInstance() {
    // TODO implement thread-safety
    if (!instance) {
        instance = new Instance();
    }
    return instance;
}

void Instance::init() {
    if (!this->instanceInitiated) {
        throwExceptionIfFailed( cuInit(0) );
        this->instanceInitiated = true;
    }
}

unsigned int Instance::getGpuCount() {
    this->init();
    int gpuCount = 0;
    throwExceptionIfFailed( cuDeviceGetCount(&gpuCount) );
    return gpuCount;
}


///// Device /////

Device::Device() : BaseDevice() { }
Device::Device(int ordinal) {
    this->libDevice = new CUdevice;
    this->deviceId = ordinal;
    throwExceptionIfFailed( cuDeviceGet(this->libDevice, ordinal) );
}
Device::~Device() {
    // We don't throw exceptions on destructors
#ifdef GSPAR_DEBUG
    std::cout << "[GSPar Device " << this << "] Destructing";
#endif
    if (this->defaultExecutionFlow) {
        delete this->defaultExecutionFlow;
        this->defaultExecutionFlow = NULL;
    }

    if (this->libContext && this->libDevice) {
        Exception* ex = Exception::checkError( cuCtxSynchronize() );
        if (ex) {
            std::cerr << "Failed when waiting for context to synchronize on Device's destructor: ";
            std::cerr << ex->what() << " - " << ex->getDetails() << std::endl;
            delete ex;
        }


        ex = Exception::checkError( cuDevicePrimaryCtxRelease(*this->libDevice) );
        if (ex) {
            std::cerr << "Failed when releasing primary device context on Device's destructor: ";
            std::cerr << ex->what() << " - " << ex->getDetails() << std::endl;
            delete ex;
        }
        this->libContext = NULL;
    }
    if (this->libDevice) {
        delete this->libDevice;
        this->libDevice = NULL;
    }
#ifdef GSPAR_DEBUG
    std::cout << "[GSPar Device " << this << "] Destructed successfully";
#endif
}
ExecutionFlow* Device::getDefaultExecutionFlow() {
    std::lock_guard<std::mutex> lock(this->defaultExecutionFlowMutex); // Auto-unlock, RAII
    if (!this->defaultExecutionFlow) {
        this->defaultExecutionFlow = new ExecutionFlow(this);
    }
    return this->defaultExecutionFlow;
    // Auto-unlock of defaultExecutionFlowMutex, RAII
}
CUcontext Device::getContext() {
    if (!this->libContext) {
        std::lock_guard<std::mutex> lock(this->libContextMutex); // Auto-unlock, RAII
        if (!this->libContext) { // Check if someone changed it while we were waiting for the lock
            CUcontext context;
            throwExceptionIfFailed( cuDevicePrimaryCtxRetain(&context, *this->libDevice) );
            this->setContext(context);
        }
        // Auto-unlock of libContextMutex, RAII
    }
    // Sets the context as current for the caller thread
    throwExceptionIfFailed( cuCtxSetCurrent(this->libContext) );
    return this->libContext;
}
CUstream Device::startDefaultExecutionFlow() {
    return this->getDefaultExecutionFlow()->start();
}
unsigned int Device::getDeviceId() {
    this->getContext(); // There must be a context to call almost everything
    return this->deviceId;
}
const std::string Device::getName() {
    this->getContext(); // There must be a context to call almost everything
    unsigned int default_size = 256;
    char* name = new char[default_size];
    throwExceptionIfFailed( cuDeviceGetName(name, default_size, *this->getBaseDeviceObject()) );
    // Try 6 times more
    while (default_size <= 16384 && std::string(name).length() > default_size) {
        default_size *= 2;
        delete name;
        name = new char[default_size];
        throwExceptionIfFailed( cuDeviceGetName(name, default_size, *this->getBaseDeviceObject()) );
    }
    return name;
}
unsigned int Device::getComputeUnitsCount() {
    this->getContext(); // There must be a context to call almost everything
    return this->queryInfoNumeric(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
}
unsigned int Device::getWarpSize() {
    this->getContext(); // There must be a context to call almost everything
    return this->queryInfoNumeric(CU_DEVICE_ATTRIBUTE_WARP_SIZE);
}
unsigned int Device::getMaxThreadsPerBlock() {
    this->getContext(); // There must be a context to call almost everything
    return this->queryInfoNumeric(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
}
unsigned long Device::getGlobalMemorySizeBytes() {
    this->getContext(); // There must be a context to call almost everything
    unsigned long bytes;
    throwExceptionIfFailed( cuDeviceTotalMem(&bytes, *this->getBaseDeviceObject()) );
    return bytes;
}
unsigned long Device::getLocalMemorySizeBytes() {
    this->getContext(); // There must be a context to call almost everything
    return this->queryInfoNumeric(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
}
unsigned long Device::getSharedMemoryPerComputeUnitSizeBytes() {
    this->getContext(); // There must be a context to call almost everything
    return this->queryInfoNumeric(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR);
}
unsigned int Device::getClockRateMHz() {
    this->getContext(); // There must be a context to call almost everything
    return (this->queryInfoNumeric(CU_DEVICE_ATTRIBUTE_CLOCK_RATE) / 1000);
}
bool Device::isIntegratedMainMemory() {
    this->getContext(); // There must be a context to call almost everything
    return this->queryInfoNumeric(CU_DEVICE_ATTRIBUTE_INTEGRATED);
}
MemoryObject* Device::malloc(long size, void* hostPtr, bool readOnly, bool writeOnly) {
    return new MemoryObject(this, size, hostPtr, readOnly, writeOnly);
}
MemoryObject* Device::malloc(long size, const void* hostPtr) {
    return new MemoryObject(this, size, hostPtr);
}
ChunkedMemoryObject* Device::mallocChunked(unsigned int chunks, long chunkSize, void** hostPointers, bool readOnly, bool writeOnly) {
    return new ChunkedMemoryObject(this, chunks, chunkSize, hostPointers, readOnly, writeOnly);
}
ChunkedMemoryObject* Device::mallocChunked(unsigned int chunks, long chunkSize, const void** hostPointers) {
    return new ChunkedMemoryObject(this, chunks, chunkSize, hostPointers);
}
Kernel* Device::prepareKernel(const std::string kernel_source, const std::string kernel_name) {
    this->getContext(); // There must be a context to call almost everything
    return new Kernel(this, kernel_source, kernel_name);
}
std::vector<Kernel*> Device::prepareKernels(const std::string kernelSource, const std::vector<std::string> kernelNames) {
    this->getContext(); // There must be a context to call almost everything
    
    std::string programName = "program_" + kernelNames.front();
    
    auto programAndModule = this->compileCudaProgramAndLoadModule(kernelSource, programName);
    nvrtcProgram cudaProgram = std::get<0>(programAndModule);
    CUmodule cudaModule = std::get<1>(programAndModule);

    std::vector<Kernel*> kernels;
    for (auto name : kernelNames) {
        kernels.push_back(new Kernel(this, cudaProgram, cudaModule, name));
    }
    return kernels;
}
const int Device::queryInfoNumeric(CUdevice_attribute paramName, bool cacheable) {
    // https://www.quora.com/Is-it-thread-safe-to-write-to-distinct-keys-different-key-for-each-thread-in-a-std-map-in-C-for-keys-that-have-existing-entries-in-the-map/answer/John-R-Grout
    if (cacheable) { // Check if the attribute is cached
        std::lock_guard<std::mutex> lock(this->attributeCacheMutex); // Auto-unlock, RAII
        auto it = this->attributeCache.find(paramName);
        if (it != this->attributeCache.end()) {
            return it->second;
        }
        // Auto-unlock of attributeCacheMutex, RAII
    }

    int pi;
    throwExceptionIfFailed( cuDeviceGetAttribute(&pi, paramName, *this->getBaseDeviceObject()) );
    if (cacheable) { // Stores the attribute in cache
        std::lock_guard<std::mutex> lock(this->attributeCacheMutex); // Auto-unlock, RAII
        this->attributeCache[paramName] = pi;
        // Auto-unlock of attributeCacheMutex, RAII
    }
    return pi;
}
std::tuple<nvrtcProgram, CUmodule> Device::compileCudaProgramAndLoadModule(std::string source, const std::string programName) {
#ifdef GSPAR_DEBUG
    std::stringstream ss; // Using stringstream eases multi-threaded debugging
    ss << "[GSPar Device " << this << "] Kernel received to compile: [" << programName << "] = \n" << source << std::endl;
    std::cout << ss.str();
    ss.str("");
#endif

    // --------------------------------------------------------------------
    // gets the compute capability
    // --------------------------------------------------------------------
    int computeCapabilityMajor = this->queryInfoNumeric(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    int computeCapabilityMinor = this->queryInfoNumeric(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    std::string computeCapabilityArg = "--gpu-architecture=compute_" + std::to_string(computeCapabilityMajor) + std::to_string(computeCapabilityMinor);

    // --------------------------------------------------------------------
    // Appending additional routines to the kernel source
    // --------------------------------------------------------------------
    std::string completeKernelSource = "";
    if (computeCapabilityMajor < 6) {
        // atomicAdd() for double-precision floating-point numbers is not available by
        // default on devices with compute capability lower than 6.0
        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicadd
        completeKernelSource.append(KernelGenerator::ATOMIC_ADD_POLYFILL);
    }
    completeKernelSource.append(Instance::getInstance()->getKernelGenerator()->generateStdFunctions());
    completeKernelSource.append(Instance::getInstance()->getKernelGenerator()->replaceMacroKeywords(source));

#ifdef GSPAR_DEBUG
    ss << "[GSPar Device " << this << "] Complete kernel for compilation: [" << programName << "] = \n" << completeKernelSource << std::endl;
    std::cout << ss.str();
    ss.str("");
#endif

    nvrtcProgram cudaProgram;
    CUmodule cudaModule;

    throwCompilationExceptionIfFailed( nvrtcCreateProgram(&cudaProgram, completeKernelSource.c_str(), programName.c_str(), 0, NULL, NULL), cudaProgram );

    // https://docs.nvidia.com/cuda/nvrtc/index.html
    int numOptions = 7;
    const char *compilationOptions[numOptions];
    compilationOptions[0] = "--device-as-default-execution-space";
    compilationOptions[1] = computeCapabilityArg.c_str();
    std::string gsparMacroKernel = "--define-macro=GSPAR_DEVICE_KERNEL=" + KernelGenerator::KERNEL_PREFIX;
    compilationOptions[2] = gsparMacroKernel.c_str();
    std::string gsparMacroGlobalMemory = "--define-macro=GSPAR_DEVICE_GLOBAL_MEMORY=" + KernelGenerator::GLOBAL_MEMORY_PREFIX;
    compilationOptions[3] = gsparMacroGlobalMemory.c_str();
    std::string gsparMacroSharedMemory = "--define-macro=GSPAR_DEVICE_SHARED_MEMORY=" + KernelGenerator::SHARED_MEMORY_PREFIX;
    compilationOptions[4] = gsparMacroSharedMemory.c_str();
    std::string gsparMacroConstant = "--define-macro=GSPAR_DEVICE_CONSTANT=" + KernelGenerator::CONSTANT_PREFIX;
    compilationOptions[5] = gsparMacroConstant.c_str();
    std::string gsparMacroDevFunction = "--define-macro=GSPAR_DEVICE_FUNCTION=" + KernelGenerator::DEVICE_FUNCTION_PREFIX;
    compilationOptions[6] = gsparMacroDevFunction.c_str();

#ifdef GSPAR_DEBUG
    ss << "[GSPar Device " << this << "] Compiling kernel with " << numOptions << " options: ";
    for (int iDebug = 0; iDebug < numOptions; iDebug++) {
        ss << compilationOptions[iDebug] << " ";
    }
    ss << std::endl;
    std::cout << ss.str();
    ss.str("");
#endif

    throwCompilationExceptionIfFailed( nvrtcCompileProgram(cudaProgram, numOptions, compilationOptions), cudaProgram );

    size_t ptxSize;
    throwCompilationExceptionIfFailed( nvrtcGetPTXSize(cudaProgram, &ptxSize), cudaProgram );
    char* ptxSource = new char[ptxSize];
    throwCompilationExceptionIfFailed( nvrtcGetPTX(cudaProgram, ptxSource), cudaProgram );

    unsigned int error_buffer_size = 1024;
    std::vector<CUjit_option> options;
    std::vector<void*> values;
    char* error_log = new char[error_buffer_size];
    //Pointer to a buffer in which to print any log messages that reflect errors
    options.push_back(CU_JIT_ERROR_LOG_BUFFER);
    values.push_back(error_log);
    //Log buffer size in bytes. Log messages will be capped at this size (including null terminator)
    options.push_back(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES);
    // Casting through uintptr_t avoids compiler warning [https://stackoverflow.com/a/30106751/3136474]
    values.push_back((void*)(uintptr_t)error_buffer_size); // https://developer.nvidia.com/nvidia_bug/2917596
    //Determines the target based on the current attached context (default)
    options.push_back(CU_JIT_TARGET_FROM_CUCONTEXT);
    values.push_back(0); //No option value required for CU_JIT_TARGET_FROM_CUCONTEXT

    Exception::throwIfFailed( cuModuleLoadDataEx(&cudaModule, ptxSource, options.size(), options.data(), values.data()), error_log);
    
    return std::make_tuple(cudaProgram, cudaModule);
}


///// Kernel /////

void Kernel::loadCudaFunction(const std::string kernelName) {
    throwExceptionIfFailed( cuModuleGetFunction(&this->cudaFunction, this->cudaModule, kernelName.c_str()) );
}

Kernel::Kernel() : BaseKernel() { }
Kernel::Kernel(Device* device, const std::string kernelSource, const std::string kernelName) : BaseKernel(device, kernelSource, kernelName) {
    std::string programName = "program_" + kernelName;

    auto programAndModule = this->device->compileCudaProgramAndLoadModule(kernelSource, programName);
    this->cudaProgram = std::get<0>(programAndModule);
    this->cudaModule = std::get<1>(programAndModule);

    this->isPrecompiled = false; //Kernel owns cudaProgram

    this->loadCudaFunction(kernelName);

    #ifdef GSPAR_DEBUG
        std::stringstream ss; // Using stringstream eases multi-threaded debugging
        ss.str("");
        // See Kernel::getNumBlocksAndThreadsFor for explanation on this code.
        int deviceRegsPerBlock = this->device->queryInfoNumeric(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK);
        int funcNumRegs = this->queryInfoNumeric(CU_FUNC_ATTRIBUTE_NUM_REGS);
        funcNumRegs *= 1.15; // +15% of margin
        ss << "[GSPar Kernel " << this << "] " << this->kernelName << " Device Num regs is " << deviceRegsPerBlock << ", Func Num regs is " << funcNumRegs << "." << std::endl;
        std::cout << ss.str();
        ss.str("");
    #endif  
}
Kernel::Kernel(Device* device, nvrtcProgram cudaProgram, CUmodule cudaModule, const std::string kernelName) : BaseKernel(device) {
    this->cudaProgram = cudaProgram;
    this->isPrecompiled = true; //Kernel shares cudaProgram

    this->cudaModule = cudaModule;

    this->loadCudaFunction(kernelName);
}
Kernel::~Kernel() {
#ifdef GSPAR_DEBUG
    std::stringstream ss; // Using stringstream eases multi-threaded debugging
    ss << "[GSPar Kernel " << this << "] Destructing..." << std::endl;
    std::cout << ss.str();
    ss.str("");
#endif
    if (this->isRunningAsync()) {
        this->waitAsync();
    }
    if (!this->isPrecompiled && this->cudaProgram) {
        nvrtcDestroyProgram(&this->cudaProgram); // We don't throw exceptions on destructors
    }
}
void Kernel::cloneInto(BaseKernelBase* baseOther) {
    BaseKernel::cloneInto(baseOther);
    Kernel* other = static_cast<Kernel*>(baseOther);
    other->cudaProgram = this->cudaProgram;
    other->cudaModule = this->cudaModule;
    other->cudaFunction = this->cudaFunction;
    other->kernelParams = this->kernelParams;
    // TODO Who will destroy the NVRTC program?
    this->isPrecompiled = true; // Now the program is shared
    other->isPrecompiled = true;
    other->attributeCache = this->attributeCache;
}
int Kernel::setParameter(MemoryObject* memoryObject) {
    CUdeviceptr* cudaObject = memoryObject->getBaseMemoryObject();
    this->kernelParams.push_back(cudaObject);
    return ++this->parameterCount;
}
int Kernel::setParameter(ChunkedMemoryObject* chunkedMemoryObject) {
    CUdeviceptr* cudaObject = chunkedMemoryObject->getBaseMemoryObject();
    this->kernelParams.push_back(cudaObject);
    return ++this->parameterCount;
}
int Kernel::setParameter(size_t parm_size, void* parm) {
    void *parmPtr = parm;
    if (parm_size <= sizeof(unsigned long long)) { // We copy single values
        // Should we copy all parameters?
        parmPtr = new unsigned char[parm_size];
        memcpy(parmPtr, parm, parm_size);
    }
    this->kernelParams.push_back(parmPtr);
    return ++this->parameterCount;
}
int Kernel::setParameter(size_t parm_size, const void* parm) {
    // cuLaunchKernel expects a void**, so we can't work with const
    // Another nice trick to cast to void*: https://migocpp.wordpress.com/2018/04/16/cuda-runtime-templates/
    return this->setParameter(parm_size, const_cast<void*>(parm));
}
void Kernel::clearParameters() {
    BaseKernel::clearParameters();
    this->kernelParams.clear();
}
GSPar::Driver::Dimensions Kernel::getNumBlocksAndThreadsFor(Dimensions dims) {
    #ifdef GSPAR_DEBUG
        std::stringstream ss; // Using stringstream eases multi-threaded debugging
        ss.str("");
    #endif

    unsigned int deviceMaxThreadsPerBlock = this->device->getMaxThreadsPerBlock();

    // #ifdef GSPAR_DEBUG
    //     ss << "[GSPar Kernel " << this << "] Max threads per block in device " << this->device << ": " << deviceMaxThreadsPerBlock << std::endl;
    //     std::cout << ss.str();
    //     ss.str("");
    // #endif

    // Check if the function uses too much registers
    int deviceRegsPerBlock = this->device->queryInfoNumeric(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK);
    int funcNumRegs = this->queryInfoNumeric(CU_FUNC_ATTRIBUTE_NUM_REGS);
    // In practice, we've seen CUDA exploding with CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: too many resources requested for launch
    // when we use the exact number of threads that can be used according to the number of registers reported by CU_FUNC_ATTRIBUTE_NUM_REGS.
    // The raytracer test is an example of such issue.
    // So, we increase this number a little bit to have some margin.
    funcNumRegs *= 1.15; // +15% of margin

    unsigned int regsMaxThreadsPerBlock = (double)deviceRegsPerBlock/funcNumRegs; // Max threads per block according to the register usage

    // Actual max threads per block according to device capability and function register usage
    unsigned int actualMaxThreadsPerBlock = deviceMaxThreadsPerBlock;
    if (regsMaxThreadsPerBlock < deviceMaxThreadsPerBlock) {
        actualMaxThreadsPerBlock = regsMaxThreadsPerBlock;
    }

    #ifdef GSPAR_DEBUG
        ss << "[GSPar Kernel " << this << "] " << this->kernelName << " Device Num regs is " << deviceRegsPerBlock << ", Func Num regs is " << funcNumRegs << ", so max threads per block is " << regsMaxThreadsPerBlock;
        ss << ". Max threads per block of device is " << deviceMaxThreadsPerBlock << ", but actual max threads is " << actualMaxThreadsPerBlock << std::endl;
        std::cout << ss.str();
        ss.str("");
    #endif

    size_t maxThreadsDimension[SUPPORTED_DIMS] = {
        (size_t)this->device->queryInfoNumeric(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X),
        (size_t)this->device->queryInfoNumeric(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y),
        (size_t)this->device->queryInfoNumeric(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z),
    };

    // #ifdef GSPAR_DEBUG
    //     ss << "[GSPar Kernel " << this << "] Max threads per dimension is " << maxThreadsDimension[0] << " x " << maxThreadsDimension[1] << " x " << maxThreadsDimension[2] << std::endl;
    //     std::cout << ss.str();
    //     ss.str("");
    // #endif

    return this->getNumBlocksAndThreads(dims, actualMaxThreadsPerBlock, maxThreadsDimension);
}
void Kernel::runAsync(Dimensions dims, ExecutionFlow* executionFlow) {

    #ifdef GSPAR_DEBUG
        std::stringstream ss; // Using stringstream eases multi-threaded debugging
        ss << "[GSPar Kernel " << this << "] Running kernel async with " << this->kernelParams.size() << " parameters for " << dims.toString() << " in flow " << executionFlow << std::endl;
        std::cout << ss.str();
        ss.str("");
    #endif

    CUstream cudaStream = ExecutionFlow::checkAndStartFlow(this->device, executionFlow);

    if (!dims.x) {
        throw Exception("The first dimension is required to run a kernel");
    }

    // #ifdef GSPAR_DEBUG
    //     ss << "[GSPar Kernel " << this << "] Checking max threads per block in device " << this->device << std::endl;
    //     std::cout << ss.str();
    //     ss.str("");
    // #endif

    Dimensions blocksAndThreads = this->getNumBlocksAndThreadsFor(dims);

    unsigned int numBlocks[SUPPORTED_DIMS] = {
        (unsigned int)blocksAndThreads.x.min,
        (unsigned int)blocksAndThreads.y.min,
        (unsigned int)blocksAndThreads.z.min
    };
    unsigned int numThreads[SUPPORTED_DIMS] = {
        (unsigned int)blocksAndThreads.x.max,
        (unsigned int)blocksAndThreads.y.max,
        (unsigned int)blocksAndThreads.z.max
    };

    #ifdef GSPAR_DEBUG
        ss << "[GSPar Kernel " << this << "] Starting kernel with " << this->kernelParams.size() << " parameters" << std::endl;
        ss << "[GSPar Kernel " << this << "] Shall start " << dims.toString() << " threads: ";
        ss << "starting (" << numThreads[0] << "," << numThreads[1] << "," << numThreads[2] << ") threads ";
        ss << "in (" << numBlocks[0] << "," << numBlocks[1] << "," << numBlocks[2] << ") blocks ";
        ss << "using " << this->sharedMemoryBytes << " bytes of shared memory in execution flow " << executionFlow << " (CUstream " << cudaStream << ")" << std::endl;
        std::cout << ss.str();
        ss.str("");
    #endif

    throwExceptionIfFailed( cuLaunchKernel(this->cudaFunction,
        numBlocks[0], numBlocks[1], numBlocks[2], // 3D blocks
        numThreads[0], numThreads[1], numThreads[2], // 3D threads
        this->sharedMemoryBytes, cudaStream, this->kernelParams.data(), NULL) );

    // #ifdef GSPAR_DEBUG
    //     ss << "[GSPar Kernel " << this << "] Started kernel execution in execution flow " << executionFlow << " (CUstream " << cudaStream << ")" << std::endl;
    //     std::cout << ss.str();
    //     ss.str("");
    // #endif

    this->setBaseAsyncObject(cudaStream);

    this->runningAsync = true;
}
const int Kernel::queryInfoNumeric(CUfunction_attribute paramName, bool cacheable) {
    if (cacheable) { // Check if the attribute is cached
        // We don't use locks here because the Kernel object is not intended to be shared among threads
        auto it = this->attributeCache.find(paramName);
        if (it != this->attributeCache.end()) {
            return it->second;
        }
    }

    int pi;
    throwExceptionIfFailed( cuFuncGetAttribute(&pi, paramName, this->cudaFunction) );
    if (cacheable) { // Stores the attribute in cache
        this->attributeCache[paramName] = pi;
    }
    return pi;
}



///// MemoryObject /////
void MemoryObject::allocDeviceMemory() {
    this->device->getContext(); // There must be a context to call cuMemAlloc

    this->devicePtr = new CUdeviceptr; // It is initialized as NULL, we have to allocate space for it
    throwExceptionIfFailed( cuMemAlloc(this->devicePtr, size) );
}

MemoryObject::MemoryObject(Device* device, size_t size, void* hostPtr, bool readOnly, bool writeOnly) : BaseMemoryObject(device, size, hostPtr, readOnly, writeOnly) {
    this->allocDeviceMemory();
}
MemoryObject::MemoryObject(Device* device, size_t size, const void* hostPtr) : BaseMemoryObject(device, size, hostPtr) {
    this->allocDeviceMemory();
}
MemoryObject::~MemoryObject() {
    if (this->devicePtr) {
        cuMemFree(*(this->devicePtr)); // We don't throw exceptions on destructors
        this->devicePtr = NULL;
    }
    if (this->isPinnedHostMemory()) {
        cuMemHostUnregister(this->hostPtr); // We don't throw exceptions on destructors
    }
}
void MemoryObject::pinHostMemory() {
    if (!this->isPinnedHostMemory()) { // TODO implement thread-safety
        CUresult result = cuMemHostRegister(this->hostPtr, this->size, 0);
        if (result != CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED) {
            throwExceptionIfFailed(result);
        }
    }
    BaseMemoryObject::pinHostMemory();
}

void MemoryObject::copyIn() {
    throwExceptionIfFailed( cuMemcpyHtoD(*(this->devicePtr), this->hostPtr, this->size) );
}
void MemoryObject::copyOut() {
    throwExceptionIfFailed( cuMemcpyDtoH(this->hostPtr, *(this->devicePtr), this->size) );
}
void MemoryObject::copyInAsync(ExecutionFlow* executionFlow) {
    CUstream cudaStream = ExecutionFlow::checkAndStartFlow(this->device, executionFlow);
    throwExceptionIfFailed( cuMemcpyHtoDAsync(*(this->devicePtr), this->hostPtr, this->size, cudaStream) );
    this->setBaseAsyncObject(cudaStream);
}
void MemoryObject::copyOutAsync(ExecutionFlow* executionFlow) {
    CUstream cudaStream = ExecutionFlow::checkAndStartFlow(this->device, executionFlow);
    throwExceptionIfFailed( cuMemcpyDtoHAsync(this->hostPtr, *(this->devicePtr), this->size, cudaStream) );
    this->setBaseAsyncObject(cudaStream);
}



///// ChunkedMemoryObject /////

void ChunkedMemoryObject::allocDeviceMemory() {
    this->device->getContext(); // There must be a context to call cuMemAlloc

    this->devicePtr = new CUdeviceptr; // It is initialized as NULL, we have to allocate space for it
    throwExceptionIfFailed( cuMemAlloc(this->devicePtr, this->getChunkSize() * this->chunks) ); // We allocate space for all the chunks
}

ChunkedMemoryObject::ChunkedMemoryObject(Device* device, unsigned int chunks, size_t chunkSize, void** hostPointers, bool readOnly, bool writeOnly) :
        BaseChunkedMemoryObject(device, chunks, chunkSize, hostPointers, readOnly, writeOnly) {
    this->allocDeviceMemory();
}
ChunkedMemoryObject::ChunkedMemoryObject(Device* device, unsigned int chunks, size_t chunkSize, const void** hostPointers) :
        BaseChunkedMemoryObject(device, chunks, chunkSize, hostPointers) {
    this->allocDeviceMemory();
}
ChunkedMemoryObject::~ChunkedMemoryObject() { }
void ChunkedMemoryObject::pinHostMemory() {
    // TODO implement pinned memory in chunked memory objects
    // We need to keep this empty method here while it is not implemented so the parent method does not get called
}
void ChunkedMemoryObject::copyIn() {
    for (unsigned int chunk = 0; chunk < this->chunks; chunk++) {
        this->copyIn(chunk);
    }
}
void ChunkedMemoryObject::copyOut() {
    for (unsigned int chunk = 0; chunk < this->chunks; chunk++) {
        this->copyOut(chunk);
    }
}
void ChunkedMemoryObject::copyInAsync(ExecutionFlow* executionFlow) {
    CUstream cudaStream = ExecutionFlow::checkAndStartFlow(this->device, executionFlow);
    for (unsigned int chunk = 0; chunk < this->chunks; chunk++) {
        // We don't call copyInAsync(chunk) to avoid calling checkAndStartFlow for each chunk
        throwExceptionIfFailed( cuMemcpyHtoDAsync((CUdeviceptr)((unsigned char*)(*this->devicePtr)+(chunk*this->getChunkSize())), this->hostPointers[chunk], this->getChunkSize(), cudaStream) );
    }
    this->setBaseAsyncObject(cudaStream);
}
void ChunkedMemoryObject::copyOutAsync(ExecutionFlow* executionFlow) {
    CUstream cudaStream = ExecutionFlow::checkAndStartFlow(this->device, executionFlow);
    for (unsigned int chunk = 0; chunk < this->chunks; chunk++) {
        // We don't call copyOutAsync(chunk) to avoid calling checkAndStartFlow for each chunk
        throwExceptionIfFailed( cuMemcpyDtoHAsync(this->hostPointers[chunk], (CUdeviceptr)((unsigned char*)(*this->devicePtr)+(chunk*this->getChunkSize())), this->getChunkSize(), cudaStream) );
    }
    this->setBaseAsyncObject(cudaStream);
}
void ChunkedMemoryObject::copyIn(unsigned int chunk) {
    throwExceptionIfFailed( cuMemcpyHtoD((CUdeviceptr)((unsigned char*)(*this->devicePtr)+(chunk*this->getChunkSize())), this->hostPointers[chunk], this->getChunkSize()) );
}
void ChunkedMemoryObject::copyOut(unsigned int chunk) {
    throwExceptionIfFailed( cuMemcpyDtoH(this->hostPointers[chunk], (CUdeviceptr)((unsigned char*)(*this->devicePtr)+(chunk*this->getChunkSize())), this->getChunkSize()) );
}
void ChunkedMemoryObject::copyInAsync(unsigned int chunk, ExecutionFlow* executionFlow) {
    CUstream cudaStream = ExecutionFlow::checkAndStartFlow(this->device, executionFlow);
    throwExceptionIfFailed( cuMemcpyHtoDAsync((CUdeviceptr)((unsigned char*)(*this->devicePtr)+(chunk*this->getChunkSize())), this->hostPointers[chunk], this->getChunkSize(), cudaStream) );
    this->setBaseAsyncObject(cudaStream);
}
void ChunkedMemoryObject::copyOutAsync(unsigned int chunk, ExecutionFlow* executionFlow) {
    CUstream cudaStream = ExecutionFlow::checkAndStartFlow(this->device, executionFlow);
    throwExceptionIfFailed( cuMemcpyDtoHAsync(this->hostPointers[chunk], (CUdeviceptr)((unsigned char*)(*this->devicePtr)+(chunk*this->getChunkSize())), this->getChunkSize(), cudaStream) );
    this->setBaseAsyncObject(cudaStream);
}


///// StreamElement /////

StreamElement::StreamElement(Device* device) : BaseStreamElement(device) {
    // Can't call this virtual function in the base constructor
    this->start();
}

StreamElement::~StreamElement() { }


///// KernelGenerator /////

const std::string KernelGenerator::KERNEL_PREFIX = "extern \"C\" __global__";
const std::string KernelGenerator::GLOBAL_MEMORY_PREFIX = "";
const std::string KernelGenerator::SHARED_MEMORY_PREFIX = "extern __shared__";
const std::string KernelGenerator::CONSTANT_PREFIX = "const";
const std::string KernelGenerator::DEVICE_FUNCTION_PREFIX = "__device__";
const std::string KernelGenerator::ATOMIC_ADD_POLYFILL = ""
    "__device__ double atomicAdd(double* address, double val){ \n"
    "    unsigned long long int* address_as_ull = (unsigned long long int*)address; \n"
    "    unsigned long long int old = *address_as_ull, assumed; \n"
    "    do { \n"
    "        assumed = old; \n"
    "        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed))); \n"
    "    } while (assumed != old); \n"
    "    return __longlong_as_double(old); \n"
    "} \n";

const std::string KernelGenerator::getKernelPrefix() {
    return KernelGenerator::KERNEL_PREFIX + " void";
}
std::string KernelGenerator::generateStdFunctions() {
    Dimensions dims({1, 1, 1});
    std::string gspar_get_gid = "__device__ size_t gspar_get_global_id(unsigned int dimension) { \n";
    std::string gspar_get_tid = "__device__ size_t gspar_get_thread_id(unsigned int dimension) { \n";
    std::string gspar_get_bid = "__device__ size_t gspar_get_block_id(unsigned int dimension) { \n";
    std::string gspar_get_bsize = "__device__ size_t gspar_get_block_size(unsigned int dimension) { \n";
    std::string gspar_get_gridsize = "__device__ size_t gspar_get_grid_size(unsigned int dimension) { \n";
    for (int d = 0; d < dims.getCount(); d++) {
        std::string dimName = dims.getName(d);
        gspar_get_gid += "   if (dimension == " + std::to_string(d) + ") return blockIdx." + dimName + " * blockDim."+dimName+" + threadIdx." + dimName + "; \n";
        gspar_get_tid += "   if (dimension == " + std::to_string(d) + ") return threadIdx." + dimName + "; \n";
        gspar_get_bid += "   if (dimension == " + std::to_string(d) + ") return blockIdx." + dimName + "; \n";
        gspar_get_bsize += "   if (dimension == " + std::to_string(d) + ") return blockDim." + dimName + "; \n";
        gspar_get_gridsize += "   if (dimension == " + std::to_string(d) + ") return gridDim." + dimName + "; \n";
    }
    gspar_get_gid += "   return 0; } \n";
    gspar_get_tid += "   return 0; } \n";
    gspar_get_bid += "   return 0; } \n";
    gspar_get_bsize += "   return 0; } \n";
    gspar_get_gridsize += "   return 0; } \n";

    return gspar_get_gid + gspar_get_tid + gspar_get_bid + gspar_get_bsize + gspar_get_gridsize +
    "extern \"C\" __device__ void gspar_synchronize_local_threads() { __syncthreads(); } \n"
    // Atomic functions
    "__device__ int gspar_atomic_add_int(int* valq, int delta) { return atomicAdd(valq, delta); } \n"
    "__device__ double gspar_atomic_add_double(double* valq, double delta) { return atomicAdd(valq, delta); } \n"
    ;
}
std::string KernelGenerator::replaceMacroKeywords(std::string kernelSource) {
    kernelSource = std::regex_replace(kernelSource, std::regex("GSPAR_DEVICE_MACRO_BEGIN"), "#define");
    kernelSource = std::regex_replace(kernelSource, std::regex("GSPAR_DEVICE_MACRO_END"), "\n");
    return kernelSource;
}
std::string KernelGenerator::generateInitKernel(Pattern::BaseParallelPattern* pattern, Dimensions dims) {
    std::string r = "";
    if (pattern->isUsingSharedMemory()) {
        auto shmem = pattern->getSharedMemoryParameter();
        r += KernelGenerator::SHARED_MEMORY_PREFIX + " " + shmem->getNonPointerTypeName() + " " + shmem->name + "[];";
    }
    return r;
}
std::string KernelGenerator::generateParams(Pattern::BaseParallelPattern* pattern, Dimensions dims) {
    std::string r = "";
    for(int d = 0; d < dims.getCount(); d++) {
        if (dims.is(d)) {
            std::string varName = this->getStdVarNameForDimension(pattern->getStdVarNames(), d);
            r += "const unsigned long gspar_max_" + varName + ",";
            if (dims[d].min && !pattern->isBatched()) { // Same check as generateStdVariables
                // TODO Support min in batches
                r += "const unsigned long gspar_min_" + varName + ",";
            }
        }
    }
    if (pattern->isBatched()) {
        // This names are used in other methods
        r += "unsigned int gspar_batch_size,";
    }
    for(auto &param : pattern->getParameterList()) {
        if (param->direction != Pattern::ParameterDirection::GSPAR_PARAM_NONE) {
            if (param->direction == Pattern::ParameterDirection::GSPAR_PARAM_IN && param->isConstant()) {
                r += "const ";
            }
            r += param->toKernelParameter() + ",";
        }
    }
    if (!r.empty()) r.pop_back(); // removes last comma
    return r;
}
std::string KernelGenerator::generateStdVariables(Pattern::BaseParallelPattern* pattern, Dimensions dims) {
    std::array<std::string, 3> patternNames = pattern->getStdVarNames();

    std::string r;
    for(int d = 0; d < dims.getCount(); d++) {
        if (dims[d]) {
            std::string varName = this->getStdVarNameForDimension(patternNames, d);
            // Standard variables are uint3 according do CUDA specification
            // By using size_t we can keep the same type of OpenCL driver
            if (pattern->isBatched()) {
                r += "size_t gspar_global_" + varName;
            } else {
                r += "size_t " + varName;
            }
            r += " = gspar_get_global_id(" + std::to_string(d) + ")";
            if (dims[d].min && !pattern->isBatched()) { // Same check as generateParams
                // TODO Support min in batches
                r += " + gspar_min_" + varName;
            }
            r += "; \n";
            // TODO Support multi-dimensional batches
            if (pattern->isBatched()) {
                // Intended implicit floor(gspar_global/dims)
                r += "size_t gspar_batch_" + varName + " = ((size_t)(gspar_global_" + varName + " / gspar_max_" + varName + ")); \n";
                r += "size_t gspar_offset_" + varName + " = gspar_batch_" + varName + " * gspar_max_" + varName + "; \n";
                // This variable names are used in other methods, keep track
                r += "size_t " + varName + " = gspar_global_" + varName + " - gspar_offset_" + varName + "; \n";
            }
        }
    }
    return r;
}
std::string KernelGenerator::generateBatchedParametersInitialization(Pattern::BaseParallelPattern* pattern, Dimensions max) {
    std::array<std::string, 3> patternNames = pattern->getStdVarNames();
    // TODO Support multi-dimensional batches
    std::string stdVarFirstDimension = this->getStdVarNameForDimension(patternNames, 0);

    std::string r = "";
    for(auto &param : pattern->getParameterList()) {
        if (param->isBatched()) {
            if (param->direction == Pattern::ParameterDirection::GSPAR_PARAM_IN && param->isConstant()) {
                r += "const ";
            }
            r += param->type.getFullName() + " " + param->name + " = ";
            if (param->paramValueType == Pattern::ParameterValueType::GSPAR_PARAM_POINTER) {
                r += "&" + param->getKernelParameterName() + "[gspar_offset_" + stdVarFirstDimension + "]";
            } else if (param->paramValueType == Pattern::ParameterValueType::GSPAR_PARAM_VALUE) {
                r += param->getKernelParameterName() + "[gspar_batch_" + stdVarFirstDimension + "]";
            }
            r += ";\n";
        }
    }
    return r;
}
