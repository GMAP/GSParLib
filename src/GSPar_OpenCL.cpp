
#include <regex>
#include <iostream>
#include <cstring>
#include <vector>
#ifdef GSPAR_DEBUG
#include <sstream>
#include <thread>
#endif

#include "GSPar_OpenCL.hpp"

using namespace GSPar::Driver::OpenCL;

// extern "C" void CL_CALLBACK ocl_pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data) {
//     std::cerr << "OpenCL notified an error: " << errinfo << std::endl;
// }

///// Exception /////

std::string Exception::getErrorString(cl_int code) {
    switch(code) {
        // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
}

Exception::Exception(std::string msg, std::string details) : BaseException(msg, details) { }
Exception::Exception(cl_int code, std::string details) : BaseException(code, details) {
    // Can't call this virtual function in the base constructor
    this->msg = this->getErrorString(code);
}
// static
Exception* Exception::checkError(cl_int code, std::string details) {
    return BaseException::checkError<Exception>(code, CL_SUCCESS, details);
}
// static
void Exception::throwIfFailed(cl_int code, std::string details) {
    BaseException::throwIfFailed<Exception>(code, CL_SUCCESS, details);
}

Exception::Exception(cl_int code, cl_program program, cl_device_id device) : Exception(code) {
    if (code == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = new char[log_size];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        this->msg += std::string(" - ") + std::string(log);
    }
}
// static
Exception* Exception::checkError(cl_int code, cl_program program, cl_device_id device) {
    if (code != CL_SUCCESS) {
        return new Exception(code, program, device);
    }
    return NULL;
}
// static
void Exception::throwIfFailed(cl_int code, cl_program program, cl_device_id device) {
    Exception* ex = Exception::checkError(code, program, device);
    if (ex != NULL) {
        throw *ex;
    }
}


///// ExecutionFlow /////

ExecutionFlow::ExecutionFlow() : BaseExecutionFlow() { }
ExecutionFlow::ExecutionFlow(Device* device) : BaseExecutionFlow(device) { }
ExecutionFlow::~ExecutionFlow() {
    // We don't throw exceptions on destructors
    if (this->flowObject) {
        #ifdef GSPAR_DEBUG
            std::stringstream ss; // Using stringstream eases multi-threaded debugging
            ss << "[" << std::this_thread::get_id() << " GSPar ExFlow] Releasing command queue " << this << std::endl;
            std::cout << ss.str();
            ss.str("");
        #endif
        Exception* ex = Exception::checkError( clReleaseCommandQueue(this->flowObject) );
        if (ex != nullptr) {
            std::cerr << "Failed when releasing OpenCL command queue of execution flow: ";
            std::cerr << ex->what() << " - " << ex->getDetails() << std::endl;
            delete ex;
        }
    }
}
cl_command_queue ExecutionFlow::start() {
    if (!this->device) {
        // Can't start flow on a NULL device
        throw Exception("A device is required to start an execution flow", defaultExceptionDetails());
    }
    if (!this->flowObject) {
        this->device = device;
        cl_int status;
        this->flowObject = clCreateCommandQueue(device->getContext(), device->getBaseDeviceObject(), 0, &status);
        throwExceptionIfFailed(status);
    }
    return this->getBaseFlowObject();
}
void ExecutionFlow::synchronize() {
    #ifdef GSPAR_DEBUG
        std::stringstream ss; // Using stringstream eases multi-threaded debugging
        ss << "[" << std::this_thread::get_id() << " GSPar ExFlow " << this << "] Synchronizing" << std::endl;
        std::cout << ss.str();
        ss.str("");
    #endif
    // clEnqueueMarker(cl_command_queue, cl_event) is deprecated in OpenCL 1.2

    // cl_event evt;
    // throwExceptionIfFailed( clEnqueueMarkerWithWaitList(this->getBaseFlowObject(), 0, NULL, &evt) );
    // throwExceptionIfFailed( clWaitForEvents(1, &evt) );
    // throwExceptionIfFailed( clReleaseEvent(evt) );
    throwExceptionIfFailed( clFinish(this->flowObject) );
}
cl_command_queue ExecutionFlow::checkAndStartFlow(Device* device, ExecutionFlow* executionFlow) {
    return BaseExecutionFlow::checkAndStartFlow(device, executionFlow);
}



///// AsyncExecutionSupport /////

AsyncExecutionSupport::AsyncExecutionSupport(cl_event *asyncObjs, unsigned int numAsyncEvents) :
        BaseAsyncExecutionSupport(asyncObjs), numAsyncEvents(numAsyncEvents) { }
AsyncExecutionSupport::~AsyncExecutionSupport() {
    try {
        this->releaseBaseAsyncObject();
    } catch (GSPar::GSParException &ex) { // We don't throw exceptions on destructors
        std::cerr << "Failed when releasing OpenCL event on AsyncExecutionSupport destructor: ";
        std::cerr << ex.what() << " - " << ex.getDetails() << std::endl;
        this->asyncObject = NULL;
    }
}
void AsyncExecutionSupport::setBaseAsyncObject(cl_event *asyncObject) {
    this->setBaseAsyncObject(asyncObject, 1);
}
void AsyncExecutionSupport::setBaseAsyncObject(cl_event *asyncObject, unsigned int numAsyncEvents) {
    this->releaseBaseAsyncObject(); // Release current object
    BaseAsyncExecutionSupport::setBaseAsyncObject(asyncObject);
    this->numAsyncEvents = numAsyncEvents;
}
void AsyncExecutionSupport::waitAsync() {
    if (this->executionFlow) {
        this->executionFlow->synchronize();
    } else if (this->asyncObject) {
        #ifdef GSPAR_DEBUG
            std::stringstream ss; // Using stringstream eases multi-threaded debugging
            ss << "[" << std::this_thread::get_id() << " GSPar Async " << this << "] Waiting for " << this->numAsyncEvents << " events: " << this->asyncObject << std::endl;
            std::cout << ss.str();
            ss.str("");

            // CL_QUEUED: 3
            // CL_SUBMITTED: 2
            // CL_RUNNING: 1
            // CL_COMPLETE: 0
            cl_int status;
            throwExceptionIfFailed( clGetEventInfo(*this->asyncObject, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, NULL) );
            // CL_COMMAND_NDRANGE_KERNEL: 4592
            // CL_COMMAND_TASK: 4593
            // CL_COMMAND_NATIVE_KERNEL: 4594
            // CL_COMMAND_READ_BUFFER: 4595
            // CL_COMMAND_WRITE_BUFFER: 4596
            // CL_COMMAND_COPY_BUFFER: 4597
            // CL_COMMAND_READ_IMAGE: 4598
            // CL_COMMAND_WRITE_IMAGE: 4599
            // CL_COMMAND_COPY_IMAGE: 4600
            // CL_COMMAND_COPY_BUFFER_TO_IMAGE: 4602
            // CL_COMMAND_COPY_IMAGE_TO_BUFFER: 4601
            // CL_COMMAND_MAP_BUFFER: 4603
            // CL_COMMAND_MAP_IMAGE: 4604
            // CL_COMMAND_UNMAP_MEM_OBJECT: 4605
            // CL_COMMAND_MARKER: 4606
            // CL_COMMAND_ACQUIRE_GL_OBJECTS: 4607
            // CL_COMMAND_RELEASE_GL_OBJECTS: 4608
            // CL_COMMAND_READ_BUFFER_RECT: 4609
            // CL_COMMAND_WRITE_BUFFER_RECT: 4610
            // CL_COMMAND_COPY_BUFFER_RECT: 4611
            // CL_COMMAND_USER: 4612
            // CL_COMMAND_BARRIER: 4613
            // CL_COMMAND_MIGRATE_MEM_OBJECTS: 4614
            // CL_COMMAND_FILL_BUFFER: 4615
            // CL_COMMAND_FILL_IMAGE: 4616
            // CL_COMMAND_GL_FENCE_SYNC_OBJECT_KHR: 8205
            cl_command_type type;
            throwExceptionIfFailed( clGetEventInfo(*this->asyncObject, CL_EVENT_COMMAND_TYPE, sizeof(cl_command_type), &type, NULL) );

            ss << "[" << std::this_thread::get_id() << " GSPar Async " << this << "] Event " << this->asyncObject << " of type " << type << " is of status " << status << std::endl;
            std::cout << ss.str();
            ss.str("");
        #endif

        throwExceptionIfFailed( clWaitForEvents(this->numAsyncEvents, this->asyncObject) );
    }
    this->releaseBaseAsyncObject();
}
void AsyncExecutionSupport::releaseBaseAsyncObject() {
    if (this->executionFlow) {
        // We don't own this ExecutionFlow, it's just a weak reference, so we don't delete it
        this->executionFlow = nullptr;
    }
    if (this->asyncObject) {
        #ifdef GSPAR_DEBUG
            std::stringstream ss; // Using stringstream eases multi-threaded debugging
            ss << "[" << std::this_thread::get_id() << " GSPar Async " << this << "] Releasing " << this->numAsyncEvents << " events: " << this->asyncObject << std::endl;
            std::cout << ss.str();
            ss.str("");
        #endif
        for (unsigned int i = 0; i < this->numAsyncEvents; i++) {
            throwExceptionIfFailed( clReleaseEvent(this->asyncObject[i]) );
        }
        this->asyncObject = NULL;
    }
    this->clearRunningAsync(); // We can't be running async since we don't have the async objects anymore
}
// static
void AsyncExecutionSupport::waitAllAsync(std::initializer_list<AsyncExecutionSupport*> asyncs) {
    std::vector<cl_event> oclEvents;
    #ifdef GSPAR_DEBUG
        std::stringstream ss; // Using stringstream eases multi-threaded debugging
        ss << "[" << std::this_thread::get_id() << " GSPar Async] Waiting for all async events" << std::endl;
        std::cout << ss.str();
        ss.str("");
    #endif
    for (auto async : asyncs) {
        // std::cout << "Waiting for all cl_events " << async->asyncObject << " " << *async->asyncObject << std::endl;
        oclEvents.insert(oclEvents.end(), async->getBaseAsyncObject(), async->getBaseAsyncObject()+async->numAsyncEvents);
    }
    if (oclEvents.size() > 0) {
        throwExceptionIfFailed( clWaitForEvents(oclEvents.size(), oclEvents.data()) );
    }
    for (auto async : asyncs) {
        async->releaseBaseAsyncObject();
    }
}


///// Instance /////

Instance *Instance::instance = nullptr;

void Instance::loadGpuList() {
    this->clearGpuList();
    
    cl_uint platformCount;
    throwExceptionIfFailed( clGetPlatformIDs(0, NULL, &platformCount) );

    cl_platform_id* platforms = new cl_platform_id[platformCount];
    throwExceptionIfFailed( clGetPlatformIDs(platformCount, platforms, NULL) );

    for (unsigned int i = 0; i < platformCount; ++i) {
        cl_uint deviceCount;
        throwExceptionIfFailed( clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount) );

        cl_device_id* deviceIds = new cl_device_id[deviceCount];
        throwExceptionIfFailed( clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, deviceCount, deviceIds, NULL) );

        for (unsigned int d = 0; d < deviceCount; ++d) {
            this->devices.push_back(new Device(deviceIds[d]));
        }
    }

    delete[] platforms;
}

Instance::Instance() : BaseInstance(Runtime::GSPAR_RT_OPENCL) { }
Instance::~Instance() {
    #ifdef GSPAR_DEBUG
        std::stringstream ss; // Using stringstream eases multi-threaded debugging
        ss << "[" << std::this_thread::get_id() << " GSPar Instance] Deleting Singleton instance " << this << std::endl;
        std::cout << ss.str();
        ss.str("");
    #endif
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
    this->instanceInitiated = true;
}

unsigned int Instance::getGpuCount() {
    unsigned int gpuCount = 0;

    cl_uint platformCount;
    throwExceptionIfFailed( clGetPlatformIDs(0, NULL, &platformCount) );

    cl_platform_id* platforms = new cl_platform_id[platformCount];
    throwExceptionIfFailed( clGetPlatformIDs(platformCount, platforms, NULL) );

    for (unsigned int i = 0; i < platformCount; ++i) {
        cl_uint deviceCount;
        throwExceptionIfFailed( clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount) );

        gpuCount += deviceCount;
    }

    delete[] platforms;

    return gpuCount;
}


///// Device /////

Device::Device() : BaseDevice() { }
Device::Device(cl_device_id device) {
    this->setBaseDeviceObject(device);
}
Device::~Device() {
    // We don't throw exceptions on destructors
    if (this->defaultExecutionFlow) {
        delete this->defaultExecutionFlow;
        this->defaultExecutionFlow = NULL;
    }

    if (this->libContext) {
        #ifdef GSPAR_DEBUG
            std::stringstream ss; // Using stringstream eases multi-threaded debugging
            ss << "[" << std::this_thread::get_id() << " GSPar Device] Releasing context " << this << std::endl;
            std::cout << ss.str();
            ss.str("");
        #endif
        Exception* ex = Exception::checkError( clReleaseContext(this->libContext) );
        if (ex) {
            std::cerr << "Failed when releasing device context on Device's destructor: ";
            std::cerr << ex->what() << " - " << ex->getDetails() << std::endl;
            delete ex;
        }
        this->libContext = NULL;
    }
}
ExecutionFlow* Device::getDefaultExecutionFlow() {
    if (!this->defaultExecutionFlow) {
        this->defaultExecutionFlow = new ExecutionFlow(this);
    }
    return this->defaultExecutionFlow;
}
cl_context Device::getContext() {
    if (!this->libContext) {
        std::lock_guard<std::mutex> lock(this->libContextMutex);
        if (!this->libContext) { // Check if someone changed it while we were waiting for the lock
            cl_int status;
            // TODO add a CL_CALLBACK to get notified of errors. Check opencl versions in test/comparison for an example
            cl_context context = clCreateContext(NULL, 1, &this->libDevice, NULL, NULL, &status);
            throwExceptionIfFailed(status);
            this->setContext(context);
        }
        // Auto-unlock of libContextMutex, RAII
    }
    return this->libContext;
}
cl_command_queue Device::startDefaultExecutionFlow() {
    return this->getDefaultExecutionFlow()->start();
}
const std::string Device::getName() {
    return this->queryInfoDevice<char>(CL_DEVICE_NAME);
}
unsigned int Device::getComputeUnitsCount() {
    return *(this->queryInfoDevice<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS));
}
unsigned int Device::getWarpSize() {
    // TODO warp size is available only for NVIDIA GPUs
    return *(this->queryInfoDevice<size_t>(CL_DEVICE_WARP_SIZE_NV));
}
unsigned int Device::getMaxThreadsPerBlock() {
    return *(this->queryInfoDevice<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE));
}
unsigned long Device::getGlobalMemorySizeBytes() {
    return *(this->queryInfoDevice<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE));
}
unsigned long Device::getLocalMemorySizeBytes() {
    return *(this->queryInfoDevice<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE));
}
unsigned long Device::getSharedMemoryPerComputeUnitSizeBytes() {
    return *(this->queryInfoDevice<cl_ulong>(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE));
}
unsigned int Device::getClockRateMHz() {
    return *(this->queryInfoDevice<cl_uint>(CL_DEVICE_MAX_CLOCK_FREQUENCY));
}
bool Device::isIntegratedMainMemory() {
    // CL_DEVICE_HOST_UNIFIED_MEMORY is deprecated in OpenCL 1.2
    // should probably use CL_DEVICE_SVM_CAPABILITIES instead in OpenCL 2.0
    return *(this->queryInfoDevice<cl_bool>(CL_DEVICE_HOST_UNIFIED_MEMORY));
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
    return new Kernel(this, kernel_source, kernel_name);
}
std::vector<Kernel*> Device::prepareKernels(const std::string kernelSource, const std::vector<std::string> kernelNames) {
    cl_program oclProgram = this->compileOCLProgram(kernelSource);

    std::vector<Kernel*> kernels;
    for (auto name : kernelNames) {
        kernels.push_back(new Kernel(this, oclProgram, name));
    }
    return kernels;
}
template<class T>
const T* Device::queryInfoDevice(cl_device_info paramName, bool cacheable) {
    //https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clGetDeviceInfo.html
    if (cacheable) { // Check if the attribute is cached
        std::lock_guard<std::mutex> lock(this->attributeCacheMutex); // Auto-unlock, RAII
        auto it = this->attributeCache.find(paramName);
        if (it != this->attributeCache.end()) {
            return (T*)it->second;
        }
    }

    size_t valueSize;
    clGetDeviceInfo(this->getBaseDeviceObject(), paramName, 0, NULL, &valueSize);
    T* value = new T[valueSize];
    clGetDeviceInfo(this->getBaseDeviceObject(), paramName, valueSize, value, NULL);
    if (cacheable) { // Stores the attribute in cache
        std::lock_guard<std::mutex> lock(this->attributeCacheMutex); // Auto-unlock, RAII
        this->attributeCache[paramName] = value;
    }
    return value;
}
cl_program Device::compileOCLProgram(std::string source) {
#ifdef GSPAR_DEBUG
    std::stringstream ss; // Using stringstream eases multi-threaded debugging
    ss << "[GSPar Device " << this << "] Kernel received to compile: \n" << source << std::endl;
    std::cout << ss.str();
    ss.str("");
#endif

    std::string openclExtensions = "#pragma OPENCL EXTENSION all: enable\n";
    std::string completeKernelSource = "";
    completeKernelSource.append(openclExtensions);
    completeKernelSource.append(Instance::getInstance()->getKernelGenerator()->generateStdFunctions());
    completeKernelSource.append(Instance::getInstance()->getKernelGenerator()->replaceMacroKeywords(source));

#ifdef GSPAR_DEBUG
    ss << "[GSPar Device " << this << "] Complete kernel for compilation: \n" << completeKernelSource << std::endl;
    std::cout << ss.str();
    ss.str("");
#endif

    cl_program oclProgram;
    cl_device_id devId = this->getBaseDeviceObject();

    // Place for inserting any additional macros
    std::string macrosGspar = "";
    macrosGspar.append("-D GSPAR_DEVICE_KERNEL=" + KernelGenerator::KERNEL_PREFIX);
    macrosGspar.append(" -D GSPAR_DEVICE_GLOBAL_MEMORY=" + KernelGenerator::GLOBAL_MEMORY_PREFIX);
    macrosGspar.append(" -D GSPAR_DEVICE_SHARED_MEMORY=" + KernelGenerator::SHARED_MEMORY_PREFIX);
    macrosGspar.append(" -D GSPAR_DEVICE_CONSTANT=" + KernelGenerator::CONSTANT_PREFIX);
    macrosGspar.append(" -D GSPAR_DEVICE_FUNCTION=" + KernelGenerator::DEVICE_FUNCTION_PREFIX);
    const char *compilationOptions = macrosGspar.c_str();

#ifdef GSPAR_DEBUG
    ss << "[GSPar Device " << this << "] Compiling kernel with arguments: " << compilationOptions;
    ss << std::endl;
    std::cout << ss.str();
    ss.str("");
#endif

    cl_int status;
    const char* src = completeKernelSource.c_str();
    oclProgram = clCreateProgramWithSource(this->getContext(), 1, &src, NULL, &status);
    Exception::throwIfFailed(status, oclProgram, devId);

    status = clBuildProgram(oclProgram, 1, &devId, compilationOptions, NULL, NULL);
    Exception::throwIfFailed(status, oclProgram, devId);

    return oclProgram;
}

///// Kernel /////

void Kernel::loadOclKernel(const std::string kernelName) {
    cl_int status;
    this->oclKernel = clCreateKernel(this->oclProgram, kernelName.c_str(), &status);
    Exception::throwIfFailed(status, this->oclProgram, this->device->getBaseDeviceObject());
    this->kernelName = kernelName;
}

Kernel::Kernel() : BaseKernel() { }
Kernel::Kernel(Device* device, const std::string kernelSource, const std::string kernelName) : BaseKernel(device, kernelSource, kernelName) {
    this->oclProgram = device->compileOCLProgram(kernelSource);

    this->isPrecompiled = false; //Kernel owns oclProgram

    this->loadOclKernel(kernelName);
}
Kernel::Kernel(Device* device, cl_program oclProgram, const std::string kernelName) : BaseKernel(device) {
    this->oclProgram = oclProgram;
    this->isPrecompiled = true; //Kernel shares oclProgram

    this->loadOclKernel(kernelName);
}
Kernel::~Kernel() {
    #ifdef GSPAR_DEBUG
        std::stringstream ss; // Using stringstream eases multi-threaded debugging
    #endif
    if (!this->isPrecompiled && this->oclProgram) {
        #ifdef GSPAR_DEBUG
            ss << "[" << std::this_thread::get_id() << " GSPar Kernel] Releasing oclProgram " << this << std::endl;
            std::cout << ss.str();
            ss.str("");
        #endif
        Exception* ex = Exception::checkError( clReleaseProgram(this->oclProgram) ); // We don't throw exceptions on destructors
        if (ex != nullptr) {
            std::cerr << "Failed when releasing OpenCL program on Kernel destructor: ";
            std::cerr << ex->what() << " - " << ex->getDetails() << std::endl;
            delete ex;
        }
        this->oclProgram = NULL;
    }
    if (this->oclKernel) {
        #ifdef GSPAR_DEBUG
            ss << "[" << std::this_thread::get_id() << " GSPar Kernel " << this << "] Releasing oclKernel" << std::endl;
            std::cout << ss.str();
            ss.str("");
        #endif
        Exception* ex = Exception::checkError( clReleaseKernel(this->oclKernel) ); // We don't throw exceptions on destructors
        if (ex != nullptr) {
            std::cerr << "Failed when releasing OpenCL kernel on Kernel destructor: ";
            std::cerr << ex->what() << " - " << ex->getDetails() << std::endl;
            delete ex;
        }
        this->oclKernel = NULL;
    }
}
void Kernel::cloneInto(BaseKernelBase* baseOther) {
    BaseKernel::cloneInto(baseOther);
    Kernel* other = static_cast<Kernel*>(baseOther);
    other->oclProgram = this->oclProgram;
    // cl_kernel objects are not thread-safe (OpenCL 1.2 Specification p. 360)
    other->loadOclKernel(this->kernelName);
    // We do not mark this kernel as precompiled, so it destroys the cl_program on destructor.
    // However, once it is destroyed, the cloned pattern cannot be further cloned because we need the program to call clCreateKernel (called during the clone process)
    // TODO I haven't tested, but this probably causes issues
    // this->isPrecompiled = true;
    other->isPrecompiled = true;
}
int Kernel::setParameter(MemoryObject* memoryObject) {
    cl_mem oclObject = memoryObject->getBaseMemoryObject();
    throwExceptionIfFailed( clSetKernelArg(this->oclKernel, this->parameterCount++, sizeof(cl_mem), &oclObject) );
    return this->parameterCount;
}
int Kernel::setParameter(ChunkedMemoryObject* chunkedMemoryObject) {
    cl_mem oclObject = chunkedMemoryObject->getBaseMemoryObject();
    throwExceptionIfFailed( clSetKernelArg(this->oclKernel, this->parameterCount++, sizeof(cl_mem), &oclObject) );
    return this->parameterCount;
}
int Kernel::setParameter(size_t parm_size, void* parm) {
    // clSetKernelArg expects a const void*, so we can treat all pointers as const
    return this->setParameter(parm_size, const_cast<const void*>(parm));
}
int Kernel::setParameter(size_t parm_size, const void* parm) {
    // The argument data pointed to by arg_value is copied and the arg_value pointer can therefore be reused by the application after clSetKernelArg returns.
    // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clSetKernelArg.html
    throwExceptionIfFailed( clSetKernelArg(this->oclKernel, this->parameterCount++, parm_size, parm) );
    return this->parameterCount;
}
GSPar::Driver::Dimensions Kernel::getNumBlocksAndThreadsFor(Dimensions dims) {

    // CL_DEVICE_MAX_WORK_GROUP_SIZE is usually 1024, but CL_KERNEL_WORK_GROUP_SIZE is 256.
    // In general, the kernels works just fine with 1024 even with the 256 limitation reported by CL_KERNEL_WORK_GROUP_SIZE.
    // What limit should we use?
    // unsigned int maxThreadsPerBlock = this->device->getMaxThreadsPerBlock(); //CL_DEVICE_MAX_WORK_GROUP_SIZE
    const size_t *kernelWorkGroupSize = this->queryInfo<size_t>(CL_KERNEL_WORK_GROUP_SIZE);
    unsigned int maxThreadsPerBlock = *kernelWorkGroupSize;

    // Should we check CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS? We only support 3 dimensions anyway.
    const size_t* maxWorkItemSizes = this->device->queryInfoDevice<size_t>(CL_DEVICE_MAX_WORK_ITEM_SIZES);
    size_t maxThreadsDimension[3]; // Copying data to remove constness
    memcpy(maxThreadsDimension, maxWorkItemSizes, sizeof(size_t) * 3);

    return this->getNumBlocksAndThreads(dims, maxThreadsPerBlock, maxThreadsDimension);
}
void Kernel::runAsync(Dimensions dims, ExecutionFlow* executionFlow) {

    #ifdef GSPAR_DEBUG
        std::stringstream ss; // Using stringstream eases multi-threaded debugging
        ss << "[" << std::this_thread::get_id() << " GSPar Kernel " << this << "] Running kernel async with " << this->parameterCount << " parameters for " << dims.toString() << " in flow " << executionFlow << std::endl;
        std::cout << ss.str();
        ss.str("");
    #endif

    cl_command_queue oclQueue = ExecutionFlow::checkAndStartFlow(this->device, executionFlow);

    if (!dims.x) {
        throw Exception("The first dimension is required to run a kernel");
    }

    Dimensions blocksAndThreads = this->getNumBlocksAndThreadsFor(dims);

    int dimensions = dims.getCount();

    size_t localSize[dimensions];
    size_t globalSize[dimensions];
    for (int d = 0; d < dimensions; d++) {
        localSize[d] = blocksAndThreads[d].max;
        globalSize[d] = blocksAndThreads[d].min * localSize[d];
    }

    // Set shared memory - https://community.khronos.org/t/dynamically-allocated-shared-memory/1562
    if (this->sharedMemoryBytes > 0) {
        throwExceptionIfFailed( clSetKernelArg(this->oclKernel, this->parameterCount++, this->sharedMemoryBytes, NULL) );
    }

    #ifdef GSPAR_DEBUG
        ss << "[" << std::this_thread::get_id() << " GSPar Kernel " << this << "] Shall start " << dims.toString() << " threads: ";
        ss << "starting (" << globalSize[0];
        if (dims.y) ss << "," << globalSize[1];
        if (dims.z) ss << "," << globalSize[2];
        ss << ") threads ";
        ss << "divided in blocks of (" << localSize[0];
        if (dims.y) ss << "," << localSize[1];
        if (dims.z) ss << "," << localSize[2];
        ss << ") threads ";
        ss << "using " << this->sharedMemoryBytes << " bytes of shared memory in execution flow " << executionFlow << std::endl;
        std::cout << ss.str();
        ss.str("");
    #endif

    cl_event *evt = new cl_event;
    throwExceptionIfFailed( clEnqueueNDRangeKernel(oclQueue, this->oclKernel, dimensions, NULL, globalSize, localSize, 0, NULL, evt) );
    #ifdef GSPAR_DEBUG
        ss << "[" << std::this_thread::get_id() << " GSPar Kernel " << this << "] Setting evt to wait: " << evt << std::endl;
        std::cout << ss.str();
        ss.str("");
    #endif

    this->setBaseAsyncObject(evt); // setBaseAsyncObject sets runningAsync to false
    // Use Execution Flow instead of the event for synchronization. See comment on executionFlow attribute.
    this->setExecutionFlowToSynchronize(executionFlow ? executionFlow : this->device->getDefaultExecutionFlow());
    this->runningAsync = true;
}
template<class T>
T* Kernel::queryInfo(cl_kernel_work_group_info param, bool cacheable) {
    // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clGetKernelWorkGroupInfo.html
    if (cacheable) { // Check if the attribute is cached
        auto it = this->attributeCache.find(param);
        if (it != this->attributeCache.end()) {
            return (T*)it->second;
        }
    }

    size_t valueSize;
    throwExceptionIfFailed( clGetKernelWorkGroupInfo(this->oclKernel, this->device->getBaseDeviceObject(), param, 0, NULL, &valueSize) );
    T* value = new T[valueSize];
    throwExceptionIfFailed( clGetKernelWorkGroupInfo(this->oclKernel, this->device->getBaseDeviceObject(), param, valueSize, value, NULL) );
    if (cacheable) { // Stores the attribute in cache
        this->attributeCache[param] = value;
    }
    return value;
}


///// MemoryObject /////

void MemoryObject::copy(bool in, bool async, ExecutionFlow* executionFlow) {
    cl_event *evt = new cl_event;
    cl_bool blocking = async ? CL_FALSE : CL_TRUE;
    int numEvtsToWait = 0;
    cl_event *evtToWait = NULL;
    if (this->getBaseAsyncObject()) {
        numEvtsToWait = this->numAsyncEvents;
        evtToWait = this->asyncObject;
        #ifdef GSPAR_DEBUG
            std::stringstream ss; // Using stringstream eases multi-threaded debugging
            ss << "[" << std::this_thread::get_id() << " GSPar MemObj " << this << "] Already has an async event: " << evtToWait << ", binding two events" << std::endl;
            std::cout << ss.str();
            ss.str("");
        #endif
    }

    cl_command_queue oclQueue = ExecutionFlow::checkAndStartFlow(this->device, executionFlow);

    if (in) {
        throwExceptionIfFailed( clEnqueueWriteBuffer(
            oclQueue, this->devicePtr,
            blocking, 0, this->size, this->hostPtr,
            numEvtsToWait, evtToWait, evt) );
    } else { //copy out
        throwExceptionIfFailed( clEnqueueReadBuffer(
            oclQueue, this->devicePtr,
            blocking, 0, this->size, this->hostPtr,
            numEvtsToWait, evtToWait, evt) );
    }
    if (this->getBaseAsyncObject()) { // Releases old async event handler
        this->releaseBaseAsyncObject();
    }
    if (async) {
        #ifdef GSPAR_DEBUG
            std::stringstream ss; // Using stringstream eases multi-threaded debugging
            ss << "[" << std::this_thread::get_id() << " GSPar MemObj " << this << "] Setting evt " << evt << " from queue " << oclQueue << " to wait" << std::endl;
            std::cout << ss.str();
            ss.str("");
        #endif

        this->setBaseAsyncObject(evt); // setBaseAsyncObject sets runningAsync to false
        // Use Execution Flow instead of the event for synchronization. See comment on executionFlow attribute.
        this->setExecutionFlowToSynchronize(executionFlow ? executionFlow : this->device->getDefaultExecutionFlow());
        this->runningAsync = true;
    }
}

void MemoryObject::allocDeviceMemory() {
    cl_int status;

    // Security check is already done in base class
    cl_mem_flags ocl_flags = CL_MEM_READ_WRITE;
    if (this->isReadOnly()) {
        ocl_flags = CL_MEM_READ_ONLY;
    } else if (this->isWriteOnly()) {
        ocl_flags = CL_MEM_WRITE_ONLY;
    }

    this->devicePtr = clCreateBuffer(device->getContext(), ocl_flags, size, NULL, &status);
    throwExceptionIfFailed(status);
}
MemoryObject::MemoryObject(Device* device, size_t size, void* hostPtr, bool readOnly, bool writeOnly) : BaseMemoryObject(device, size, hostPtr, readOnly, writeOnly) {
    this->allocDeviceMemory();
}
MemoryObject::MemoryObject(Device* device, size_t size, const void* hostPtr) : BaseMemoryObject(device, size, hostPtr) {
    this->allocDeviceMemory();
}
MemoryObject::~MemoryObject() {
    if (this->devicePtr) {
        #ifdef GSPAR_DEBUG
            std::stringstream ss; // Using stringstream eases multi-threaded debugging
            ss << "[" << std::this_thread::get_id() << " GSPar MemObj] Releasing Memory Object " << this << std::endl;
            std::cout << ss.str();
            ss.str("");
        #endif
        Exception* ex = Exception::checkError( clReleaseMemObject(this->devicePtr) ); // We don't throw exceptions on destructors
        if (ex != nullptr) {
            std::cerr << "Failed when releasing OpenCL memory object: ";
            std::cerr << ex->what() << " - " << ex->getDetails() << std::endl;
            delete ex;
        }
        this->devicePtr = NULL;
    }
}
void MemoryObject::copyIn() { copy(true, false); }
void MemoryObject::copyOut() { copy(false, false); }
void MemoryObject::copyInAsync(ExecutionFlow* executionFlow) { copy(true, true, executionFlow); }
void MemoryObject::copyOutAsync(ExecutionFlow* executionFlow) { copy(false, true, executionFlow); }


///// ChunkedMemoryObject /////

void ChunkedMemoryObject::copy(bool in, bool async, unsigned int chunkFrom, unsigned int chunkTo, ExecutionFlow* executionFlow) {
    unsigned int numChunksToCopy = chunkTo - chunkFrom;
    cl_event *newEvents = new cl_event[numChunksToCopy];

    cl_bool blocking = async ? CL_FALSE : CL_TRUE;
    unsigned int currentNumEvents = 0;
    cl_event *currentEvents = NULL;
    if (this->getBaseAsyncObject()) {
        currentNumEvents = this->numAsyncEvents;
        currentEvents = this->asyncObject;
    }

    cl_command_queue oclQueue = ExecutionFlow::checkAndStartFlow(this->device, executionFlow);

    for (unsigned int chunk = chunkFrom, evtIdx = 0; chunk < chunkTo; chunk++, evtIdx++) {
        if (in) {
            throwExceptionIfFailed( clEnqueueWriteBuffer(
                oclQueue, this->devicePtr,
                blocking, chunk * this->getChunkSize(), this->getChunkSize(), this->hostPointers[chunk],
                currentNumEvents, currentEvents, &newEvents[evtIdx]) );
        } else { //copy out
            throwExceptionIfFailed( clEnqueueReadBuffer(
                oclQueue, this->devicePtr,
                blocking, chunk * this->getChunkSize(), this->getChunkSize(), this->hostPointers[chunk],
                currentNumEvents, currentEvents, &newEvents[evtIdx]) );
        }
    }
    if (this->getBaseAsyncObject()) { // Releases old async event handler
        this->releaseBaseAsyncObject();
    }
    if (async) {
        #ifdef GSPAR_DEBUG
            std::stringstream ss; // Using stringstream eases multi-threaded debugging
            ss << "[" << std::this_thread::get_id() << " GSPar ChunkedMemObj " << this << "] Setting evts (" << numChunksToCopy << ") to wait: " << newEvents << std::endl;
            std::cout << ss.str();
            ss.str("");
        #endif
        this->setBaseAsyncObject(newEvents, numChunksToCopy); // setBaseAsyncObject sets runningAsync to false
        // Use Execution Flow instead of the event for synchronization. See comment on executionFlow attribute.
        this->setExecutionFlowToSynchronize(executionFlow ? executionFlow : this->device->getDefaultExecutionFlow());
        this->runningAsync = true;
    }
}

void ChunkedMemoryObject::allocDeviceMemory() {
    cl_int status;

    // Security check is already done in base class
    cl_mem_flags ocl_flags = CL_MEM_READ_WRITE;
    if (this->isReadOnly()) {
        ocl_flags = CL_MEM_READ_ONLY;
    } else if (this->isWriteOnly()) {
        ocl_flags = CL_MEM_WRITE_ONLY;
    }

    // We allocate space for all the memory chunks
    this->devicePtr = clCreateBuffer(device->getContext(), ocl_flags, this->getChunkSize() * this->chunks, NULL, &status);
    throwExceptionIfFailed(status);
}
ChunkedMemoryObject::ChunkedMemoryObject(Device* device, unsigned int chunks, size_t chunkSize, void** hostPointers, bool readOnly, bool writeOnly) :
        BaseChunkedMemoryObject(device, chunks, chunkSize, hostPointers, readOnly, writeOnly) {
    this->allocDeviceMemory();
}
ChunkedMemoryObject::ChunkedMemoryObject(Device* device, unsigned int chunks, size_t chunkSize, const void** hostPointers) :
        BaseChunkedMemoryObject(device, chunks, chunkSize, hostPointers) {
    this->allocDeviceMemory();
}
ChunkedMemoryObject::~ChunkedMemoryObject() {
    //devicePtr is released in ~MemoryObject
}
void ChunkedMemoryObject::copyIn() { copy(true, false, 0, this->chunks); }
void ChunkedMemoryObject::copyOut() { copy(false, false, 0, this->chunks); }
void ChunkedMemoryObject::copyInAsync(ExecutionFlow* executionFlow) { copy(true, true, 0, this->chunks, executionFlow); }
void ChunkedMemoryObject::copyOutAsync(ExecutionFlow* executionFlow) { copy(false, true, 0, this->chunks, executionFlow); }
void ChunkedMemoryObject::copyIn(unsigned int chunk) { copy(true, false, chunk, chunk+1); }
void ChunkedMemoryObject::copyOut(unsigned int chunk) { copy(false, false, chunk, chunk+1); }
void ChunkedMemoryObject::copyInAsync(unsigned int chunk, ExecutionFlow* executionFlow) { copy(true, true, chunk, chunk+1, executionFlow); }
void ChunkedMemoryObject::copyOutAsync(unsigned int chunk, ExecutionFlow* executionFlow) { copy(false, true, chunk, chunk+1, executionFlow); }


///// StreamElement /////

StreamElement::StreamElement(Device* device) : BaseStreamElement(device) {
    // Can't call this virtual function in the base constructor
    this->start();
}

StreamElement::~StreamElement() { }


///// KernelGenerator /////

const std::string KernelGenerator::KERNEL_PREFIX = "__kernel";
const std::string KernelGenerator::GLOBAL_MEMORY_PREFIX = "__global";
const std::string KernelGenerator::SHARED_MEMORY_PREFIX = "__local";
const std::string KernelGenerator::CONSTANT_PREFIX = "__constant";
const std::string KernelGenerator::DEVICE_FUNCTION_PREFIX = "";

const std::string KernelGenerator::getKernelPrefix() {
    return KernelGenerator::KERNEL_PREFIX + " void";
}
std::string KernelGenerator::generateStdFunctions() {
    return ""
    "size_t gspar_get_global_id(unsigned int dimension) { return get_global_id(dimension); } \n"
    "size_t gspar_get_thread_id(unsigned int dimension) { return get_local_id(dimension); } \n"
    "size_t gspar_get_block_id(unsigned int dimension) { return get_group_id(dimension); } \n"
    "size_t gspar_get_block_size(unsigned int dimension) { return get_local_size(dimension); } \n"
    "size_t gspar_get_grid_size(unsigned int dimension) { return get_num_groups(dimension); } \n"
    "void gspar_synchronize_local_threads() { barrier(CLK_LOCAL_MEM_FENCE); } \n"
    "int gspar_atomic_add_int(__global int *valq, int delta){ atomic_add(valq, delta); } \n"
    "double gspar_atomic_add_double(__global double *valq, double delta){ \n "
    "    union { double f; unsigned long i; } old; \n"
    "    union { double f; unsigned long i; } new1; \n"
    "    do { \n"
    "        old.f = *valq; \n"
    "        new1.f = old.f + delta; \n"
    "    } while (atom_cmpxchg((volatile __global unsigned long *)valq, old.i, new1.i) != old.i); \n"
    "    return old.f; \n"
    "} \n"
    ;
}
std::string KernelGenerator::replaceMacroKeywords(std::string kernelSource) {
    kernelSource = std::regex_replace(kernelSource, std::regex("GSPAR_DEVICE_MACRO_BEGIN"), "#define");
    kernelSource = std::regex_replace(kernelSource, std::regex("GSPAR_DEVICE_MACRO_END"), "\n");
    return kernelSource;
}
std::string KernelGenerator::generateInitKernel(Pattern::BaseParallelPattern* pattern, Dimensions max) {
    return "";
}
std::string KernelGenerator::generateParams(Pattern::BaseParallelPattern* pattern, Dimensions dims) {
    std::string r = "";
    for(int d = 0; d < dims.getCount(); d++) {
        if (dims.is(d)) {
            std::string varName = this->getStdVarNameForDimension(pattern->getStdVarNames(), d);
            r += "const unsigned long gspar_max_" + varName + ",";
            if (dims[d].min && !pattern->isBatched()) {
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
            if (param->paramValueType == Pattern::ParameterValueType::GSPAR_PARAM_POINTER || param->isBatched()) { // Batched values are always pointers
                r += KernelGenerator::GLOBAL_MEMORY_PREFIX + " ";
            }
            if (param->direction == Pattern::ParameterDirection::GSPAR_PARAM_IN && param->isConstant()) {
                r += "const ";
            }
            r += param->toKernelParameter() + ",";
        }
    }
    if (pattern->isUsingSharedMemory()) {
        auto shmem = pattern->getSharedMemoryParameter();
        r += KernelGenerator::SHARED_MEMORY_PREFIX + " " + shmem->toString();
    } else {
        if (!r.empty()) r.pop_back(); // removes last comma
    }
    return r;
}
std::string KernelGenerator::generateStdVariables(Pattern::BaseParallelPattern* pattern, Dimensions dims) {
    std::array<std::string, 3> patternNames = pattern->getStdVarNames();

    // OpenCL get_global_id returns a size_t, so this is the type of our std variables
    // https://www.khronos.org/registry/OpenCL/specs/opencl-1.2.pdf#page=244
    std::string r;
    for(int d = 0; d < dims.getCount(); d++) {
        if (dims.is(d)) {
            std::string varName = this->getStdVarNameForDimension(patternNames, d);
            if (pattern->isBatched()) {
                r += "size_t gspar_global_" + varName;
            } else {
                r += "size_t " + varName;
            }
            r += " = gspar_get_global_id(" + std::to_string(d) + ")";
            if (dims[d].min && !pattern->isBatched()) {
                // TODO Support min in batches
                r += " + gspar_min_" + varName;
            }
            r += "; \n";
            // TODO Support multi-dimensional batches
            if (pattern->isBatched()) {
                // Intended implicit floor(gspar_global/dims)
                r += "size_t gspar_batch_" + varName + " = ((size_t)(gspar_global_" + varName + " / " + std::to_string(dims[d].max) + ")); \n";
                r += "size_t gspar_offset_" + varName + " = gspar_batch_" + varName + " * " + std::to_string(dims[d].max) + "; \n";
                // This variable names are used in other methods, keep track
                r += "size_t " + varName + " = gspar_global_" + varName + " - gspar_offset_" + varName + "; \n";
            }
        }
    }
    return r;
}
std::string KernelGenerator::generateBatchedParametersInitialization(Pattern::BaseParallelPattern* pattern, Dimensions dims) {
    std::array<std::string, 3> patternNames = pattern->getStdVarNames();
    // TODO Support multi-dimensional batches
    std::string stdVarFirstDimension = this->getStdVarNameForDimension(patternNames, 0);

    std::string r = "";
    for(auto &param : pattern->getParameterList()) {
        if (param->isBatched()) {
            if (param->paramValueType == Pattern::ParameterValueType::GSPAR_PARAM_POINTER) {
                r += "__global ";
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
