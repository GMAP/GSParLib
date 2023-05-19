#include <iostream>

#include "GSPar_PatternReduce.hpp"

using namespace GSPar::Pattern;

PointerParameter* Reduce::getOutputParameter() {
    auto param = this->getParameter(this->outputParameterName);
    if (!param) {
        throw GSParException("Could not find output parameter with name '" + this->outputParameterName + "' in Reduce pattern");
    }
    return static_cast<PointerParameter*>(param);
}

PointerParameter* Reduce::generateSharedMemoryParameter(Driver::Dimensions dims, Driver::BaseKernelBase *kernel) {
    if (dims.y || dims.z) {
        // TODO support multiple dimensions
        throw GSParException("Reduce pattern currently does not support multi-dimensional kernels");
    }

    // if (this->sharedMemoryParameter == nullptr || !this->sharedMemoryParameter->isComplete()) {
        this->getSharedMemoryParameter(); // Generate the placeholder parameter

        std::lock_guard<std::mutex> lock(this->sharedMemoryParameterMutex); // Auto-unlock, RAII
        if (!this->sharedMemoryParameter->isComplete()) { // Check if there was a race condition for this resource
            Driver::Dimensions blocksAndThreads = kernel->getNumBlocksAndThreadsFor(dims);
            size_t sharedMemSize = (dims.x.max > blocksAndThreads.x.max) ? blocksAndThreads.x.max : dims.x.max;

            auto outParam = this->getOutputParameter();
            this->sharedMemoryParameter->numberOfElements = sharedMemSize;
            this->sharedMemoryParameter->size = outParam->size * sharedMemSize;
            this->sharedMemoryParameter->setComplete(true);
        }
        // Auto-unlock of sharedMemoryParameterMutex, RAII
    // }
    return this->sharedMemoryParameter;
}

PointerParameter* Reduce::getSharedMemoryParameter() {
    if (this->sharedMemoryParameter == nullptr) {
        std::lock_guard<std::mutex> lock(this->sharedMemoryParameterMutex); // Auto-unlock, RAII
        if (this->sharedMemoryParameter == nullptr) { // Check if there was a race condition for this resource
            auto outParam = this->getOutputParameter();
            std::string paramName = "gspar_shared_" + getRandomString(5);
            this->sharedMemoryParameter = new PointerParameter(paramName, outParam->type, 0, nullptr);
        }
        // Auto-unlock of sharedMemoryParameterMutex, RAII
    }
    return this->sharedMemoryParameter;
};

std::string Reduce::getKernelCore(Driver::Dimensions dims, std::array<std::string, 3> stdVarNames) {
    if (dims.y || dims.z) {
        // TODO support multiple dimensions
        throw GSParException("Reduce pattern currently does not support multi-dimensional kernels");
    }

    PointerParameter *outParam = this->getOutputParameter();
    auto shmemParam = this->getSharedMemoryParameter();
    std::string shmem = shmemParam->name;

    std::string op = this->binaryOperation;
    std::string gid = stdVarNames[0];
    std::string max = "gspar_max_" + stdVarNames[0];
    std::string tid = "gspar_tid_" + stdVarNames[0];
    std::string bid = "gspar_bid_" + stdVarNames[0];
    std::string bsize = "gspar_bsize_" + stdVarNames[0];
    
    // TODO support batches and min-max in Reduce

    // https://devblogs.nvidia.com/using-shared-memory-cuda-cc/
    // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    std::string kernelSource =
    "   size_t " + tid + " = gspar_get_thread_id(0); \n"
    "   size_t " + bid + " = gspar_get_block_id(0); \n"
    "   size_t " + bsize + " = gspar_get_block_size(0); \n"
    "   " + shmem + "["+tid+"] = " + this->vectorName + "["+gid+"]; \n"
    "   gspar_synchronize_local_threads(); \n"

    "   for (unsigned int s="+bsize+"/2; s>0; s>>=1) { \n"
    "       if ("+tid+" < s && "+gid+"+s < "+max+") { \n"
    "           "+shmem+"["+tid+"] = "+shmem+"["+tid+"]" + op + shmem+"["+tid+"+s]; \n"
    "       } \n"
    "       gspar_synchronize_local_threads(); \n"
    "       if ("+tid+" == 0 && s > 1 && s % 2 != 0) { \n"
    "           "+shmem+"["+tid+"] = "+shmem+"["+tid+"]" + op + shmem+"[s-1]; \n"
    "       } \n"
    "       gspar_synchronize_local_threads(); \n"
    "   } \n"
    "   if ("+tid+" == 0) { \n"
    "       if ("+bsize+" % 2 != 0) { \n"
    "           "+shmem+"[0] = "+shmem+"[0]" + op + shmem+"["+max+"-1]; \n"
    "       } \n"
    "       " + this->partialTotalsParamName + "["+bid+"] = "+shmem+"[0]; \n"
    // If the param is input, we reduce it together in the end
    + (outParam->isIn() ?
    "       if (gspar_get_grid_size(0) == 1) { \n"
    "           " + this->partialTotalsParamName+"["+bid+"] = " + this->partialTotalsParamName+"["+bid+"]" + op + "*" + outParam->name + "; \n"
    "       } \n"
        : "") +
    "   } \n"
    ;

    return kernelSource;
};

bool Reduce::isKernelCompiledFor(Driver::Dimensions dims) {
    // We only compile if the kernel wasn't compiled yet and the configuration didn't change
    return this->_isKernelCompiled && !this->isKernelStale && this->compiledKernelDimension.getCount() == dims.getCount();
}

void Reduce::callbackBeforeGeneratingKernelSource() {
    auto partialTotalsParam = this->getParameter(this->partialTotalsParamName);
    if (!partialTotalsParam) {
        #ifdef GSPAR_DEBUG
            std::stringstream ss;
            ss << "[GSPar Reduce "<<this<<"] Adding parameter for Reduce partial totals (" << this->partialTotalsParamName << ")" << std::endl;
            std::cout << ss.str();
            ss.str("");
        #endif
        // It is a ParameterPlaceholder, but we don't have the type here to call the proper function
        auto outParam = this->getOutputParameter();
        VarType partialsTotalsType = outParam->type;
        if (!partialsTotalsType.isPointer) {
            partialsTotalsType.name += "*";
            partialsTotalsType.isPointer = true;
        }
        this->setPointerParameter(this->partialTotalsParamName, partialsTotalsType, 0, nullptr, GSPAR_PARAM_OUT);
    }
}

void Reduce::callbackBeforeAllocatingMemoryOnGpu(Driver::Dimensions dims, Driver::BaseKernelBase *kernel) {
    auto partialTotalsParam = this->getParameter(this->partialTotalsParamName);
    if (!partialTotalsParam || !partialTotalsParam->isComplete()) {
        // TODO we could use the previous value (~15 lines above)
        Driver::Dimensions blocksAndThreads = kernel->getNumBlocksAndThreadsFor(dims);
        auto outParam = this->getOutputParameter();
        
        size_t partialTotalsSize = blocksAndThreads.x.min * outParam->size; // Number of blocks * data size
        // Should we store this pointer in a class-wide attribute?
        void *partialTotals = malloc(partialTotalsSize);
        #ifdef GSPAR_DEBUG
            std::stringstream ss;
            ss << "[GSPar Reduce "<<this<<"] Setting parameter for Reduce partial totals (" << this->partialTotalsParamName << ") as " << partialTotals << " (pointer of " << partialTotalsSize << " bytes)" << std::endl;
            std::cout << ss.str();
            ss.str("");
        #endif
        VarType partialsTotalsType = outParam->type;
        if (!partialsTotalsType.isPointer) {
            partialsTotalsType.name += "*";
            partialsTotalsType.isPointer = true;
        }
        this->setPointerParameter(this->partialTotalsParamName, partialsTotalsType, partialTotalsSize, partialTotals, GSPAR_PARAM_OUT);
    }
}
