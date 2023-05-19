
#ifndef __GSPAR_PATTERNREDUCE_INCLUDED__
#define __GSPAR_PATTERNREDUCE_INCLUDED__

#include "GSPar_BaseParallelPattern.hpp"

namespace GSPar {
    namespace Pattern {

        /**
         * Reduce parallel pattern
         */
        class Reduce : public BaseParallelPattern {
        private:
            const std::string partialTotalsParamName = "gspar_partial_reductions";
            PointerParameter* getOutputParameter();

        protected:
            std::string vectorName;
            std::string binaryOperation; // https://northstar-www.dartmouth.edu/doc/ibmcxx/en_US/doc/language/ref/ruclxbin.htm
            std::string outputParameterName;

            PointerParameter* generateSharedMemoryParameter(Driver::Dimensions dims, Driver::BaseKernelBase *kernel) override;
            PointerParameter* getSharedMemoryParameter() override;

        public:
            Reduce() : BaseParallelPattern() { };
            Reduce(std::string vectorName, std::string binaryOperation, std::string outputParameterName) : BaseParallelPattern("") {
                this->vectorName = vectorName;
                this->binaryOperation = binaryOperation;
                this->outputParameterName = outputParameterName;
                this->useSharedMemory = true;
            };

            template<class TDriverInstance>
            Reduce* clone() const {
                Reduce* other = new Reduce();
                this->cloneInto<TDriverInstance>(other);
                other->vectorName = this->vectorName;
                other->binaryOperation = this->binaryOperation;
                other->outputParameterName = this->outputParameterName;
                return other;
            };

            std::string getKernelCore(Driver::Dimensions dims, std::array<std::string, 3> stdVarNames) override;

            bool isKernelCompiledFor(Driver::Dimensions dims) override;

            // Callback override
            void callbackBeforeGeneratingKernelSource() override;
            void callbackBeforeAllocatingMemoryOnGpu(Driver::Dimensions dims, Driver::BaseKernelBase *kernel) override;

            // Main run function for Reduce Pattern
            // TODO this does not override base class due to templates. Fix this.
            template<class TDriverInstance>
            void run(Driver::Dimensions dimsToUse) {
                if (dimsToUse.y || dimsToUse.z) {
                    // TODO support multiple dimensions
                    throw GSParException("Reduce pattern currently does not support multi-dimensional kernels");
                }
                
                // TODO support batched Reduce pattern

                #ifdef GSPAR_DEBUG
                    std::stringstream ss;
                #endif
                this->compile<TDriverInstance>(dimsToUse);

                // #ifdef GSPAR_DEBUG
                //     auto gpu = this->getGpu<TDriverInstance, decltype(TDriverInstance::getDeviceType())>();
                //     ss << "[GSPar Reduce "<<this<<"] Working with GPU " << gpu << " - " << gpu->getName() << std::endl;
                //     std::cout << ss.str();
                //     ss.str("");
                // #endif

                auto kernel = this->getCompiledKernel<TDriverInstance>();
                kernel->clearParameters();

                this->callbackBeforeAllocatingMemoryOnGpu(dimsToUse, kernel);

                this->mallocParametersInGpu<TDriverInstance>();

                this->copyParametersFromHostToGpuAsync<TDriverInstance>();

                auto executionFlow = this->getExecutionFlow<TDriverInstance>();

                Driver::Dimensions dimsToRun = dimsToUse;

                // We start reducing the input vector
                PointerParameter *inputVector = static_cast<PointerParameter*>(this->getParameter(this->vectorName));
                if (inputVector == nullptr) {
                    throw GSParException("Could not find input parameter with name '" + this->vectorName + "' in Reduce pattern");
                }
                decltype(TDriverInstance::getMemoryObjectType())* inputMemoryObject = dynamic_cast<decltype(TDriverInstance::getMemoryObjectType())*>(inputVector->getMemoryObject());

                // In the first iteration, partialTotals is the output. After the first iteration, it is the input and output parameters
                PointerParameter *partialTotals = static_cast<PointerParameter*>(this->getParameter(this->partialTotalsParamName));
                if (partialTotals == nullptr) {
                    throw GSParException("Could not find partial totals parameter with name '" + this->partialTotalsParamName + "' in Reduce pattern");
                }

                while (true) {

                    Driver::Dimensions blocksAndThreads = kernel->getNumBlocksAndThreadsFor(dimsToRun);

                    this->setSharedMemoryInKernel<TDriverInstance>(kernel, dimsToRun);

                    // Init this->setParametersInKernel
                    this->setDimsParametersInKernel<TDriverInstance>(kernel, dimsToRun);

                    // Sets Pattern parameters in Kernel object
                    for (auto& paramName : this->paramsOrder) {
                        if (paramName == this->vectorName) { // Input parameter
                            if (inputMemoryObject) {
                                inputMemoryObject->waitAsync(); // Waits for async copy to finish
                            }
                            kernel->setParameter(inputMemoryObject); // We can simply set the memory object
                        } else {
                            auto param = this->getParameter(paramName);
                            this->setParameterInKernel<TDriverInstance>(kernel, param);
                        }

                    }
                    // Finish this->setParametersInKernel

                    this->callbackAfterCopyDataFromHostToGpu();
                    this->callbackBeforeRunInGpu();

                    #ifdef GSPAR_DEBUG
                        ss << "[GSPar Reduce "<<this<<"] Running kernel " << kernel << " for " << dimsToRun.toString() << " in flow " << executionFlow << std::endl;
                        std::cout << ss.str();
                        ss.str("");
                    #endif

                    kernel->runAsync(dimsToRun, executionFlow);

                    kernel->waitAsync();

                    #ifdef GSPAR_DEBUG
                        ss << "[GSPar Reduce "<<this<<"] Finished running kernel " << kernel << " in flow " << executionFlow;
                        ss << ". Reduced to " << blocksAndThreads.x.min << " element(s)" << std::endl;
                        std::cout << ss.str();
                        ss.str("");
                    #endif

                    if (blocksAndThreads.x.min == 1) break;

                    Driver::Dimensions newDims(blocksAndThreads.x.min, 0, 0);
                    dimsToRun = newDims;

                    inputMemoryObject = dynamic_cast<decltype(TDriverInstance::getMemoryObjectType())*>(partialTotals->getMemoryObject());

                    kernel->clearParameters();
                }

                // "Hack" to copy partial totals into output parameter
                PointerParameter *outParam = this->getOutputParameter();
                decltype(TDriverInstance::getMemoryObjectType())* outputMemoryObject = dynamic_cast<decltype(TDriverInstance::getMemoryObjectType())*>(partialTotals->getMemoryObject());
                outputMemoryObject->bindTo(outParam->getPointer(), outParam->size);
                outputMemoryObject->copyOut();
                outParam->direction = GSPAR_PARAM_NONE; // We already copied the parameter out, copyParametersFromGpuToHostAsync should ignore it

                this->callbackAfterRunInGpu();

                this->copyParametersFromGpuToHostAsync<TDriverInstance>();

                this->callbackAfterCopyDataFromGpuToHost(dimsToUse, kernel);
            }
        };

    }
}

#endif
