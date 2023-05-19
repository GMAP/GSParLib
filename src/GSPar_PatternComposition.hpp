
#ifndef __GSPAR_PATTERNCOMPOSITION_INCLUDED__
#define __GSPAR_PATTERNCOMPOSITION_INCLUDED__

#include <vector>
#include <map>
#include <initializer_list>
#include <utility>

///// Forward declarations /////

namespace GSPar {
    namespace Pattern {
        class PatternComposition;
    }
}

#include "GSPar_Base.hpp"
#include "GSPar_BaseGPUDriver.hpp"
#include "GSPar_BaseParallelPattern.hpp"
#include "GSPar_PatternMap.hpp"
#include "GSPar_PatternReduce.hpp"

namespace GSPar {
    namespace Pattern {

        enum PatternType {
            GSPAR_PATTERN_MAP,
            GSPAR_PATTERN_REDUCE
        };
        
        class PatternComposition {
        protected:
            bool built = false;
            std::string extraKernelCode;
            std::array<std::string, 3> stdVarNames;
            std::vector<BaseParallelPattern*> patterns;
            std::map<BaseParallelPattern*, PatternType> patternsTypes;
            Driver::Dimensions compiledPatternsDimension;

            template<typename Base, typename T>
            inline bool instanceof(const T*) {
                return std::is_base_of<Base, T>::value;
            }

            template<class TDriverInstance>
            std::string generateKernelSource(Driver::Dimensions max, unsigned int gpuIndex = 0) {

                std::string kernelSource = this->extraKernelCode;
                if (!this->extraKernelCode.empty()) {
                    kernelSource += "\n";
                }
                bool addedKernel = false;
                for(auto pattern : patterns) {
                    if (pattern->getGpuIndex() != gpuIndex) {
                        continue;
                    }
                    addedKernel = true;

                    pattern->callbackBeforeGeneratingKernelSource();
                    kernelSource += pattern->generateKernelSource<TDriverInstance>(max);
                    kernelSource += "\n";
                }

                return addedKernel ? kernelSource : "";
            }

            template<class T>
            PatternComposition& addPatternInverseOrder(T* pattern) {
                this->assertValidParallelPattern(pattern);
                //This has a terrible performance, but this vector shouldn't be that large for this to be a problem
                patterns.insert(patterns.begin(), 1, pattern);
                this->patternsTypes[pattern] = this->instanceof<Pattern::Map>(pattern) ? GSPAR_PATTERN_MAP : GSPAR_PATTERN_REDUCE;
                return *this;
            }

            void assertAnyPatternAdded() {
                if (this->patterns.empty()) {
                    throw GSParException("No patterns added in composition, interrupting");
                }
            }
            template<class T>
            void assertValidParallelPattern(T* pattern) {
                if (!this->instanceof<BaseParallelPattern>(pattern)) {
                    throw GSParException("Trying to add invalid pattern. All patterns must inherit BaseParallelPattern.");
                }
            }

            template<class TDriverInstance>
            void run(Driver::Dimensions pDims, bool useCompiledDim) {
                this->assertAnyPatternAdded();
                Driver::Dimensions dims = useCompiledDim ? this->compiledPatternsDimension : pDims;
                if (!dims.getCount()) {
                    throw GSParException("No dimensions set to run the pattern composition");
                }

                // TODO validade if dims is valid

                this->compilePatterns<TDriverInstance>(dims);

                for (const auto& pattern : this->patterns) {
                    // We pass dims again in Run case we have other thread asking the pattern to compile to another dims (which shouldn't happen anyway)
                    switch (this->patternsTypes[pattern]) {
                        case GSPAR_PATTERN_MAP:
                            (static_cast<Map*>(pattern))->run<TDriverInstance>(dims);
                            break;
                        case GSPAR_PATTERN_REDUCE:
                            // Almost https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
                            (static_cast<Reduce*>(pattern))->run<TDriverInstance>(dims);
                            break;
                    }
                }
            }

        public:
            PatternComposition() = default;

            template<class T>
            PatternComposition(std::initializer_list<T*> patterns) {
                for (auto p : patterns) {
                    this->addPattern(p);
                }
            }

            template<class TFirst, class... TArgs>
            PatternComposition(TFirst pattern, TArgs... args) : PatternComposition(args...) {
                this->addPatternInverseOrder(pattern); // The elements are processed from last to first
            }

            virtual ~PatternComposition() { }


            template<class TDriverInstance>
            PatternComposition* clone() const {
                PatternComposition* other = new PatternComposition();
                for (const auto &pattern : this->patterns) {
                    switch (this->patternsTypes.at(pattern)) {
                        case GSPAR_PATTERN_MAP:
                            other->addPattern((static_cast<Map*>(pattern))->clone<TDriverInstance>());
                            break;
                        case GSPAR_PATTERN_REDUCE:
                            other->addPattern((static_cast<Reduce*>(pattern))->clone<TDriverInstance>());
                            break;
                    }
                }
                other->built = this->built;
                other->extraKernelCode = this->extraKernelCode;
                other->stdVarNames = this->stdVarNames;
                if (this->compiledPatternsDimension.getCount()) {
                    Driver::Dimensions compiledPatternsDimension = this->compiledPatternsDimension;
                    other->compiledPatternsDimension = compiledPatternsDimension;
                }
                return other;
            }
            
            virtual PatternComposition& addExtraKernelCode(std::string extraKernelCode) {
                this->extraKernelCode += extraKernelCode;
                return *this;
            }

            virtual BaseParallelPattern* getPattern(size_t index) {
                return patterns[index];
            }

            template<class T>
            PatternComposition& addPattern(T* pattern) {
                this->assertValidParallelPattern(pattern);
                patterns.push_back(pattern);
                this->patternsTypes[pattern] = this->instanceof<Pattern::Map>(pattern) ? GSPAR_PATTERN_MAP : GSPAR_PATTERN_REDUCE;
                return *this;
            }

            virtual bool isAllPatternsCompiledFor(Driver::Dimensions dims) {
                if (this->compiledPatternsDimension != dims) { // We are compiled with a different dims
                    return false;
                }
                for (auto pattern : this->patterns) {
                    if (!pattern->isKernelCompiledFor(dims)) {
                        return false;
                    }
                }
                return true;
            }

            template<class TDriverInstance>
            PatternComposition& compilePatterns(Driver::Dimensions dims) {
                this->assertAnyPatternAdded();
                if (this->isAllPatternsCompiledFor(dims)) {
                    // The kernels are already compiled
                    return *this;
                }
                
                // Init GPU driver
                TDriverInstance* driver = TDriverInstance::getInstance();
                // Driver::OpenCL::Instance driver = TDriverInstance::getInstance(); //Provides autocomplete
                driver->init();

                if (driver->getGpuCount() == 0) {
                    throw GSParException("No GPU found, interrupting");
                }

                auto gpus = driver->getGpuList();

                unsigned int gpuIndex = 0;
                for (const auto& gpu : gpus) {
                    // Prepare kernels
                    std::string kernelSource = this->generateKernelSource<TDriverInstance>(dims, gpuIndex);
                    if (kernelSource.empty()) {
                        continue; // If there's no patterns in this GPU, we can move on
                    }

                    std::vector<std::string> kernelNames;
                    for (auto pattern : this->patterns) {
                        if (pattern->getGpuIndex() != gpuIndex) {
                            continue;
                        }
                        kernelNames.push_back(pattern->getKernelName());
                    }

                    #ifdef GSPAR_DEBUG
                        std::stringstream ss;
                        ss << "[GSPar "<<this<<"] Compiling " << kernelNames.size() << " kernels in GPU " << gpu << " with " << dims.toString() << ":" << std::endl;
                        ss << kernelSource << std::endl;
                        std::cout << ss.str();
                        ss.str("");
                    #endif

                    if (!kernelNames.empty()) { // If there's no patterns in this GPU, we can move on
                        auto kernels = gpu->prepareKernels(kernelSource.c_str(), kernelNames);
                        int patternIndex = 0;
                        for (auto pattern : this->patterns) {
                            if (pattern->getGpuIndex() != gpuIndex) {
                                continue;
                            }
                            pattern->setCompiledKernel<TDriverInstance>(kernels.at(patternIndex), dims);
                            patternIndex++;
                        }
                    }
                    gpuIndex++;
                }

                this->compiledPatternsDimension = dims;

                return *this;
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
        };

    }
}

#endif
