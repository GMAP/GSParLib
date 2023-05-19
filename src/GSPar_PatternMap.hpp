
#ifndef __GSPAR_PATTERNMAP_INCLUDED__
#define __GSPAR_PATTERNMAP_INCLUDED__

#include "GSPar_BaseParallelPattern.hpp"

namespace GSPar {
    namespace Pattern {

        /**
         * Map parallel pattern
         */
        class Map : public BaseParallelPattern {
        public:
            Map() : BaseParallelPattern() { };
            Map(std::string source) : BaseParallelPattern(source) { };

            template<class TDriverInstance>
            Map* clone() const {
                Map* other = new Map();
                this->cloneInto<TDriverInstance>(other);
                return other;
            }
        };

    }
}

#endif
