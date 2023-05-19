
#ifndef __GSPAR_BASE_INCLUDED__
#define __GSPAR_BASE_INCLUDED__

#include <chrono>
#include <string>
#include <algorithm> //std::generate_n

#define GSPAR_STRINGIZE_SOURCE(...) #__VA_ARGS__

namespace GSPar {

    class GSParException : public std::exception {
    protected:
        std::string msg;
        std::string details;

    public:
        GSParException() : std::exception() { }
        explicit GSParException(std::string msg, std::string details = "") {
            this->msg = msg;
            this->details = details;
        }
        virtual std::string what() { return this->msg; }
        virtual std::string getDetails() { return this->details; }
    };

    // Auxiliary functions
    std::string getRandomString(short length);

    template<typename Base, typename T>
    inline bool instanceof(const T*) {
        return std::is_base_of<Base, T>::value;
    }
}

#endif
