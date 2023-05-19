#include <iostream>
#include <chrono>
#include <iomanip>

std::chrono::steady_clock::time_point tInitialization;
std::chrono::steady_clock::time_point tComputation;
std::chrono::steady_clock::time_point tFinishing;
std::chrono::steady_clock::time_point tEnd;

#ifdef GSPARDRIVER_OPENCL
    #include "GSPar_OpenCL.hpp"
    using namespace GSPar::Driver::OpenCL;
#else
    #include "GSPar_CUDA.hpp"
    using namespace GSPar::Driver::CUDA;
#endif

#include "GSPar_PatternComposition.hpp"
using namespace GSPar::Pattern;

void print_vector(unsigned long size, const unsigned long *vector, bool compact = false) {
    if (compact || size > 100) {
        std::cout << vector[0] << "..." << vector[size-1];
    } else {
        for (unsigned long i = 0; i < size; i++) {
            std::cout << vector[i] << " ";
        }
    }
    std::cout << std::endl;
}

unsigned long vector_sum(const unsigned long max, const unsigned long *a, const unsigned long *b, unsigned long *result) {
    try {

        auto map = new Map("result[x] = a[x] + b[x];");
        map->setParameter("a", sizeof(unsigned long) * max, a)
            .setParameter("b", sizeof(unsigned long) * max, b)
            .setParameter("result", sizeof(unsigned long) * max, result, GSPAR_PARAM_INOUT);

        unsigned long total = 5;
        // "result" is the vector with the data
        // "+" is the binary associative operator
        // "total" must be an OUT pointer parameter
        auto reduce = new Reduce("result", "+", "total");
        reduce->setParameter("result", sizeof(unsigned long) * max, result, GSPAR_PARAM_INOUT)
               .setParameter("total", sizeof(unsigned long), &total, GSPAR_PARAM_INOUT);

        // Using initializer_list
        // PatternComposition mapReduce {map, reduce};
        // Using variadic templates constructor
        auto mapReduce = new PatternComposition(map, reduce);

        mapReduce->compilePatterns<Instance>({max, 0});


        tComputation = std::chrono::steady_clock::now(); // Ends initialization, start computation


        mapReduce->run<Instance>();


        tFinishing = std::chrono::steady_clock::now(); // Ends computation, start finishing

        delete mapReduce;
        delete reduce;
        delete map;

        return total;

    } catch (GSPar::GSParException &ex) {
        std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
        exit(-1);
    }
}

int main(int argc, const char * argv[]) {
    if (argc < 2) {
        std::cerr << "Use: " << argv[0] << " <vector_size>" << std::endl;
        exit(-1);
    }
    std::cout << std::fixed << std::setprecision(0);

    const unsigned long VECTOR_SIZE = std::stoul(argv[1]);

    // Create memory objects
    unsigned long *result = new unsigned long[VECTOR_SIZE];
    unsigned long *a = new unsigned long[VECTOR_SIZE];
    unsigned long *b = new unsigned long[VECTOR_SIZE];
    for (unsigned long i = 0; i < VECTOR_SIZE; i++) {
        a[i] = i;
        b[i] = i + 1;
        result[i] = 0;
    }

#ifdef DEBUG
    std::cout << "Summing vectors:" << std::endl;
    std::cout << "Vector A: ";
    print_vector(VECTOR_SIZE, a);
    std::cout << "Vector B: ";
    print_vector(VECTOR_SIZE, b);
#endif

    tInitialization = std::chrono::steady_clock::now(); // Begins initialization

    unsigned long total = vector_sum(VECTOR_SIZE, a, b, result);

    tEnd = std::chrono::steady_clock::now(); // Ends finish

#ifdef DEBUG
    // Output the result buffer
    std::cout << "Result:   ";
    print_vector(VECTOR_SIZE, result);
#endif

    delete[] result;
    delete[] a;
    delete[] b;

    double msTotal = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tInitialization).count();
    double msInitialization = std::chrono::duration_cast<std::chrono::milliseconds>(tComputation - tInitialization).count();
    double msComputation = std::chrono::duration_cast<std::chrono::milliseconds>(tFinishing - tComputation).count();
    double msFinishing = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tFinishing).count();

#ifdef DEBUG
    std::cout << "Total:    " << total << std::endl;
    std::cout << "Total: " << msTotal << " ms" << std::endl;
    std::cout << "Initialization: " << msInitialization << " ms" << std::endl;
    std::cout << "Computation: " << msComputation << " ms" << std::endl;
    std::cout << "Finishing: " << msFinishing << " ms" << std::endl;
#else
    std::cout << total << ";" << msTotal << ";" << msInitialization << ";" << msComputation << ";" << msFinishing << std::endl;
#endif

    return 0;
}
