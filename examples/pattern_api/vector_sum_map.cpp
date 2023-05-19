#include <iostream>
#include <chrono>

#ifdef GSPARDRIVER_OPENCL
    #include "GSPar_OpenCL.hpp"
    using namespace GSPar::Driver::OpenCL;
#else
    #include "GSPar_CUDA.hpp"
    using namespace GSPar::Driver::CUDA;
#endif

#include "GSPar_PatternMap.hpp"
using namespace GSPar::Pattern;

void vector_sum(const unsigned int max, const unsigned int* a, const unsigned int* b, unsigned int* result) {
    try {

        auto pattern = new Map(GSPAR_STRINGIZE_SOURCE(
            result[x] = a[x] + b[x];
        ));

        pattern->setParameter("a", sizeof(unsigned int) * max, a)
            .setParameter("b", sizeof(unsigned int) * max, b)
            .setParameter("result", sizeof(unsigned int) * max, result, GSPAR_PARAM_OUT);

        // This set only max values
        unsigned long dims[3] = {max, 0, 0}; // Pass ulong max values directly
        // GSPar::Driver::Dimensions dims(max, 0, 0); // Makes struct passing max values
        // GSPar::Driver::Dimensions dims = {max, 0, 0}; // Makes struct using auto-initialization with ulong max values

        // This way we can set max and min values
        // GSPar::Driver::Dimensions dims = { // Makes struct using auto-initialization with ulong max and min values
        //     {max, 0}, // X: max, min
        //     {0, 0}, // Y: max, min
        //     {0, 0} // Z: max, min
        // };

        // Makes empty struct and them fill values for intended dimensions
        // GSPar::Driver::Dimensions dims;
        // dims.x = GSPar::Driver::SingleDimension(max, 5);

        pattern->run<Instance>(dims);

        // We could also call initialize the Dimensions directly when calling the method:
        // pattern->run<Instance>({max, 0});

        delete pattern;

    } catch (GSPar::GSParException &ex) {
        std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
        exit(-1);
    }
}

void print_vector(unsigned int size, const unsigned int* vector, bool compact = false) {
    if (compact || size > 100) {
        std::cout << vector[0] << "..." << vector[size-1];
    } else {
        for (unsigned int i = 0; i < size; i++) {
            std::cout << vector[i] << " ";
        }
    }
    std::cout << std::endl;
}

int main(int argc, const char * argv[]) {
    if (argc < 2) {
        std::cerr << "Use: " << argv[0] << " <vector_size>" << std::endl;
        exit(-1);
    }

    const unsigned int VECTOR_SIZE = std::stoi(argv[1]);

    // Create memory objects
    unsigned int* result = new unsigned int[VECTOR_SIZE];
    unsigned int* a = new unsigned int[VECTOR_SIZE];
    unsigned int* b = new unsigned int[VECTOR_SIZE];
    for (unsigned int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = i;
        b[i] = i + 1;
        result[i] = 0;
    }

    std::cout << "Summing vectors:" << std::endl;
    std::cout << "Vector A: ";
    print_vector(VECTOR_SIZE, a);
    std::cout << "Vector B: ";
    print_vector(VECTOR_SIZE, b);

    auto t_start = std::chrono::steady_clock::now();

    vector_sum(VECTOR_SIZE, a, b, result);

    auto t_end = std::chrono::steady_clock::now();

    // Output the result buffer
    std::cout << "Result:   ";
    print_vector(VECTOR_SIZE, result);

    delete result;
    delete a;
    delete b;

    std::cout << "Test finished succesfully in " << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() << " ms " << std::endl;

    return 0;
}
