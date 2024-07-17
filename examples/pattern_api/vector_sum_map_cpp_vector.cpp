#include <iostream>
#include <chrono>
#include <vector>

#ifdef GSPARDRIVER_OPENCL
    #include "GSPar_OpenCL.hpp"
    using namespace GSPar::Driver::OpenCL;
#else
    #include "GSPar_CUDA.hpp"
    using namespace GSPar::Driver::CUDA;
#endif

#include "GSPar_PatternMap.hpp"
using namespace GSPar::Pattern;

void vector_sum(const std::vector<unsigned int> &a,
                const std::vector<unsigned int> &b,
                std::vector<unsigned int> &result) {
    try {

        auto pattern = new Map(GSPAR_STRINGIZE_SOURCE(
            result[x] = a[x] + b[x];
        ));

        pattern->setParameter("a", a)
            .setParameter("b", b)
            .setParameter("result", result, GSPAR_PARAM_OUT);

        unsigned int max = a.size();

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

void print_vector(const std::vector<unsigned int> vector, bool compact = false) {
    if (compact || vector.size() > 100) {
        std::cout << vector.front() << "..." << vector.back();
    } else {
        for (const unsigned int& i : vector) {
            std::cout << i << " ";
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
    std::vector<unsigned int> result = std::vector<unsigned int>(VECTOR_SIZE);
    std::vector<unsigned int> a = std::vector<unsigned int>(VECTOR_SIZE);
    std::vector<unsigned int> b = std::vector<unsigned int>(VECTOR_SIZE);
    for (unsigned int i = 0; i < VECTOR_SIZE; i++) {
        a.at(i) = i;
        b.at(i) = i + 1;
        result.at(i) = 0;
    }

    std::cout << "Summing vectors:" << std::endl;
    std::cout << "Vector A: ";
    print_vector(a);
    std::cout << "Vector B: ";
    print_vector(b);

    auto t_start = std::chrono::steady_clock::now();

    vector_sum(a, b, result);

    auto t_end = std::chrono::steady_clock::now();

    // Output the result buffer
    std::cout << "Result:   ";
    print_vector(result);

    result = std::vector<unsigned int>();
    a = std::vector<unsigned int>();
    b = std::vector<unsigned int>();

    std::cout << "Test finished succesfully in " << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() << " ms " << std::endl;

    return 0;
}
