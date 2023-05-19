#ifdef GSPARDRIVER_CUDA
    #include "GSPar_CUDA.hpp"
    using namespace GSPar::Driver::CUDA;
#else
    #include "GSPar_OpenCL.hpp"
    using namespace GSPar::Driver::OpenCL;
#endif
#include "GSPar_PatternReduce.hpp"
using namespace GSPar::Pattern;

int reduce_sum(const int size, const int *vector) {
    int total;
    try {
        auto pattern = new Reduce("in_vector", "+", "total");
        pattern->setParameter("in_vector", sizeof(int) * size, vector)
                .setParameter("total", sizeof(int), &total, GSPAR_PARAM_OUT);
        pattern->run<Instance>({(unsigned int)size, 0});
        delete pattern;
    } catch (GSPar::GSParException &ex) {
        std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
        exit(-1);
    }
    return total;
}

void print_vector(int size, const int* vector, bool compact = false) {
    if (compact || size > 100) {
        std::cout << vector[0] << "..." << vector[size-1];
    } else {
        for (int i = 0; i < size; i++) {
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

    const int VECTOR_SIZE = std::stoul(argv[1]);
    int *vector = new int[VECTOR_SIZE];
    for (int i = 0; i < VECTOR_SIZE; i++) {
        vector[i] = i;
    }

    std::cout << "Summing vector: ";
    print_vector(VECTOR_SIZE, vector);

    int total = reduce_sum(VECTOR_SIZE, vector);

    std::cout << "Summed vector of " << VECTOR_SIZE << " elements: " << total << std::endl;
}