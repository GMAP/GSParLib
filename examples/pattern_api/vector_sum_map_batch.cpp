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

void vector_sum(const unsigned int num_vectors, const unsigned int batch_size, const unsigned int vector_size, unsigned int **as, unsigned int **bs, unsigned int **results) {
    try {

        auto pattern = new Map(GSPAR_STRINGIZE_SOURCE(
            result[x] = a[x] + b[x];
        ));

        pattern->setParameter("size", vector_size)
            .setParameterPlaceholder<unsigned int *>("a", GSPAR_PARAM_POINTER, GSPAR_PARAM_IN, true)
            .setParameterPlaceholder<unsigned int *>("b", GSPAR_PARAM_POINTER, GSPAR_PARAM_IN, true)
            .setParameterPlaceholder<unsigned int *>("result", GSPAR_PARAM_POINTER, GSPAR_PARAM_OUT, true);

        pattern->setBatchSize(batch_size);

        pattern->compile<Instance>({vector_size, 0});

        // If num_vectors is not divisible by batch_size, the lib issues a segfault.
        // unsigned int batches = ceil((double)num_vectors/batch_size);
        unsigned int batches = num_vectors/batch_size;
        for (unsigned int b = 0; b < batches; b++) {
            pattern->setBatchedParameter("a", sizeof(unsigned int) * vector_size, &as[b*batch_size])
                .setBatchedParameter("b", sizeof(unsigned int) * vector_size, &bs[b*batch_size])
                .setBatchedParameter("result", sizeof(unsigned int) * vector_size, &results[b*batch_size], GSPAR_PARAM_OUT);

            pattern->run<Instance>();
        }

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
    if (argc < 4) {
        std::cerr << "Use: " << argv[0] << " <vector_size> <vectors> <batch_size>" << std::endl;
        exit(-1);
    }

    const unsigned int VECTOR_SIZE = std::stoi(argv[1]);
    const unsigned int NUM_VECTORS = std::stoi(argv[2]);
    const unsigned int BATCH_SIZE = std::stoi(argv[3]);

    // Create memory objects
    unsigned int** results = new unsigned int*[NUM_VECTORS];
    unsigned int** as = new unsigned int*[NUM_VECTORS];
    unsigned int** bs = new unsigned int*[NUM_VECTORS];
    for (unsigned int v = 0; v < NUM_VECTORS; v++) {
        results[v] = new unsigned int[VECTOR_SIZE];
        as[v] = new unsigned int[VECTOR_SIZE];
        bs[v] = new unsigned int[VECTOR_SIZE];
        for (unsigned int i = 0; i < VECTOR_SIZE; i++) {
            as[v][i] = i + v;
            bs[v][i] = i + v + 1;
            results[v][i] = 0;
        }
    }

    std::cout << "Summing " << NUM_VECTORS << " vectors:" << std::endl;
    for (unsigned int v = 0; v < NUM_VECTORS; v++) {
        std::cout << "Vector A" << v+1 << ": ";
        print_vector(VECTOR_SIZE, as[v]);
        std::cout << "Vector B" << v+1 << ": ";
        print_vector(VECTOR_SIZE, bs[v]);
    }

    auto t_start = std::chrono::steady_clock::now();

    vector_sum(NUM_VECTORS, BATCH_SIZE, VECTOR_SIZE, as, bs, results);

    auto t_end = std::chrono::steady_clock::now();

    // Output the result buffer
    std::cout << "Results:" << std::endl;
    for (unsigned int v = 0; v < NUM_VECTORS; v++) {
        std::cout << "Vector " << v+1 << ": ";
        print_vector(VECTOR_SIZE, results[v]);
    }

    for (unsigned int v = 0; v < NUM_VECTORS; v++) {
        delete results[v];
        delete as[v];
        delete bs[v];
    }
    delete results;
    delete as;
    delete bs;

    std::cout << "Test finished succesfully in " << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() << " ms " << std::endl;

    return 0;
}
