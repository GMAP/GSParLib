#include <iostream>
#include <chrono>
#include <iomanip>

std::chrono::steady_clock::time_point tInitialization;
std::chrono::steady_clock::time_point tComputation;
std::chrono::steady_clock::time_point tFinishing;
std::chrono::steady_clock::time_point tEnd;


unsigned int reduce_vector(const size_t vector_size, const unsigned int* vector) {
    tInitialization = std::chrono::steady_clock::now(); // Begins initialization

    tComputation = std::chrono::steady_clock::now(); // Ends initialization, start computation

    unsigned int total = 0;
    for (size_t i = 0; i < vector_size; i++) {
        total += vector[i];
    }

    tFinishing = std::chrono::steady_clock::now(); // Ends computation, start finishing

    tEnd = std::chrono::steady_clock::now(); // Ends finish

    return total;
}

void print_vector(size_t size, const unsigned int* vector, bool compact = false) {
    if (compact || size > 100) {
        std::cout << vector[0] << "..." << vector[size-1];
    } else {
        for (size_t i = 0; i < size; i++) {
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
    std::cout << std::fixed << std::setprecision(0);

    const size_t VECTOR_SIZE = std::stoi(argv[1]);

    // Create memory objects
    unsigned int *vector = new unsigned int[VECTOR_SIZE];
    for (size_t i = 0; i < VECTOR_SIZE; i++) {
        vector[i] = 1;
    }

#ifdef DEBUG
    std::cout << "Reducing vector:" << std::endl;
    print_vector(VECTOR_SIZE, vector);
#endif

    unsigned int total = reduce_vector(VECTOR_SIZE, vector);

    delete[] vector;

    double msTotal = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tInitialization).count();
    double msInitialization = std::chrono::duration_cast<std::chrono::milliseconds>(tComputation - tInitialization).count();
    double msComputation = std::chrono::duration_cast<std::chrono::milliseconds>(tFinishing - tComputation).count();
    double msFinishing = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tFinishing).count();

#ifdef DEBUG
    std::cout << "Result: " << total << std::endl;
    std::cout << "Total: " << msTotal << " ms" << std::endl;
    std::cout << "Initialization: " << msInitialization << " ms" << std::endl;
    std::cout << "Computation: " << msComputation << " ms" << std::endl;
    std::cout << "Finishing: " << msFinishing << " ms" << std::endl;
#else
    std::cout << total << ";" << msTotal << ";" << msInitialization << ";" << msComputation << ";" << msFinishing << std::endl;
#endif

    return 0;
}
