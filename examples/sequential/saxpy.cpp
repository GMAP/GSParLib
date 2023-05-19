#include <iostream>
#include <chrono>
#include <iomanip>

std::chrono::steady_clock::time_point tInitialization;
std::chrono::steady_clock::time_point tComputation;
std::chrono::steady_clock::time_point tFinishing;
std::chrono::steady_clock::time_point tEnd;


unsigned long saxpy(const unsigned long vector_size, const unsigned long scal, const unsigned long* a, const unsigned long* b, unsigned long* result) {
    tInitialization = std::chrono::steady_clock::now(); // Begins initialization

    tComputation = std::chrono::steady_clock::now(); // Ends initialization, start computation

    unsigned long total = 0;
    for (unsigned long i = 0; i < vector_size; i++) {
        result[i] = scal*a[i] + b[i];
        total += result[i];
    }

    tFinishing = std::chrono::steady_clock::now(); // Ends computation, start finishing

    tEnd = std::chrono::steady_clock::now(); // Ends finish

    return total;
}

void print_vector(unsigned long size, const unsigned long* vector, bool compact = false) {
    if (compact || size > 100) {
        std::cout << vector[0] << "..." << vector[size-1];
    } else {
        for (unsigned long i = 0; i < size; i++) {
            std::cout << vector[i] << " ";
        }
    }
    std::cout << std::endl;
}

int main(int argc, const char * argv[]) {
    if (argc < 3) {
        std::cerr << "Use: " << argv[0] << " <vector_size> <scalar>" << std::endl;
        exit(-1);
    }
    std::cout << std::fixed << std::setprecision(0);

    const unsigned long VECTOR_SIZE = std::stoul(argv[1]);
    const unsigned long SCALAR = std::stoul(argv[2]);

    // Create memory objects
    unsigned long *result = new unsigned long[VECTOR_SIZE];
    unsigned long *a = new unsigned long[VECTOR_SIZE];
    unsigned long *b = new unsigned long[VECTOR_SIZE];
    for (unsigned long i = 0; i < VECTOR_SIZE; i++) {
        a[i] = (unsigned long)i;
        b[i] = (unsigned long)i + 1;
        result[i] = 0;
    }

#ifdef DEBUG
    std::cout << "Summing vectors:" << std::endl;
    std::cout << "Vector A: ";
    print_vector(VECTOR_SIZE, a);
    std::cout << "Vector B: ";
    print_vector(VECTOR_SIZE, b);
#endif

    unsigned long total = saxpy(VECTOR_SIZE, SCALAR, a, b, result);

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
