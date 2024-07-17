#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>

std::chrono::steady_clock::time_point tInitialization;
std::chrono::steady_clock::time_point tComputation;
std::chrono::steady_clock::time_point tFinishing;
std::chrono::steady_clock::time_point tEnd;


unsigned long vector_sum(const std::vector<unsigned long> &a,
                        const std::vector<unsigned long> &b,
                        std::vector<unsigned long> &result) {
    tComputation = std::chrono::steady_clock::now(); // Ends initialization, start computation

    unsigned long vector_size = result.size();
    unsigned long total = 0;
    for (unsigned long i = 0; i < vector_size; i++) {
        result.at(i) = a.at(i) + b.at(i);
        total += result.at(i);
    }

    tFinishing = std::chrono::steady_clock::now(); // Ends computation, start finishing

    return total;
}

void print_vector(const std::vector<unsigned long> vector, bool compact = false) {
    if (compact || vector.size() > 100) {
        std::cout << vector.front() << "..." << vector.back();
    } else {
        for (const unsigned long& i : vector) {
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
    std::cout << std::fixed << std::setprecision(0);

    const unsigned long VECTOR_SIZE = std::stoul(argv[1]);

    // Create memory objects
    std::vector<unsigned long> result = std::vector<unsigned long>(VECTOR_SIZE);
    std::vector<unsigned long> a = std::vector<unsigned long>(VECTOR_SIZE);
    std::vector<unsigned long> b = std::vector<unsigned long>(VECTOR_SIZE);
    for (unsigned long i = 0; i < VECTOR_SIZE; i++) {
        a.at(i) = i;
        b.at(i) = i + 1;
        result.at(i) = 0;
    }

#ifdef DEBUG
    std::cout << "Summing vectors:" << std::endl;
    std::cout << "Vector A: ";
    print_vector(a);
    std::cout << "Vector B: ";
    print_vector(b);
#endif

    tInitialization = std::chrono::steady_clock::now(); // Begins initialization

    unsigned long total = vector_sum(a, b, result);

    tEnd = std::chrono::steady_clock::now(); // Ends finish

#ifdef DEBUG
    // Output the result buffer
    std::cout << "Result:   ";
    print_vector(result);
#endif

    result = std::vector<unsigned long>();
    a = std::vector<unsigned long>();
    b = std::vector<unsigned long>();

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
