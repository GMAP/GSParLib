#include <iostream>
#include <chrono>
#include <iomanip>

std::chrono::steady_clock::time_point tInitialization;
std::chrono::steady_clock::time_point tComputation;
std::chrono::steady_clock::time_point tFinishing;
std::chrono::steady_clock::time_point tEnd;


void multiply(const unsigned long size, const float *matrixA, const float *matrixB, float *result) {
    tComputation = std::chrono::steady_clock::now(); // Ends initialization, start computation

    for (unsigned long i = 0; i < size; i++) {
        for (unsigned long j = 0; j < size; j++) {
            float sum = 0;
            for(unsigned long k = 0; k < size; k++) {
                sum += matrixA[i * size + k] * matrixB[k * size + j]; // Row-major
            }
            result[i * size + j] = sum;
        }
    }

    tFinishing = std::chrono::steady_clock::now(); // Ends computation, start finishing
}

void printMatrix(const unsigned long size, float *matrix, bool compact = false) {
    if (compact || size > 100) {
        std::cout << matrix[0] << ".." << matrix[size-1] << ".." << matrix[size*size-1];
    } else {
        for (unsigned long i = 0; i < size; i++) {
            std::cout << std::endl;
            for (unsigned long j = 0; j < size; j++) {
                std::cout << matrix[i * size + j] << " "; // Row-major
            }
        }
    }
    std::cout << std::endl;
}

int main(int argc, char const *argv[]) {
    if (argc < 2) {
        std::cerr << "Use: " << argv[0] << " <matrix_size>" << std::endl;
        exit(-1);
    }
    std::cout << std::fixed << std::setprecision(0);
    
    const unsigned long MX = std::stoi(argv[1]);

    float *matrixA = new float[MX * MX];
    float *matrixB = new float[MX * MX];
    float *result = new float[MX * MX];
    for (unsigned long i = 0; i < MX; i++) {
        for (unsigned long j = 0; j < MX; j++) {
            // Row-major
            matrixA[i * MX + j] = i+1;
            matrixB[i * MX + j] = j+1;
            result[i * MX + j] = 0;
        }
    }

    #ifdef DEBUG
        std::cout << "Multiplying matrixes of " << MX << " x " << MX << std::endl;
        std::cout << "Matrix A: ";
        printMatrix(MX, matrixA);
        std::cout << "Matrix B: ";
        printMatrix(MX, matrixB);
    #endif

    tInitialization = std::chrono::steady_clock::now(); // Begins initialization

    multiply(MX, matrixA, matrixB, result);

    tEnd = std::chrono::steady_clock::now(); // Ends finish

    double msTotal = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tInitialization).count();
    double msInitialization = std::chrono::duration_cast<std::chrono::milliseconds>(tComputation - tInitialization).count();
    double msComputation = std::chrono::duration_cast<std::chrono::milliseconds>(tFinishing - tComputation).count();
    double msFinishing = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tFinishing).count();

    #ifdef DEBUG
        std::cout << "Result: ";
        printMatrix(MX, result);
        std::cout << "Total: " << msTotal << " ms" << std::endl;
        std::cout << "Initialization: " << msInitialization << " ms" << std::endl;
        std::cout << "Computation: " << msComputation << " ms" << std::endl;
        std::cout << "Finishing: " << msFinishing << " ms" << std::endl;
    #else
        std::cout << result[0] << ".." << result[MX-1] << ".." << result[MX*MX-1] << ";" << msTotal << ";" << msInitialization << ";" << msComputation << ";" << msFinishing << std::endl;
    #endif

    delete[] matrixA;
    delete[] matrixB;
    delete[] result;
}
