#include <iostream>
#include <chrono>
#include <iomanip>
#ifdef DEBUG
#include "marX2/marX2.h"
#endif

std::chrono::steady_clock::time_point tInitialization;
std::chrono::steady_clock::time_point tComputation;
std::chrono::steady_clock::time_point tFinishing;
std::chrono::steady_clock::time_point tEnd;

void mandelbrot(const double init_a, const double init_b, const double range, const unsigned long dim, const unsigned long niter, unsigned char *M) {
    double step = range/((double) dim);

    tComputation = std::chrono::steady_clock::now(); // Ends initialization, start computation


    for(unsigned long i = 0; i < dim; i++) {
        double im=init_b+(step*i);
        for (unsigned long j = 0; j < dim; j++) {
            double cr;
            double a=cr=init_a+step*j;
            double b=im;
            unsigned long k = 0;
            for (k = 0; k < niter; k++) {
                double a2=a*a;
                double b2=b*b;
                if ((a2+b2)>4.0) break;
                b=2*a*b+im;
                a=a2-b2+cr;
            }
            M[i*dim+j]= (unsigned char)(255-((k*255/niter)));
        }
    }


    tFinishing = std::chrono::steady_clock::now(); // Ends computation, start finishing
}

int main(int argc, char **argv) {
    double init_a=-2.125,init_b=-1.5,range=3.0;
    unsigned long dim = 1000;
    unsigned long niter = 1000;
    std::cout << std::fixed << std::setprecision(0);

    #ifndef DEBUG
        if (argc<3) {
            std::cerr << "Usage: " << argv[0] << " <size> <niterations>" << std::endl;
            exit(-1);
        }
    #endif
    if (argc > 1) {
        dim = strtoul(argv[1], 0, 10);
    }
    if (argc > 2) {
        niter = strtoul(argv[2], 0, 10);
    }

    unsigned char *M = new unsigned char[dim*dim];

    #ifdef DEBUG
        SetupXWindows(dim,dim,1,NULL,"Mandelbroot");
    #endif

    tInitialization = std::chrono::steady_clock::now(); // Begins initialization

    mandelbrot(init_a, init_b, range, dim, niter, M);

    tEnd = std::chrono::steady_clock::now(); // Ends finish

    #ifdef DEBUG
        for(unsigned long i=0; i<dim; i++) {
            ShowLine(&M[i*dim],dim,i);
        }
    #endif

    double msTotal = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tInitialization).count();
    double msInitialization = std::chrono::duration_cast<std::chrono::milliseconds>(tComputation - tInitialization).count();
    double msComputation = std::chrono::duration_cast<std::chrono::milliseconds>(tFinishing - tComputation).count();
    double msFinishing = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tFinishing).count();

    #ifdef DEBUG
        std::cout << "Teste: " << argv[0] << " " << dim << " " << niter << std::endl;
        std::cout << "Total: " << msTotal << " ms" << std::endl;
        std::cout << "Initialization: " << msInitialization << " ms" << std::endl;
        std::cout << "Computation: " << msComputation << " ms" << std::endl;
        std::cout << "Finishing: " << msFinishing << " ms" << std::endl;
    #else
        std::cout << argv[0] << " " << dim << " " << niter << ";" << msTotal << ";" << msInitialization << ";" << msComputation << ";" << msFinishing << std::endl;
    #endif

    #ifdef DEBUG
        getchar();
        CloseXWindows();
    #endif

    delete[] M;
    return 0;
}
