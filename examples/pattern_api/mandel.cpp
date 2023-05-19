#include <iostream>
#include <chrono>
#ifdef DEBUG
#include "marX2/marX2.h"
#endif

std::chrono::steady_clock::time_point tInitialization;
std::chrono::steady_clock::time_point tComputation;
std::chrono::steady_clock::time_point tFinishing;
std::chrono::steady_clock::time_point tEnd;

#ifdef GSPARDRIVER_CUDA

    #include "GSPar_CUDA.hpp"
    using namespace GSPar::Driver::CUDA;

// #elif GSPARDRIVER_OPENCL
#else // This way my IDE doesn't complain

    #include "GSPar_OpenCL.hpp"
    using namespace GSPar::Driver::OpenCL;

#endif

#include "GSPar_PatternMap.hpp"
using namespace GSPar::Pattern;

void mandelbrot(const double init_a, const double init_b, const double range, const unsigned long dim, const unsigned long niter, unsigned char *M) {
    double step = range/((double) dim);

    auto pattern = new Map(GSPAR_STRINGIZE_SOURCE(
        double im=init_b+(step*i);
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
        M[i*dim+j] = (unsigned char)(255-((k*255/niter)));
    ));

    try {

        pattern->setParameter("init_a", init_a)
            .setParameter("init_b", init_b)
            .setParameter("step", step)
            .setParameter("dim", dim)
            .setParameter("niter", niter)
            .setParameter("M", dim*dim, M, GSPAR_PARAM_OUT);

        pattern->setStdVarNames({"i", "j", ""});

        pattern->compile<Instance>({dim, dim, 0});

    } catch (GSPar::GSParException &ex) {
        std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
        exit(-1);
    }


    tComputation = std::chrono::steady_clock::now(); // Ends initialization, start computation


    try {
        pattern->run<Instance>();
    } catch (GSPar::GSParException &ex) {
        std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
        exit(-1);
    }


    tFinishing = std::chrono::steady_clock::now(); // Ends computation, start finishing


    delete pattern;
}

int main(int argc, char **argv) {
    double init_a=-2.125,init_b=-1.5,range=3.0;
    unsigned long dim = 1000;
    unsigned long niter = 1000;

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
