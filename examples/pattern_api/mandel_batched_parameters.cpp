/* ***************************************************************************
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License version 2 as
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *  As a special exception, you may use this file as part of a free software
 *  library without restriction.  Specifically, if other files instantiate
 *  templates or use macros or inline functions from this file, or you compile
 *  this file and link it with other files to produce an executable, this
 *  file does not by itself cause the resulting executable to be covered by
 *  the GNU General Public License.  This exception does not however
 *  invalidate any other reasons why the executable file might be covered by
 *  the GNU General Public License.
 *
 ****************************************************************************
 */

/*

   Author: Marco Aldinucci.
   email:  aldinuc@di.unipi.it
   marco@pisa.quadrics.com
   date :  15/11/97

Modified by:

****************************************************************************
 *  Author: Dalvan Griebler <dalvangriebler@gmail.com>
 *  Author: Dinei Rockenbach <dinei.rockenbach@edu.pucrs.br>
 *
 *  Copyright: GNU General Public License
 *  Description: This program simply computes the mandelbroat set.
 *  File Name: mandel.cpp
 *  Version: 1.0 (25/05/2018)
 *  Compilation Command: make
 ****************************************************************************
*/


#include <stdio.h>
#ifdef DEBUG
#include "marX2.h"
#endif
#include <sys/time.h>
#include <math.h>

#include <iostream>
#include <chrono>

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

#define DIM 800
#define ITERATION 1024

double diffmsec(struct timeval  a,  struct timeval  b) {
    long sec  = (a.tv_sec  - b.tv_sec);
    long usec = (a.tv_usec - b.tv_usec);

    if(usec < 0) {
        --sec;
        usec += 1000000;
    }
    return ((double)(sec*1000)+ (double)usec/1000.0);
}



int main(int argc, char **argv) {
    double init_a=-2.125,init_b=-1.5,range=3.0;
    unsigned long dim = DIM, niter = ITERATION;
    // stats
    struct timeval t1,t2;
    int retries=1;
    double avg = 0;
    int batch_size = 1;

    if (argc<5) {
        printf("Usage: %s size niterations retries batch_size\n\n", argv[0]);
        exit(-1);
    }
    else {
        dim = atoi(argv[1]);
        niter = atoi(argv[2]);
        retries = atoi(argv[3]);
        batch_size = atoi(argv[4]);
    }

    double * runs = new double[retries];
    unsigned char **Ms = new unsigned char*[batch_size];
    for (int b = 0; b < batch_size; b++) {
        Ms[b] = new unsigned char[dim];
    }

    unsigned int batches = ceil((double)dim/batch_size);

    double step = range/((double) dim);

#ifdef DEBUG
    SetupXWindows(dim,dim,1,NULL,"Sequential Mandelbroot");
#endif
    
    printf("bin;size;numiter;time (ms);workers;batch size\n");
    for (int r=0; r<retries; r++) {

        auto pattern = new Map(GSPAR_STRINGIZE_SOURCE(
            double im=init_b+(step*i);
            double cr;
            double a=cr=init_a+step*j;
            double b=im;
            int k = 0;
            for (k=0; k<niter; k++) {
                double a2=a*a;
                double b2=b*b;
                if ((a2+b2)>4.0) break;
                b=2*a*b+im;
                a=a2-b2+cr;
            }
            M[j]= (unsigned char) 255-((k*255/niter));
        ));

        unsigned long dimensions[3] = {dim, 0, 0};

        try {

            pattern->setParameterPlaceholder<int>("i", GSPAR_PARAM_VALUE, GSPAR_PARAM_IN, true)
                .setParameter("dim", dim)
                .setParameter("init_a", init_a)
                .setParameter("init_b", init_b)
                .setParameter("step", step)
                .setParameter("niter", niter)
                .setParameterPlaceholder<unsigned char*>("M", GSPAR_PARAM_POINTER, GSPAR_PARAM_INOUT, true);

            pattern->setStdVarNames({"j", "", ""})
                .setBatchSize(batch_size);

            pattern->compile<Instance>(dimensions);

        } catch (GSPar::GSParException &ex) {
            std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
            exit(-1);
        }

        // Start time
        gettimeofday(&t1,NULL);

        int* Is = new int[batch_size];

        for(unsigned int b=0; b<batches; b++) {
            // std::cout << "Processing batch " << b << ", lines ";
            for (int i = 0; i < batch_size; i++) {
                Is[i] = b*batch_size + i;
                // std::cout << Is[i] << " ";
            }
            // std::cout << std::endl;

            try {

                pattern->setBatchedParameter("i", Is)
                    .setBatchedParameter("M", dim, Ms, GSPAR_PARAM_INOUT);

                pattern->run<Instance>(dimensions);

            } catch (GSPar::GSParException &ex) {
                std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
                exit(-1);
            }
            
#ifdef DEBUG
            for (int i = 0; i < batch_size; i++) {
                ShowLine(Ms[i],dim,Is[i]);
            }
#endif
        }
        // Stop time
        gettimeofday(&t2,NULL);

        avg += runs[r] = diffmsec(t2,t1);
        printf("%s;%lu;%lu;%.2f;1;1\n", argv[0], dim, niter, runs[r]);
    }
    avg = avg / (double) retries;
    double var = 0;
    for (int r=0; r<retries; r++) {
        var += (runs[r] - avg) * (runs[r] - avg);
    }
    var /= retries;

#ifdef DEBUG
    printf("Average on %d experiments = %f (ms) Std. Dev. %f\n\nPress a key\n",retries,avg,sqrt(var));
    getchar();
    CloseXWindows();
#endif

    delete[] runs;
    for (int b = 0; b < batch_size; b++) {
        delete[] Ms[b];
    }
    delete[] Ms;
    return 0;
}
