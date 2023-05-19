
#ifndef __CUDABASE_INCLUDED__
#define __CUDABASE_INCLUDED__

#include <stdio.h>
#include <cuda.h>

#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
// #define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( CUresult err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( CUDA_SUCCESS != err )
    {
        const char* errName;
        cuGetErrorName(err, &errName);
        const char* errString;
        cuGetErrorString(err, &errString);

        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s - %s\n",
                 file, line, errName, errString );
        exit( -1 );
    }
#endif

    return;
}

// inline void __cudaCheckError( const char *file, const int line )
// {
// #ifdef CUDA_ERROR_CHECK
//     CUresult err = cudaGetLastError();
//     if ( cudaSuccess != err )
//     {
//         fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
//                  file, line, cudaGetErrorString( err ) );
//         exit( -1 );
//     }

//     // More careful checking. However, this will affect performance.
//     // Comment away if needed.
//     err = cudaDeviceSynchronize();
//     if( cudaSuccess != err )
//     {
//         fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
//                  file, line, cudaGetErrorString( err ) );
//         exit( -1 );
//     }
// #endif

//     return;
// }
#endif