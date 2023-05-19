
#ifndef __CUDABASENVRTC_INCLUDED__
#define __CUDABASENVRTC_INCLUDED__

#include <stdio.h>
#include <nvrtc.h>

#define CUDA_ERROR_CHECK

#define NvrtcSafeCall( err ) __nvrtcSafeCall( err, __FILE__, __LINE__ )
#define NvrtcSafeBuild( err, prog ) __nvrtcSafeBuild( prog, err, __FILE__, __LINE__ )

inline void __nvrtcSafeCall( nvrtcResult err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( NVRTC_SUCCESS != err )
    {
        const char* errString = nvrtcGetErrorString(err);

        fprintf( stderr, "nvrtcSafeCall() failed at %s:%i : %s\n",
                 file, line, errString );
        exit( -1 );
    }
#endif

    return;
}

inline void __nvrtcSafeBuild( nvrtcProgram prog, nvrtcResult err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( NVRTC_SUCCESS != err )
    {
        const char* errString = nvrtcGetErrorString(err);

        fprintf( stderr, "nvrtcSafeBuild() failed at %s:%i : %s\n",
                 file, line, errString );

        size_t logSize = 0;
        nvrtcGetProgramLogSize(prog, &logSize);

        if (logSize > 0) {
            char *buildLog = new char[logSize];
            nvrtcGetProgramLog(prog, buildLog);

            fprintf( stderr, "Build log:\n%s", buildLog);
            delete[] buildLog;
        }

        exit( -1 );
    }
#endif

    return;
}

#endif