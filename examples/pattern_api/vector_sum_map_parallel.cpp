#include <iostream>
#include <chrono>
#include <thread>
#include <sstream>

#ifdef GSPARDRIVER_OPENCL
    #include "GSPar_OpenCL.hpp"
    using namespace GSPar::Driver::OpenCL;
#else
    #include "GSPar_CUDA.hpp"
    using namespace GSPar::Driver::CUDA;
#endif

#include "GSPar_PatternMap.hpp"
using namespace GSPar::Pattern;

struct Task {
    float* a;
    float* b;
    float* result;
    float total;
};

void vector_sum(const unsigned int from, const unsigned int to, const unsigned int max, Task* tasks, Map* pattern) {
    // Sequential version, for debugging purposes
    // for (unsigned int t = from; t < to; t++) {
    //     tasks[t].total = 0;
    //     for (unsigned int x = 0; x < max; x++) {
    //         tasks[t].result[x] = tasks[t].a[x] + tasks[t].b[x];
    //         tasks[t].total += tasks[t].result[x];
    //     }
    // }
    // return;

    std::stringstream ss;

#ifdef GSPAR_DEBUG
    ss << "Pattern " << pattern << " processing tasks " << from+1 << " to " << to << std::endl;
    std::cout << ss.str();
    ss.str("");
#endif
    for (unsigned int t = from; t < to; t++) {
        try {

            // Now we set the real parameter values
            pattern->setParameter("a", sizeof(float) * max, tasks[t].a)
                .setParameter("b", sizeof(float) * max, tasks[t].b)
                .setParameter("result", sizeof(float) * max, tasks[t].result, GSPAR_PARAM_OUT);


            // As we compiled the kernel before, it is not needed to compile it again now.
            // The pattern will automatically skip the compiling phase.
            unsigned long dimensions[3] = {max, 0, 0}; // If the dimensions were to be different from the already compiled kernel, it would be re-compiled.
#ifdef GSPAR_DEBUG
            ss << "Pattern " << pattern << " running task " << (t+1) << std::endl;
            std::cout << ss.str();
            ss.str("");
#endif
            pattern->run<Instance>(dimensions);
            
            // Reduce on CPU
            for (unsigned int x = 0; x < max; x++) {
                tasks[t].total += tasks[t].result[x];
            }

        } catch (GSPar::GSParException &ex) {
            std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
            exit(-1);
        }
    }

    delete pattern;

}

void process_tasks(const unsigned int max, unsigned int tasks_size, Task* tasks, unsigned int workers) {
    // Sequential version, for debugging purposes
    // for (unsigned int t = 0; t < tasks_size; t++) {
    //     tasks[t].total = 0;
    //     for (unsigned int x = 0; x < max; x++) {
    //         tasks[t].result[x] = tasks[t].a[x] + tasks[t].b[x];
    //         tasks[t].total += tasks[t].result[x];
    //     }
    // }
    // return;

    // We assume that tasks_size is divisible by workers
    const unsigned int work_for_each = tasks_size/workers;
    std::cout << "Starting " << workers << " workers to process " << tasks_size << " tasks, " << work_for_each << " tasks for each worker" << std::endl;

    auto pattern = new Map(GSPAR_STRINGIZE_SOURCE(
        result[x] = a[x] + b[x];
    ));

    try {

        // Fixed value parameters can be set. Parameter placeholder are for compiling the kernel.
        pattern->setParameterPlaceholder<float*>("a")
            .setParameterPlaceholder<float*>("b")
            .setParameterPlaceholder<float*>("result", GSPAR_PARAM_POINTER, GSPAR_PARAM_OUT);
        
        // Compile the kernel once before cloning the pattern. The compiled Kernel would be copied over to all pattern's clones
        unsigned long dimensions[3] = {max, 0, 0};
        pattern->compile<Instance>(dimensions);

    } catch (GSPar::GSParException &ex) {
        std::cerr << "Exception: " << ex.what() << " - " << ex.getDetails() << std::endl;
        exit(-1);
    }

    // std::cout << "Pattern have " << pattern->getParameterList().size() << " parameters" << std::endl;

    std::thread* threads = new std::thread[workers];
    for (unsigned int w = 0; w < workers; w++) {
        unsigned int from = w*work_for_each;
        unsigned int to = from+work_for_each;
        // Pattern must be cloned for each thread. The compiled kernel is thread-safe and therefore is carried over.
        auto patternCopy = pattern->clone<Instance>();
        threads[w] = std::thread(vector_sum, from, to, max, tasks, patternCopy);
    }

    for (unsigned int w = 0; w < workers; w++) {
        threads[w].join();
    }
}

void print_vector(unsigned int size, const float* vector, float total = 0, bool compact = false) {
    if (compact || size > 100) {
        std::cout << vector[0] << "..." << vector[size-1];
        if (total) std::cout << " = " << total;
    } else {
        for (unsigned int i = 0; i < size; i++) {
            std::cout << vector[i] << " ";
        }
        if (total) std::cout << "= " << total;
    }
    std::cout << std::endl;
}

int main(int argc, const char * argv[]) {
    if (argc < 4) {
        std::cerr << "Use: " << argv[0] << " <vector_size> <workers> <tasks>" << std::endl;
        exit(-1);
    }

    const unsigned int VECTOR_SIZE = std::stoi(argv[1]);
    const unsigned int WORKERS = std::stoi(argv[2]);
    const unsigned int NUM_TASKS = std::stoi(argv[3]);

    if (NUM_TASKS % WORKERS != 0) {
        std::cerr << "Number of tasks (" << NUM_TASKS << ") must be divisible by number of workers (" << WORKERS << ")!" << std::endl;
        exit(-1);
    }
    std::cout << "Summing vectors:" << std::endl;

    Task* tasks = new Task[NUM_TASKS];
    // Create memory objects
    for (unsigned int t = 0; t < NUM_TASKS; t++) {
        tasks[t].result = new float[VECTOR_SIZE];
        tasks[t].a = new float[VECTOR_SIZE];
        tasks[t].b = new float[VECTOR_SIZE];
        for (unsigned int i = 0; i < VECTOR_SIZE; i++) {
            tasks[t].a[i] = (float)(i+t);
            tasks[t].b[i] = (float)((i+t) * 2);
            tasks[t].result[i] = 0;
        }

        std::cout << "Task " << (t+1) << " vector A: ";
        print_vector(VECTOR_SIZE, tasks[t].a);
        std::cout << "Task " << (t+1) << " vector B: ";
        print_vector(VECTOR_SIZE, tasks[t].b);
    }

    auto t_start = std::chrono::steady_clock::now();

    process_tasks(VECTOR_SIZE, NUM_TASKS, tasks, WORKERS);

    auto t_end = std::chrono::steady_clock::now();

    // Output the result buffer
    std::cout << "Results: " << std::endl;
    for (unsigned int t = 0; t < NUM_TASKS; t++) {
        std::cout << "Task " << (t+1) << ": ";
        print_vector(VECTOR_SIZE, tasks[t].result, tasks[t].total);

        delete[] tasks[t].result;
        delete[] tasks[t].a;
        delete[] tasks[t].b;
    }
    delete tasks;

    std::cout << "Test finished succesfully in " << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() << " ms " << std::endl;

    return 0;
}
