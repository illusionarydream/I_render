#ifndef ERROR_HPP
#define ERROR_HPP

#include <cassert>
#include <cstdlib>
#include <iostream>

// assert check
#define ASSERT(condition, message)                                             \
    do {                                                                       \
        if (!(condition)) {                                                    \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__   \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (false)

#define CHECK_CUDA_ERROR(call)                                                           \
    {                                                                                    \
        cudaError_t err = call;                                                          \
        if (err != cudaSuccess) {                                                        \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl;                           \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    }

#endif  // ERROR_HPP