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

#endif  // ERROR_HPP