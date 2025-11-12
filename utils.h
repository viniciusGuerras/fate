#ifndef UTILS_H
#define UTILS_H 

#include <stddef.h>

typedef enum { // Supported data types 
    DT_INT,
    DT_FLOAT,
    DT_DOUBLE
} DataType; 

size_t get_dtype_size(DataType dtype);

#endif
