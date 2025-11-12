#include "utils.h"

/*
 * get_dtype_size - Get the size in bytes of a given data type.
 * @dtype: The data type to query (DT_INT, DT_FLOAT, DT_DOUBLE, etc.)
 *
 * Returns: Size in bytes of the specified data type. Returns 0 if the type is unknown or unsupported.
 */
size_t get_dtype_size(DataType dtype) {
    switch (dtype) {
        case DT_INT:    return sizeof(int);
        case DT_FLOAT:  return sizeof(float);
        case DT_DOUBLE: return sizeof(double);
        default: return 0;
    }
}


