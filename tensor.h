#ifndef TENSOR_H
#define TENSOR_H

/*
 * tensor.h
 * Implements a simple tensor library for numerical operations.
 * Supports creation, manipulation, and arithmetic on tensors.
 * Author: Vinicius Guerra
 * Start-Date: 2025-10-16
 */

#include <stdlib.h>
#include <string.h>
#include "errors.h"
#include "utils.h"
#include <stdio.h>
#include "arena.h"
#include <math.h>
#include <time.h>
#include "rng.h"

typedef struct {
    /*--- Tensor internal management ---*/
    DataType dtype;          // struct specified above
    size_t   order_max;      // max-size for space and stride
    size_t  remaining_extra; // per-dim extra space
    size_t  capacity;        // current max-size for space and stride (without extra)
    size_t  order;           // number of dimensions
    size_t  size;            // total elements that the tensor can hold
    /*---     External management    ---*/
    size_t* shape;           // current shape of the tensor
    size_t* stride;          // strides for each dimension
    void*  data;             // pointer to the tensor data
} Tensor;

// ScalarType - Union for holding a single scalar of any supported type
typedef union {
    int i;
    float f;
    double d;
} ScalarType;

// Function pointer types for unary operations
int tensor_negation(Tensor* t);
int tensor_abs(Tensor* t);
int tensor_exp(Tensor* t);

// Function pointer types for binary operations
typedef double (*tensor_op_double)(double, double);
typedef float  (*tensor_op_float)(float, float);
typedef int    (*tensor_op_int)(int, int);

// Tensor creation and basic operations
Tensor* tensor_create(const size_t* shape, size_t order, size_t extra, DataType dtype);
Tensor* tensor_clone(Tensor* c);
Tensor* scalar_tensor(ScalarType v, DataType dtype, size_t extra);

// Tensor filling
int tensor_fill_random(Tensor* t);

// Element-wise binary operations with broadcasting
Tensor* tensor_apply_binary_op_double(Tensor* t1, Tensor* t2, tensor_op_double op);
Tensor* tensor_apply_binary_op_float(Tensor* t1, Tensor* t2, tensor_op_float op);
Tensor* tensor_apply_binary_op_int(Tensor* t1, Tensor* t2, tensor_op_int op);

// Linear algebra 
Tensor* tensor_matmul(Tensor* t1, Tensor* t2);

// Convenience wrappers for common operations 
Tensor* tensor_sum_double(Tensor* t1, Tensor* t2);
Tensor* tensor_subtract_double(Tensor* t1, Tensor* t2);
Tensor* tensor_multiply_double(Tensor* t1, Tensor* t2);
Tensor* tensor_divide_double(Tensor* t1, Tensor* t2);

Tensor* tensor_sum_float(Tensor* t1, Tensor* t2);
Tensor* tensor_subtract_float(Tensor* t1, Tensor* t2);
Tensor* tensor_multiply_float(Tensor* t1, Tensor* t2);
Tensor* tensor_divide_float(Tensor* t1, Tensor* t2);

Tensor* tensor_sum_int(Tensor* t1, Tensor* t2);
Tensor* tensor_subtract_int(Tensor* t1, Tensor* t2);
Tensor* tensor_multiply_int(Tensor* t1, Tensor* t2);
Tensor* tensor_divide_int(Tensor* t1, Tensor* t2);

// Tensor manipulation (reshape, transpose, squeeze, flatten)
int tensor_permute(Tensor* t, size_t* permute_arr, size_t permute_arr_size);
int tensor_reshape(Tensor* t, size_t* shape, size_t new_order);
int tensor_transpose(Tensor* t, size_t from, size_t to);
int tensor_squeeze_at(Tensor* t, size_t idx);
int tensor_unsqueeze(Tensor* t, size_t idx);
int tensor_squeeze(Tensor* t);
int tensor_flatten(Tensor* t);

// Debugging / printing utilities
void tensor_print_recursive(void* data, size_t* shape, DataType dtype, size_t order, size_t pos, size_t offset);
void tensor_print_stride(Tensor* t);
void tensor_print_shape(Tensor* t);
void tensor_print(Tensor* t);

#endif // TENSOR_H
