#ifndef TENSOR_H
#define TENSOR_H

/*
 * tensor
 * Implements a simple tensor library for numerical operations.
 * Supports creation, manipulation, and arithmetic on tensors.
 * Author: Vinicius Guerra
 * Start-Date: 2025-10-16
 */

#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "arena.h"
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "rng.h"

typedef struct {
    /*--- Tensor internal management ---*/
    DataType dtype;           // struct specified above
    size_t   order_max;       // max-size for space and stride 
    size_t   order;           // number of dimensions
    size_t   size;            // total elements that the tensor can hold
    /*---     External management    ---*/
    size_t*  shape;           // current shape of the tensor
    size_t*  stride;          // strides for each dimension
    void*    data;            // pointer to the tensor data
} Tensor;

// Tensor-arena related
void tensor_request(RequestState* rs, char* identifier, const size_t* shape, size_t order, size_t extra, DataType dtype);
void* tensor_find(RequestState* rs, char* identifier);
Tensor* tensor_instantiate(RequestState* rs, char* identifier);
void tensor_op_elementwise_request(RequestState* rs, char* identifier_r, char* identifier_1, char* identifier_2);
void tensor_matmul_request(RequestState* rs, char* identifier, const size_t* shape_1, size_t order_1, size_t extra_1, DataType dtype_1, const size_t* shape_2, size_t order_2, size_t extra_2, DataType dtype_2);

// Tensor populating
int tensor_fill_random(Tensor* t);

// Function pointer types for unary operations
int tensor_negation(Tensor* t);
int tensor_abs(Tensor* t);
int tensor_exp(Tensor* t);

// Function pointer types for binary operations
typedef double (*tensor_op_double)(double, double);
typedef float  (*tensor_op_float)(float, float);
typedef int    (*tensor_op_int)(int, int);

Tensor* tensor_matmul(Tensor* t1, Tensor* t2, Tensor* r);
// Tensor creation and basic operations
Tensor* tensor_clone(Tensor* c);

// Element-wise binary operations with broadcasting
void tensor_apply_binary_op_double(Tensor* t1, Tensor* t2, Tensor* r, tensor_op_double op);
void tensor_apply_binary_op_float( Tensor* t1, Tensor* t2, Tensor* r, tensor_op_float op);
void tensor_apply_binary_op_int(   Tensor* t1, Tensor* t2, Tensor* r, tensor_op_int op);

// Convenience wrappers for common operations 
int tensor_sum_double(Tensor* r, Tensor* t1, Tensor* t2);
int tensor_subtract_double(Tensor* r, Tensor* t1, Tensor* t2);
int tensor_multiply_double(Tensor* r, Tensor* t1, Tensor* t2);
int tensor_divide_double(Tensor* r, Tensor* t1, Tensor* t2);

int tensor_sum_float(Tensor* r, Tensor* t1, Tensor* t2);
int tensor_subtract_float(Tensor* r, Tensor* t1, Tensor* t2);
int tensor_multiply_float(Tensor* r, Tensor* t1, Tensor* t2);
int tensor_divide_float(Tensor* r, Tensor* t1, Tensor* t2);

int tensor_sum_int(Tensor* r, Tensor* t1, Tensor* t2);
int tensor_subtract_int(Tensor* r, Tensor* t1, Tensor* t2);
int tensor_multiply_int(Tensor* r, Tensor* t1, Tensor* t2);
int tensor_divide_int(Tensor* r, Tensor* t1, Tensor* t2);

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

// Free tensor memory
void tensor_free(Tensor* t);

#endif 
