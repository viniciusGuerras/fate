#ifndef AQUA_H
#define AQUA_H

#include <stdlib.h>
#include <stdio.h>
    
typedef enum {
    DT_INT,
    DT_FLOAT,
    DT_DOUBLE
} DataType;

typedef struct {
    DataType dtype;
    void*  data;
    size_t  size;
    size_t  order;
    size_t* shape;
    size_t* stride;
} Tensor;

typedef union {
    int i;
    float f;
    double d;
} ScalarType;

typedef double (*tensor_op_double)(double, double);
typedef float (*tensor_op_float)(float, float);
typedef int (*tensor_op_int)(int, int);

size_t* broadcast_shape(const size_t* s1, size_t n1, const size_t* s2, size_t n2, size_t* out_order);

Tensor* tensor_apply_binary_op_double(Tensor* t1, Tensor* t2, tensor_op_double op);
Tensor* tensor_apply_binary_op_float(Tensor* t1, Tensor* t2, tensor_op_float op);
Tensor* tensor_apply_binary_op_int(Tensor* t1, Tensor* t2, tensor_op_int op);
Tensor* tensor_create(const size_t* shape, size_t order, DataType dtype);

Tensor* tensor_subtract_double(Tensor* t1, Tensor* t2);
Tensor* tensor_multiply_double(Tensor* t1, Tensor* t2);
Tensor* tensor_divide_double(Tensor* t1, Tensor* t2);
Tensor* tensor_sum_double(Tensor* t1, Tensor* t2);

Tensor* tensor_subtract_float(Tensor* t1, Tensor* t2);
Tensor* tensor_multiply_float(Tensor* t1, Tensor* t2);
Tensor* tensor_divide_float(Tensor* t1, Tensor* t2);
Tensor* tensor_sum_float(Tensor* t1, Tensor* t2);

Tensor* tensor_subtract_int(Tensor* t1, Tensor* t2);
Tensor* tensor_multiply_int(Tensor* t1, Tensor* t2);
Tensor* tensor_divide_int(Tensor* t1, Tensor* t2);
Tensor* tensor_sum_int(Tensor* t1, Tensor* t2);

Tensor* tensor_clone(Tensor* c);
Tensor* scalar_tensor(ScalarType v, DataType dtype);

int tensor_reshape(Tensor* t, size_t* shape, size_t new_order);
int tensor_squeeze_at(Tensor* t, size_t idx);
int tensor_unsqueeze(Tensor* t, size_t idx);
int tensor_squeeze(Tensor* t);
int tensor_flatten(Tensor* t);

void print_helper(Tensor* t, int idx);
void tensor_print_stride(Tensor* t);
void tensor_print_shape(Tensor* t);
void tensor_print(Tensor* t);
void tensor_free(Tensor* t);

#endif
