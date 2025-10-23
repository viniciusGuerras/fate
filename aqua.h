#ifndef AQUA_H
#define AQUA_H

#include <stdlib.h>
#include <stdio.h>

typedef struct {
    float*  data;
    size_t  size;
    size_t  ndim;
    size_t* shape;
    size_t* strides;
} Tensor;

typedef float (*tensor_op)(float, float);

Tensor* tensor_create(const size_t* shape, size_t ndim);
Tensor* scalar_tensor(float s);
size_t* broadcast_shape(const size_t* s1, size_t n1, const size_t* s2, size_t n2, size_t* out_ndim);
Tensor* tensor_apply_binary_op(Tensor* t1, Tensor* t2, tensor_op op);
Tensor* tensor_sum(Tensor* t1, Tensor* t2);
Tensor* tensor_subtract(Tensor* t1, Tensor* t2);
Tensor* tensor_multiply(Tensor* t1, Tensor* t2);
Tensor* tensor_divide(Tensor* t1, Tensor* t2);
Tensor* tensor_clone(Tensor* c);

void tensor_print(Tensor* t);
void tensor_free(Tensor* t);

#endif
