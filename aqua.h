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
    // Tensor internal management
    size_t  remaining_extra; // per-dim extra space
    size_t  capacity;        // current max-size for space and stride (without extra)
    size_t  order;           // number of dimensions
    size_t  size;            // total elements that the tensor can hold
    // External management
    size_t* shape;           // current shape of the tensor
    size_t* stride;          // strides for each dimension
    void*  data;             // pointer to the tensor data
} Tensor;

typedef union {
    int i;
    float f;
    double d;
} ScalarType;

typedef double (*tensor_op_double)(double, double);
typedef float (*tensor_op_float)(float, float);
typedef int (*tensor_op_int)(int, int);

Tensor* tensor_create(const size_t* shape, size_t order, size_t extra, DataType dtype);
Tensor* tensor_apply_binary_op_double(Tensor* t1, Tensor* t2, tensor_op_double op);
Tensor* tensor_apply_binary_op_float(Tensor* t1, Tensor* t2, tensor_op_float op);
Tensor* tensor_apply_binary_op_int(Tensor* t1, Tensor* t2, tensor_op_int op);

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
Tensor* scalar_tensor(ScalarType v, DataType dtype, size_t extra);

int tensor_reshape(Tensor* t, size_t* shape, size_t new_order);
int tensor_squeeze_at(Tensor* t, size_t idx);
int tensor_unsqueeze(Tensor* t, size_t idx);
int tensor_squeeze(Tensor* t);
int tensor_flatten(Tensor* t);

size_t* broadcast_shape(const size_t* s1, size_t n1, const size_t* s2, size_t n2, size_t* out_order);

void calculate_double(double* result, size_t* out_shape, size_t pos, size_t* s1, size_t* s2, size_t offset1, size_t offset2, size_t* rpos, size_t out_dim, Tensor* t1, Tensor* t2, tensor_op_double op);
void calculate_float(float* result, size_t* out_shape, size_t pos, size_t* s1, size_t* s2, size_t offset1, size_t offset2, size_t* rpos, size_t out_dim, Tensor* t1, Tensor* t2, tensor_op_float op); 
void calculate_int(int* result, size_t* out_shape, size_t pos, size_t* s1, size_t* s2, size_t offset1, size_t offset2, size_t* rpos, size_t out_dim, Tensor* t1, Tensor* t2, tensor_op_int op);
void broadcasted_stride(size_t** s1, size_t** s2, size_t out_dim, Tensor* t1, Tensor* t2);
void print_helper(Tensor* t, int idx);
void tensor_print_stride(Tensor* t);
void tensor_print_shape(Tensor* t);
void tensor_print(Tensor* t);
void tensor_free(Tensor* t);


#endif
