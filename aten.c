#include <sys/sysctl.h>
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct {
	float*  data;
	int*   shape;
	int* strides;
	int     ndim;
	int     size;
} Tensor;

typedef double (*tensor_op)(double, double);

Tensor* tensor_create(int* shape, int ndim){
	Tensor* t  = (Tensor*)malloc(sizeof(Tensor));
	t->shape   = malloc(ndim * (sizeof(int)));
	t->strides = malloc(ndim * (sizeof(int)));
	t->ndim    = ndim;
	t->size    = 1;

	for(int i = 0; i < ndim; i++){
		t->shape[i] = shape[i];
		t->strides[i] = 1;
		t->size *= shape[i];
	}

	t->strides[ndim - 1] = 1;
	for(int i = ndim - 2; i >= 0; i--){
	    t->strides[i] = t->strides[i + 1] * t->shape[i + 1];
	}

	t->data = calloc(t->size, sizeof(float));
	return t;
}

int shape_match(int* shapes_t1, int ndims_t1, int* shapes_t2, int ndims_t2){
	if(ndims_t1 != ndims_t2){
		return -1;
	}
	for(int i = 0; i < ndims_t1; i++){
		if(shapes_t1[i] != shapes_t2[i]){
			return -1;
		}
	}
	return 1;
}

int* broadcast_shape(int* s1, int n1, int* s2, int n2, int* out_ndim) {
	int max_dim = n1 > n2 ? n1 : n2;
	int* out_shape = malloc(sizeof(int) * max_dim);

	for(int i = 0; i < max_dim; i++){
		int dim1 = (i < max_dim - n1) ? 1 : s1[i - (max_dim - n1)]; 
		int dim2 = (i < max_dim - n2) ? 1 : s2[i - (max_dim - n2)]; 
		if(dim1 != dim2 && (dim1 != 1 && dim2 != 1)){
			return NULL;
		}

		out_shape[i] = (dim1 > dim2) ? dim1 : dim2;
	}

	*out_ndim = max_dim;
	return out_shape;
}

Tensor* tensor_apply_binary_op(Tensor* t1, Tensor* t2, tensor_op op){
    int out_dim;
    int* out_shape = broadcast_shape(t1->shape, t1->ndim, t2->shape, t2->ndim, &out_dim); 
    if(!out_shape){
        printf("Tensor shapes aren't broadcastable.\n");
        return NULL;
    }

    int t1_broadcasted_strides[out_dim];
    int t2_broadcasted_strides[out_dim];

    for(int i = 0; i < out_dim; i++){
        int t1_idx = i - (out_dim - t1->ndim);
        int t2_idx = i - (out_dim - t2->ndim);
        t1_broadcasted_strides[i] = (t1_idx < 0 || t1->shape[t1_idx] == 1) ? 0 : t1->strides[t1_idx];
        t2_broadcasted_strides[i] = (t2_idx < 0 || t2->shape[t2_idx] == 1) ? 0 : t2->strides[t2_idx];
    }

    Tensor* result = tensor_create(out_shape, out_dim);
    int* idx = calloc(out_dim, sizeof(int));

    for(int i = 0; i < result->size; i++){
        int offset1 = 0, offset2 = 0;

        for(int d = 0; d < out_dim; d++){
            offset1 += idx[d] * t1_broadcasted_strides[d];
            offset2 += idx[d] * t2_broadcasted_strides[d];
        }

        result->data[i] = op(t1->data[offset1], t2->data[offset2]);

        for(int d = out_dim - 1; d >= 0; d--){
            idx[d]++;
            if(idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }
    free(idx);
    return result;
}

double op_add(double t1, double t2) { return t1 + t2; }
double op_sub(double t1, double t2) { return t1 - t2; }
double op_mul(double t1, double t2) { return t1 * t2; }
double op_div(double t1, double t2) { return (t2 > 0 ? (t1 / t2) : 0); }

Tensor* tensor_sum(Tensor* t1, Tensor* t2)      { return tensor_apply_binary_op(t1, t2, op_add); }
Tensor* tensor_subtract(Tensor* t1, Tensor* t2) { return tensor_apply_binary_op(t1, t2, op_sub); }
Tensor* tensor_multiply(Tensor* t1, Tensor* t2) { return tensor_apply_binary_op(t1, t2, op_mul); }
Tensor* tensor_divide(Tensor* t1, Tensor* t2)   { return tensor_apply_binary_op(t1, t2, op_div); }

void tensor_free(Tensor* t){
	free(t->data);
	free(t->shape);
	free(t->strides);
	free(t);
}

