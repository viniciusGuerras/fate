#include "aqua.h"

Tensor* tensor_create(const size_t* shape, size_t ndim) {
	t->size    = 1;
	t->ndim    = ndim;
	Tensor* t = (Tensor*)malloc(size);
	if (!t) {
		printf("Failed to allocate Tensor");
		return NULL;
        }

	t->shape   = malloc(sizeof(size_t) * ndim);
	if (!t->shape) {
		printf("Failed to allocate shape array\n");
		free(t);
		return NULL;
        }

	t->strides = malloc(sizeof(size_t) * ndim);
	if (!t->strides) {
		printf("Failed to allocate strides array\n");
		free(t->shape);
		free(t);
		return NULL;
	}

	for (size_t i = 0; i < ndim; i++) {
		t->shape[i] = (size_t)shape[i];
		t->size *= shape[i];
	}

	t->strides[ndim - 1] = 1;
	for (size_t i = ndim - 1; i > 0; i--) {
		t->strides[i - 1] = t->strides[i] * t->shape[i];
	}

	t->data = malloc(sizeof(float) * t->size);
	if(!t->data){
		printf("Failed to allocate strides array\n");
		free(t->shape);
		free(t->strides);
		free(t);
		return NULL;
	}

	return t;
}

Tensor* scalar_tensor(float s) {
    size_t shape[1] = {1};
    Tensor* t = tensor_create(shape, 1);
    if (t) t->data[0] = s;
    return t;
}

size_t* broadcast_shape(const size_t* s1, size_t n1, const size_t* s2, size_t n2, size_t* out_ndim) {
    *out_ndim = (n1 > n2) ? n1 : n2;
    size_t* out_shape = malloc(sizeof(size_t) * (*out_ndim));
    if (!out_shape) return NULL;

    for (size_t i = 0; i < *out_ndim; i++) {
        size_t dim1 = (i < *out_ndim - n1) ? 1 : s1[i - (*out_ndim - n1)];
        size_t dim2 = (i < *out_ndim - n2) ? 1 : s2[i - (*out_ndim - n2)];
        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            free(out_shape);
            return NULL;
        }
        out_shape[i] = (dim1 > dim2) ? dim1 : dim2;
    }

    return out_shape;
}

float op_add(float a, float b) { return a + b; }
float op_sub(float a, float b) { return a - b; }
float op_mul(float a, float b) { return a * b; }
float op_div(float a, float b) { return a / b; }

Tensor* tensor_apply_binary_op(Tensor* t1, Tensor* t2, tensor_op op) {
	// Calculates and create the result array
	size_t out_dim;
	size_t* out_shape = broadcast_shape(t1->shape, t1->ndim, t2->shape, t2->ndim, &out_dim);
	if (!out_shape) {
		printf("Tensor shapes aren't broadcastable.\n");
		return NULL;
	}

	Tensor* result = tensor_create(out_shape, out_dim);
	if (!result){
		free(out_shape);
		return NULL;
	}

	// Creates and calculates the broadcasting strides
	size_t* s1 = calloc(out_dim, sizeof(size_t)); size_t* s2 = calloc(out_dim, sizeof(size_t));

	for (size_t i = 0; i < out_dim; i++) {
		int t1_idx = (int)i - (int)(out_dim - t1->ndim);
		int t2_idx = (int)i - (int)(out_dim - t2->ndim);
		s1[i] = (t1_idx < 0 || t1->shape[t1_idx] == 1) ? 0 : t1->strides[t1_idx];
		s2[i] = (t2_idx < 0 || t2->shape[t2_idx] == 1) ? 0 : t2->strides[t2_idx];
	}

	size_t* idx = calloc(out_dim, sizeof(size_t));
	size_t size = result->size;

	//still need to find a way to do this faster and more contiguous
	for(size_t i = 0; i < size; i++){
		int offset1 = 0, offset2 = 0;

		for (size_t j = 0; j < out_dim; j++) {
			offset1 += (s1[j] == 0 ? 0 : idx[j] * s1[j]);
			offset2 += (s2[j] == 0 ? 0 : idx[j] * s2[j]);
		}

		result->data[i] = op(t1->data[offset1], t2->data[offset2]);

		for(int j = (out_dim - 1); j >= 0; j--){
			idx[j]++;
			if (idx[j] < out_shape[j]) break;
		        idx[j] = 0;
		}
	}

	free(out_shape);
	free(idx);
	free(s1);
	free(s2);
	return result;
}

Tensor* tensor_sum(Tensor* t1, Tensor* t2)      { return tensor_apply_binary_op(t1, t2, op_add); }
Tensor* tensor_subtract(Tensor* t1, Tensor* t2) { return tensor_apply_binary_op(t1, t2, op_sub); }
Tensor* tensor_multiply(Tensor* t1, Tensor* t2) { return tensor_apply_binary_op(t1, t2, op_mul); }
Tensor* tensor_divide(Tensor* t1, Tensor* t2)   { return tensor_apply_binary_op(t1, t2, op_div); }

int tensor_reshape(Tensor* t, size_t* shape, size_t ndim){
	size_t new_size = 1;
	for(int i = 0; i < ndim; i++){
		new_size *= shape[i];
	}
	if(new_size != t->size){
		printf("Reshaping not allowed.");
		return -1;
	}

	if(new_dim != t->ndim){
		t->shape = realloc(t->shape, new_ndim * sizeof(size_t));
		t->strides = realloc(t->strides, new_ndim * sizeof(size_t));
	}

	t->ndim = new_dim;
	for (size_t i = 0; i < ndim; i++) t->shape[i] = (size_t)shape[i];

	t->strides[ndim - 1] = 1;
	for (size_t i = ndim - 1; i > 0; i--) t->strides[i - 1] = t->strides[i] * t->shape[i];

	return 0;
}

Tensor* tensor_clone(Tensor* c){
	Tensor* t = tensor_create(c->shape, c->ndim);
	t->size = c->size;
	t->ndim = c->ndim;
	for(size_t i = 0; i < c->ndim; i++){
		t->strides[i] = c->strides[i];
		t->shape[i] = c->shape[i];
	}
	for(size_t i = 0; i < c->size; i++){
		t->data[i] = c->data[i];
	}
	return t;
}

void tensor_print(Tensor* t){
	printf("Tensor(shape=[");
	for (size_t i = 0; i < t->ndim; i++) {
		printf("%zu", t->shape[i]);
		if (i < t->ndim - 1) printf(", ");
	}
	printf("], data=[");
	for (size_t i = 0; i < t->size; i++) {
		printf("%f", t->data[i]);
		if (i < t->size - 1) printf(", ");
	}
	printf("])\n");
}

void tensor_free(Tensor* t) {
    if (!t) return;
    free(t);
}

