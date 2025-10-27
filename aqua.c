#include "aqua.h"

size_t get_dtype_size(DataType dtype) {
    switch (dtype) {
        case DT_INT:    return sizeof(int);
        case DT_FLOAT:  return sizeof(float);
        case DT_DOUBLE: return sizeof(double);
        default: return 0;
    }
}

Tensor* tensor_create(const size_t* shape, size_t order, DataType dtype) {
	Tensor* t = (Tensor*)malloc(sizeof(Tensor));
	if (!t) {
		printf("Failed to allocate Tensor");
		return NULL;
        }

	t->size  = 1;
	t->order = order;
	t->dtype = dtype;

	size_t type_size = get_dtype_size(dtype);

	t->shape   = malloc(sizeof(size_t) * order);
	if (!t->shape) {
		printf("Failed to allocate shape array\n");
		free(t);
		return NULL;
        }

	t->stride = malloc(sizeof(size_t) * order);
	if (!t->stride) {
		printf("Failed to allocate stride array\n");
		free(t->shape);
		free(t);
		return NULL;
	}

	for (size_t i = 0; i < order; i++) {
		t->shape[i] = (size_t)shape[i];
		t->size *= shape[i];
	}

	t->stride[order - 1] = 1;
	for (size_t i = order - 1; i > 0; i--) {
		t->stride[i - 1] = t->stride[i] * t->shape[i]; }

	t->data = malloc(type_size * t->size);
	if(!t->data){
		printf("Failed to allocate stride array\n");
		free(t->shape);
		free(t->stride);
		free(t);
		return NULL;
	}

	return t;
}

Tensor* scalar_tensor(ScalarType v, DataType dtype) {
	size_t shape[1] = {1};
	Tensor* t = tensor_create(shape, 1, dtype);
	size_t type_size = get_dtype_size(dtype);
	switch (dtype) {
		case DT_INT:    ((int*)t->data)[0] = v.i; break;
		case DT_FLOAT:  ((float*)t->data)[0] = v.f; break;
		case DT_DOUBLE: ((double*)t->data)[0] = v.d; break;
	}
	return t;
}

size_t* broadcast_shape(const size_t* s1, size_t n1, const size_t* s2, size_t n2, size_t* out_order) {
    *out_order = (n1 > n2) ? n1 : n2;
    size_t* out_shape = malloc(sizeof(size_t) * (*out_order));
    if (!out_shape) return NULL;

    for (size_t i = 0; i < *out_order; i++) {
        size_t dim1 = (i < *out_order - n1) ? 1 : s1[i - (*out_order - n1)];
        size_t dim2 = (i < *out_order - n2) ? 1 : s2[i - (*out_order - n2)];
        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            free(out_shape);
            return NULL;
        }
        out_shape[i] = (dim1 > dim2) ? dim1 : dim2;
    }

    return out_shape;
}

double op_add_double(double a, double b) { return a + b; }
double op_sub_double(double a, double b) { return a - b; }
double op_mul_double(double a, double b) { return a * b; }
double op_div_double(double a, double b) { return a / b; }

float op_add_float(float a, float b) { return a + b; }
float op_sub_float(float a, float b) { return a - b; }
float op_mul_float(float a, float b) { return a * b; }
float op_div_float(float a, float b) { return a / b; }

int op_add_int(int a, int b) { return a + b; }
int op_sub_int(int a, int b) { return a - b; }
int op_mul_int(int a, int b) { return a * b; }
int op_div_int(int a, int b) { return a / b; }

Tensor* tensor_apply_binary_op_double(Tensor* t1, Tensor* t2, tensor_op_double op) {
	size_t out_dim;
	size_t* out_shape = broadcast_shape(t1->shape, t1->order, t2->shape, t2->order, &out_dim);
	if (!out_shape) {
		printf("Tensor shapes aren't broadcastable.\n");
		return NULL;
	}
	Tensor* result = tensor_create(out_shape, out_dim, DT_DOUBLE);
	if (!result){
		free(out_shape);
		return NULL;
	}
	size_t* s1 = calloc(out_dim, sizeof(size_t));
	size_t* s2 = calloc(out_dim, sizeof(size_t));
	for (size_t i = 0; i < out_dim; i++) {
		int t1_idx = (int)i - (int)(out_dim - t1->order);
		int t2_idx = (int)i - (int)(out_dim - t2->order);
		s1[i] = (t1_idx < 0 || t1->shape[t1_idx] == 1) ? 0 : t1->stride[t1_idx];
		s2[i] = (t2_idx < 0 || t2->shape[t2_idx] == 1) ? 0 : t2->stride[t2_idx];
	}
	size_t* idx = calloc(out_dim, sizeof(size_t));
	size_t size = result->size;
	for(size_t i = 0; i < size; i++){
		int offset1 = 0, offset2 = 0;

		for (size_t j = 0; j < out_dim; j++) {
			offset1 += (s1[j] == 0 ? 0 : idx[j] * s1[j]);
			offset2 += (s2[j] == 0 ? 0 : idx[j] * s2[j]);
		}
		((double*)result->data)[i] = op(((double*)t1->data)[offset1], ((double*)t2->data)[offset2]);
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

Tensor* tensor_apply_binary_op_float(Tensor* t1, Tensor* t2, tensor_op_float op) {
	size_t out_dim;
	size_t* out_shape = broadcast_shape(t1->shape, t1->order, t2->shape, t2->order, &out_dim);
	if (!out_shape) {
		printf("Tensor shapes aren't broadcastable.\n");
		return NULL;
	}
	Tensor* result = tensor_create(out_shape, out_dim, DT_FLOAT);
	if (!result){
		free(out_shape);
		return NULL;
	}
	size_t* s1 = calloc(out_dim, sizeof(size_t)); size_t* s2 = calloc(out_dim, sizeof(size_t));
	for (size_t i = 0; i < out_dim; i++) {
		int t1_idx = (int)i - (int)(out_dim - t1->order);
		int t2_idx = (int)i - (int)(out_dim - t2->order);
		s1[i] = (t1_idx < 0 || t1->shape[t1_idx] == 1) ? 0 : t1->stride[t1_idx];
		s2[i] = (t2_idx < 0 || t2->shape[t2_idx] == 1) ? 0 : t2->stride[t2_idx];
	}
	size_t* idx = calloc(out_dim, sizeof(size_t));
	size_t size = result->size;
	for(size_t i = 0; i < size; i++){
		int offset1 = 0, offset2 = 0;

		for (size_t j = 0; j < out_dim; j++) {
			offset1 += (s1[j] == 0 ? 0 : idx[j] * s1[j]);
			offset2 += (s2[j] == 0 ? 0 : idx[j] * s2[j]);
		}
		((float*)result->data)[i] = op(((float*)t1->data)[offset1], ((float*)t2->data)[offset2]);
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

Tensor* tensor_apply_binary_op_int(Tensor* t1, Tensor* t2, tensor_op_int op) {
	size_t out_dim;
	size_t* out_shape = broadcast_shape(t1->shape, t1->order, t2->shape, t2->order, &out_dim);
	if (!out_shape) {
		printf("Tensor shapes aren't broadcastable.\n");
		return NULL;
	}
	Tensor* result = tensor_create(out_shape, out_dim, DT_INT);
	if (!result){
		free(out_shape);
		return NULL;
	}
	size_t* s1 = calloc(out_dim, sizeof(size_t)); size_t* s2 = calloc(out_dim, sizeof(size_t));
	for (size_t i = 0; i < out_dim; i++) {
		int t1_idx = (int)i - (int)(out_dim - t1->order);
		int t2_idx = (int)i - (int)(out_dim - t2->order);
		s1[i] = (t1_idx < 0 || t1->shape[t1_idx] == 1) ? 0 : t1->stride[t1_idx];
		s2[i] = (t2_idx < 0 || t2->shape[t2_idx] == 1) ? 0 : t2->stride[t2_idx];
	}
	size_t* idx = calloc(out_dim, sizeof(size_t));
	size_t size = result->size;
	for(size_t i = 0; i < size; i++){
		int offset1 = 0, offset2 = 0;

		for (size_t j = 0; j < out_dim; j++) {
			offset1 += (s1[j] == 0 ? 0 : idx[j] * s1[j]);
			offset2 += (s2[j] == 0 ? 0 : idx[j] * s2[j]);
		}
		((int*)result->data)[i] = op(((int*)t1->data)[offset1], ((int*)t2->data)[offset2]);
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

Tensor* tensor_sum_double(Tensor* t1, Tensor* t2)      { return tensor_apply_binary_op_double(t1, t2, op_add_double); }
Tensor* tensor_subtract_double(Tensor* t1, Tensor* t2) { return tensor_apply_binary_op_double(t1, t2, op_sub_double); }
Tensor* tensor_multiply_double(Tensor* t1, Tensor* t2) { return tensor_apply_binary_op_double(t1, t2, op_mul_double); }
Tensor* tensor_divide_double(Tensor* t1, Tensor* t2)   { return tensor_apply_binary_op_double(t1, t2, op_div_double); }

Tensor* tensor_sum_float(Tensor* t1, Tensor* t2)      { return tensor_apply_binary_op_float(t1, t2, op_add_float); }
Tensor* tensor_subtract_float(Tensor* t1, Tensor* t2) { return tensor_apply_binary_op_float(t1, t2, op_sub_float); }
Tensor* tensor_multiply_float(Tensor* t1, Tensor* t2) { return tensor_apply_binary_op_float(t1, t2, op_mul_float); }
Tensor* tensor_divide_float(Tensor* t1, Tensor* t2)   { return tensor_apply_binary_op_float(t1, t2, op_div_float); }

Tensor* tensor_sum_int(Tensor* t1, Tensor* t2)      { return tensor_apply_binary_op_int(t1, t2, op_add_int); }
Tensor* tensor_subtract_int(Tensor* t1, Tensor* t2) { return tensor_apply_binary_op_int(t1, t2, op_sub_int); }
Tensor* tensor_multiply_int(Tensor* t1, Tensor* t2) { return tensor_apply_binary_op_int(t1, t2, op_mul_int); }
Tensor* tensor_divide_int(Tensor* t1, Tensor* t2)   { return tensor_apply_binary_op_int(t1, t2, op_div_int); }

int tensor_reshape(Tensor* t, size_t* shape, size_t new_order){
	size_t new_size = 1;
	for(int i = 0; i < new_order; i++){
		new_size *= shape[i];
	}
	if(new_size != t->size){
		printf("Reshaping not allowed.");
		return -1;
	}

	if(new_order != t->order){
		size_t t_shape* = realloc(t->shape, new_order * sizeof(size_t));
		size_t t_stride* = realloc(t->stride, new_order * sizeof(size_t));
		if(!t_shape || t_stride){
			free(t_stride);
			free(t_shape);
			return -1;
		}
		t->stride = t_stride;
		t->shape = t_shape;
	}

	t->order = new_order;
	for (size_t i = 0; i < new_order; i++) t->shape[i] = (size_t)shape[i];

	t->stride[new_order - 1] = 1;
	for (size_t i = new_order - 1; i > 0; i--) t->stride[i - 1] = t->stride[i] * t->shape[i];

	return 0;
}

int tensor_squeeze(Tensor* t){
	if (!t || !t->shape || !t->stride) return -1;
	size_t new_order = 0;

	for (size_t i = 0; i < t->order; i++) {
		if (t->shape[i] != 1) {
		    t->shape[new_order++] = t->shape[i];
		}
	}

	t->stride[new_order - 1] = 1;
	for (size_t i = new_order - 1; i > 0; i--){
		t->stride[i - 1] = t->stride[i] * t->shape[i];
	}

	t->order = new_order;
	return 0;
}

int tensor_squeeze_at(Tensor* t, size_t idx){
	if (!t || !t->shape || !t->stride) return -1;
	if(t->shape[idx] != 1){
		printf("Error: Invalid axis for squeeze operation.\n");
		return -1;
	}

	for (size_t i = idx + 1; i < t->order; i++) {
		t->shape[i - 1] = t->shape[i];
		t->stride[i - 1] = t->stride[i];
	}

	t->order--;
	return 0;
}

int tensor_unsqueeze(Tensor* t, size_t idx){
	if(!t || idx > t->order){
		printf("Error: Invalid axis for unsqueeze operation.\n");
		return -1;
	}

	size_t new_order = t->order + 1;

	size_t* new_shape = malloc(sizeof(size_t) * new_order);
	size_t* new_stride = malloc(sizeof(size_t) * new_order);
	if(!new_shape || !new_stride){
		free(new_shape);
		free(new_stride);
		return -1;
	}

	new_shape[idx] = 1;
	new_stride[idx] = (idx < t->order) ? t->stride[idx] : 1;

	size_t diff = 0;
	for(int i = 0; i < new_order - 1; i++){
		if(i == idx ){
			diff++;
		}
		temp_shape[i + diff] =  t->shape[i];
		temp_stride[i + diff] = t->stride[i];

	 }

	free(t->shape);
	free(t->stride);

	t->stride = temp_stride;
	t->shape = temp_shape;
	t->order = new_order;
	return 0;
}

Tensor* tensor_clone(Tensor* c){
	Tensor* t = tensor_create(c->shape, c->order, c->dtype);
	t->size = c->size;
	t->order = c->order;
	for(size_t i = 0; i < c->order; i++){
		t->stride[i] = c->stride[i];
		t->shape[i] = c->shape[i];
	}
	for(size_t i = 0; i < c->size; i++){
		switch (c->dtype) {
			case DT_INT:     
				((int*)t->data)[i] = ((int*)c->data)[i];
				break;
			case DT_FLOAT:  
				((float*)t->data)[i] = ((float*)c->data)[i];
				break;
			case DT_DOUBLE: 
				((double*)t->data)[i] = ((double*)c->data)[i];
				break;
		}
	}
	return t;
}

int tensor_flatten(Tensor* t){
	if(!t){
		return -1;
	}
	free(t->shape);
	free(t->stride);

	t->shape = malloc(sizeof(size_t));
	t->stride = malloc(sizeof(size_t));

	t->shape[0] = t->size;
	t->stride[0] = 1;
	t->order = 1;
	return 0;
}

void print_helper(Tensor* t, int idx){
	int count = 0;
	for(int j = 0; j < t->order - 1; j++){
		if(idx%t->stride[j]==0){
			count++;
		}
	}
	if(idx > 0){
		for(int k = 0; k < count; k++){
			printf("]");
		}
		if(idx < t->size){
			printf(", ");
		}
	}
	if(idx < t->size){
		for(int k = 0; k < count; k++){
			printf("[");
		}
	}
}

void tensor_print(Tensor* t){
	printf("data:([");
	size_t i = 0;
	switch (t->dtype) {
		case DT_INT: {
			int* data = (int*)t->data;
			for (; i < t->size; i++) {
				print_helper(t, i);
				printf("%d", data[i]);
			}
			break;
		}
		case DT_FLOAT: {
			float* data = (float*)t->data;
			for (; i < t->size; i++) {
				print_helper(t, i);
				printf("%.2f", data[i]);
			}
			break;
		}
		case DT_DOUBLE: {
			double* data = (double*)t->data;
			for (; i < t->size; i++) {
				print_helper(t, i);
				printf("%.3lf", data[i]);
			}
			break;
		}
		default:
		    printf("unknown dtype");
		    break;
	}
	//needed for the last closing ] to appear
	print_helper(t, i);
	printf("])\n");
}

void tensor_print_shape(Tensor* t){
	printf("shape:(");
	for (size_t i = 0; i < t->order; i++) {
		printf("%zu", t->shape[i]);
		if (i < t->order - 1) printf(", ");
	}
	printf(")\n");
}

void tensor_print_stride(Tensor* t){
	printf("stride:(");
	for (size_t i = 0; i < t->order; i++) {
		printf("%zu", t->stride[i]);
		if (i < t->order - 1) printf(", ");
	}
	printf(")\n");
}

void tensor_free(Tensor* t) {
	if (!t) return;
	free(t->data);
	free(t->stride);
	free(t->shape);
	free(t);
}

