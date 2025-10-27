#include "aqua.h"

size_t get_dtype_size(DataType dtype) {
    switch (dtype) {
        case DT_INT:    return sizeof(int);
        case DT_FLOAT:  return sizeof(float);
        case DT_DOUBLE: return sizeof(double);
        default: return 0;
    }
}

Tensor* tensor_create(const size_t* shape, size_t order, size_t extra, DataType dtype) {
	size_t total_elements = 1;
	for(int i = 0; i < order; i++){
		total_elements *= shape[i];
	}

	size_t type_size = get_dtype_size(dtype);

	size_t size =  sizeof(Tensor) 
			+ (((2 * sizeof(size_t)) + extra) * order) 
			+ (type_size * total_elements);

	char* block = calloc(1, size);
	if (!block) {
		printf("ERROR: Failed to allocate Tensor.\n");
		return NULL;
        }

	Tensor* t =  (Tensor*)block;
	t->order = order;
	t->remaining_extra = extra;
	t->dtype = dtype;
	t->size  = total_elements;

	char* ptr = block + sizeof(Tensor);
	t->shape = (size_t*)ptr;

	ptr += sizeof(size_t) * (order + extra);
	t->stride = (size_t*)ptr; 

	ptr += sizeof(size_t) * (order + extra);
	t->data = (void*)ptr; 

	for (size_t i = 0; i < order; i++) {
		t->shape[i] = (size_t)shape[i];
	}

	t->stride[order - 1] = 1;
	for (size_t i = order - 1; i > 0; i--) {
		t->stride[i - 1] = t->stride[i] * t->shape[i]; }

	return t;
}

Tensor* scalar_tensor(ScalarType v, DataType dtype, size_t extra) {
	size_t shape[1] = {1};
	Tensor* t = tensor_create(shape, 1, extra, dtype);
	size_t type_size = get_dtype_size(dtype);
	switch (dtype) {
		case DT_INT:    ((int*)t->data)[0] = v.i; break;
		case DT_FLOAT:  ((float*)t->data)[0] = v.f; break;
		case DT_DOUBLE: ((double*)t->data)[0] = v.d; break;
	}
	return t;
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

inline size_t* broadcast_shape(const size_t* s1, size_t n1, const size_t* s2, size_t n2, size_t* out_order) {
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

inline void broadcasted_stride(size_t** s1, size_t** s2, 
			size_t out_dim, Tensor* t1, Tensor* t2) {
	for (size_t i = 0; i < out_dim; i++) {
		int t1_idx = (int)i - (int)(out_dim - t1->order);
		int t2_idx = (int)i - (int)(out_dim - t2->order);
		(*s1)[i] = (t1_idx < 0 || t1->shape[t1_idx] == 1) ? 0 : t1->stride[t1_idx];
		(*s2)[i] = (t2_idx < 0 || t2->shape[t2_idx] == 1) ? 0 : t2->stride[t2_idx];
	}
}

void calculate_double(double* result, size_t* out_shape, 
               size_t pos, size_t* s1, size_t* s2, 
               size_t offset1, size_t offset2, size_t* rpos,
               size_t out_dim, Tensor* t1, Tensor* t2, tensor_op_double op) {

    if (pos == out_dim) {
	result[*rpos] = op(((double*)t1->data)[offset1], ((double*)t2->data)[offset2]);
        (*rpos)++;
        return;
    }

    for (size_t i = 0; i < out_shape[pos]; i++) {
        size_t new_offset1 = offset1 + i * s1[pos];
        size_t new_offset2 = offset2 + i * s2[pos];
        calculate_double(result, out_shape, pos + 1, s1, s2, new_offset1, new_offset2, rpos, out_dim, t1, t2, op);
    }
}

void calculate_float(float* result, size_t* out_shape, 
               size_t pos, size_t* s1, size_t* s2, 
               size_t offset1, size_t offset2, size_t* rpos,
               size_t out_dim, Tensor* t1, Tensor* t2, tensor_op_float op) {

    if (pos == out_dim) {
        result[*rpos] = op(((float*)t1->data)[offset1], ((float*)t2->data)[offset2]);
        (*rpos)++;
        return;
    }

    for (size_t i = 0; i < out_shape[pos]; i++) {
        size_t new_offset1 = offset1 + i * s1[pos];
        size_t new_offset2 = offset2 + i * s2[pos];
        calculate_float(result, out_shape, pos + 1, s1, s2, new_offset1, new_offset2, rpos, out_dim, t1, t2, op);
    }
}

void calculate_int(int* result, size_t* out_shape, 
               size_t pos, size_t* s1, size_t* s2, 
               size_t offset1, size_t offset2, size_t* rpos,
               size_t out_dim, Tensor* t1, Tensor* t2, tensor_op_int op) {

    if (pos == out_dim) {
        result[*rpos] = op(((int*)t1->data)[offset1], ((int*)t2->data)[offset2]);
        (*rpos)++;
        return;
    }

    for (size_t i = 0; i < out_shape[pos]; i++) {
        size_t new_offset1 = offset1 + i * s1[pos];
        size_t new_offset2 = offset2 + i * s2[pos];
        calculate_int(result, out_shape, pos + 1, s1, s2, new_offset1, new_offset2, rpos, out_dim, t1, t2, op);
    }
}

/*
*
* Calculates "out_shape" the broadcasted shape between both of the tensors
* Creates the result tensor with shape equal to out_shape 
*
*/
Tensor* tensor_apply_binary_op_double(Tensor* t1, Tensor* t2, tensor_op_double op) {
	size_t out_dim;
	size_t extra = t1->remaining_extra > t2->remaining_extra ? 
		   t1->remaining_extra : t2->remaining_extra;
	size_t* out_shape = broadcast_shape(t1->shape, t1->order, t2->shape, t2->order, &out_dim);
	if (!out_shape) {
		printf("ERROR: Tensor shapes aren't broadcastable.\n");
		return NULL;
	}

	Tensor* result = tensor_create(out_shape, out_dim, extra, DT_DOUBLE);
	if (!result){
		printf("ERROR: Fail creating result tensor.\n");
		free(out_shape);
		return NULL;
	}

	size_t* s1 = malloc(out_dim * sizeof(size_t));
	size_t* s2 = malloc(out_dim * sizeof(size_t));
	if(!s1 || !s2){
		printf("ERROR: Fail creating s1 or s2 arrays.\n");
	}
	broadcasted_stride(&s1, &s2, out_dim, t1, t2);

	size_t rpos = 0;
	calculate_double((double*)result->data, out_shape, 0, s1, s2, 0, 0, &rpos, out_dim, t1, t2, op);

	free(out_shape);
	free(s1);
	free(s2);
	return result;
}


Tensor* tensor_apply_binary_op_float(Tensor* t1, Tensor* t2, tensor_op_float op) {
	size_t extra = t1->remaining_extra > t2->remaining_extra ? t1->remaining_extra : t2->remaining_extra;
	size_t out_dim;
	size_t* out_shape = broadcast_shape(t1->shape, t1->order, t2->shape, t2->order, &out_dim);
	if (!out_shape) {
		printf("ERROR: Tensor shapes aren't broadcastable.\n");
		return NULL;
	}
	Tensor* result = tensor_create(out_shape, out_dim, extra, DT_FLOAT);
	if (!result){
		free(out_shape);
		return NULL;
	}

	size_t* s1 = calloc(out_dim, sizeof(size_t)); 
	size_t* s2 = calloc(out_dim, sizeof(size_t));
	if(!s1 || !s2){
		printf("ERROR: Fail creating s1 or s2 arrays.\n");
	}
	broadcasted_stride(&s1, &s2, out_dim, t1, t2);

	size_t rpos = 0;
	calculate_float((float*)result->data, out_shape, 0, s1, s2, 0, 0, &rpos, out_dim, t1, t2, op);

	free(out_shape);
	free(s1);
	free(s2);
	return result;
}

Tensor* tensor_apply_binary_op_int(Tensor* t1, Tensor* t2, tensor_op_int op) {
	size_t extra = t1->remaining_extra > t2->remaining_extra ? t1->remaining_extra : t2->remaining_extra;
	size_t out_dim;
	size_t* out_shape = broadcast_shape(t1->shape, t1->order, t2->shape, t2->order, &out_dim);
	if (!out_shape) {
		printf("ERROR: Tensor shapes aren't broadcastable.\n");
		return NULL;
	}
	Tensor* result = tensor_create(out_shape, out_dim, extra, DT_INT);
	if (!result){
		free(out_shape);
		return NULL;
	}

	size_t* s1 = calloc(out_dim, sizeof(size_t)); 
	size_t* s2 = calloc(out_dim, sizeof(size_t));
	if(!s1 || !s2){
		printf("ERROR: Fail creating s1 or s2 arrays.\n");
	}
	
	broadcasted_stride(&s1, &s2, out_dim, t1, t2);

	size_t rpos = 0;
	calculate_int((int*)result->data, out_shape, 0, s1, s2, 0, 0, &rpos, out_dim, t1, t2, op);

	free(out_shape);
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
		printf("ERROR: Reshaping not allowed.");
		return -1;
	}

	if(new_order != t->order && t->remaining_extra + t->order < new_order){
		printf("ERROR: Not enough memory for stride and shape.");
		return -1;
	}
	else{
		t->remaining_extra -= (new_order - t->order);
		t->order = new_order;

	}

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
		printf("ERROR: Invalid axis for squeeze operation.\n");
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
		printf("ERROR: Invalid axis for unsqueeze operation.\n");
		return -1;
	}

	if(t->remaining_extra < 1){
		printf("ERROR: Not enough memory for stride and shape.");
		return -1;
	}

	t->remaining_extra--;
	size_t new_order = t->order + 1;

	for(size_t i = t->order; i > idx; i--){
		t->shape[i] =  t->shape[i - 1];
		t->stride[i] = t->stride[i - 1];
	 }

	t->shape[idx] = 1;
	t->stride[idx] = (idx < t->order) ? t->stride[idx + 1] : 1;
	t->order = new_order;
	return 0;
}

Tensor* tensor_clone(Tensor* c){
	Tensor* t = tensor_create(c->shape, c->order, c->remaining_extra, c->dtype);
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
	t->order = 1;
	t->shape[0] = t->size;
	t->stride[0] = 1;
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
	free(t);
}

