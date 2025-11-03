/*
 * aqua.c
 * Implements a simple tensor library for numerical operations.
 * Supports creation, manipulation, and arithmetic on tensors.
 * Author: Vinicius Guerra
 * Start-Date: 2025-10-16
 */

#include "utils.h"
#include "aqua.h"

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

/*
 * tensor_create - Creates a contiguous tensor in memory
 * @shape: array of dimensions
 * @order: number of dimensions
 * @extra: extra space for future operations
 * @dtype: data type of tensor elements
 *
 * Returns: Pointer to a Tensor structure, or NULL on allocation failure.
 * The returned tensor stores shape, stride, and data contiguously.
 */
Tensor* tensor_create(const size_t* shape, size_t order, size_t extra, DataType dtype) {
	size_t total_elements = 1;
	for(int i = 0; i < order; i++){
		total_elements *= shape[i];
	}
	size_t type_size = get_dtype_size(dtype);

	size_t size =  sizeof(Tensor) 
			+ (2 * (sizeof(size_t) * (order + extra))) 
			+ (type_size * total_elements);

	char* block = calloc(1, size);
	if (!block) {
		printf("ERROR: Failed to allocate Tensor.\n");
		return NULL;
	}
	Tensor* t = (Tensor*)block;
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

	// Compute strides from last element to first
	t->stride[order - 1] = 1;
	for (size_t i = order - 1; i > 0; i--) {
		t->stride[i - 1] = t->stride[i] * t->shape[i]; 
	}

	return t;
}

/*
* scalar_tensor - Transforms a scalar value into a single element Tensor 
* @v:     scalar value (ScalarType struct)
* @dtype: data type for Tensor's elements
* @extra: extra memory space given for stride and shape
*
* Returns: Pointer to a Tensor structure, or NULL on allocation failure.
*/
Tensor* scalar_tensor(ScalarType v, DataType dtype, size_t extra) {
	size_t shape[1] = {1};
	Tensor* t = tensor_create(shape, 1, extra, dtype);
	if(!t){
		printf("ERROR: Tensor creation failed.\n");
		return NULL;
	}
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

/*
 * broadcast_shape - Calculates the broadcasted shape of two tensors
 * @s1: Pointer to the array representing the shape of the first tensor.
 * @n1: Number of dimensions in the first tensor.
 * @s2: Pointer to the array representing the shape of the second tensor.
 * @n2: Number of dimensions in the second tensor.
 * @out_order: Pointer to a size_t variable where the resulting number of dimensions
 *             (rank) of the broadcasted shape will be stored.
 *
 * Returns:
 * Pointer to a newly allocated array representing the broadcasted shape if successful.
 * Returns NULL if memory allocation fails or if the shapes cannot be broadcast together.
 * Uses out_dim pointer to set it to the size of the new allocated array.
 */
static inline size_t* broadcast_shape(const size_t* s1, size_t n1, const size_t* s2, size_t n2, size_t* out_order) {
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

/*
 * broadcasted_stride - Computes strides for broadcasting two tensors
 * @s1: Pointer to an array where the resulting strides for the first tensor will be stored.
 * @s2: Pointer to an array where the resulting strides for the second tensor will be stored.
 * @out_dim: The total number of dimensions in the broadcasted output.
 * @t1: Pointer to the first tensor structure.
 * @t2: Pointer to the second tensor structure.
 */
static inline void broadcasted_stride(size_t* s1, size_t* s2, 
			size_t out_dim, Tensor* t1, Tensor* t2) {
	for (size_t i = 0; i < out_dim; i++) {
		int t1_idx = (int)i - (int)(out_dim - t1->order);
		int t2_idx = (int)i - (int)(out_dim - t2->order);
		s1[i] = (t1_idx < 0 || t1->shape[t1_idx] == 1) ? 0 : t1->stride[t1_idx];
		s2[i] = (t2_idx < 0 || t2->shape[t2_idx] == 1) ? 0 : t2->stride[t2_idx];
	}
}

/*
 * tensor_apply_binary_op - Applies a binary operation element-wise to two tensors
 * @t1: Pointer to the first tensor.
 * @t2: Pointer to the second tensor.
 * @op: Function pointer representing the binary operation to apply. It must have the
 *      signature: double op(double a, double b);
 *
 * This function performs an element-wise binary operation on two tensors,
 * automatically handling broadcasting according to standard broadcasting rules.
 * The result is stored in a newly created tensor.
*
 * Returns:
 * Pointer to a newly allocated Tensor containing the result of the operation,
 * or NULL if:
 * - The tensor shapes cannot be broadcast together.
 * - Memory allocation fails.
*/

/*--- Double version ---*/
Tensor* tensor_apply_binary_op_double(Tensor* t1, Tensor* t2, tensor_op_double op) {
	size_t out_dim;
	size_t extra = t1->remaining_extra > t2->remaining_extra ? t1->remaining_extra : t2->remaining_extra;
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
		free(out_shape);
		free(s1);
		free(s2);
		return NULL;
	}
	broadcasted_stride(s1, s2, out_dim, t1, t2);

	size_t* indices = (size_t*)alloca(out_dim * sizeof(size_t));
	memset(indices, 0, out_dim * sizeof(size_t));

	double* r_data = (double*)result->data;
	double* t1_data = (double*)t1->data;
	double* t2_data = (double*)t2->data;
	
	size_t offset1 = 0;
	size_t offset2 = 0;

	for (size_t i = 0; i < result->size; ++i) {
		r_data[i] = op(t1_data[offset1], t2_data[offset2]);

		for (int d = out_dim - 1; d >= 0; --d) {
		    indices[d]++;
		    offset1 += s1[d];
		    offset2 += s2[d];
		    if (indices[d] < out_shape[d]) {
			break;
		    }
		    offset1 -= indices[d] * s1[d]; 
		    offset2 -= indices[d] * s2[d];
		    indices[d] = 0;
		}
	}	

	free(out_shape);
	free(s1);
	free(s2);
	return result;
}

/*--- Float version ---*/
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
		free(out_shape);
		free(s1);
		free(s2);
		return NULL;
	}
	broadcasted_stride(s1, s2, out_dim, t1, t2);

	size_t* indices = (size_t*)alloca(out_dim * sizeof(size_t));
	memset(indices, 0, out_dim * sizeof(size_t));

	float* r_data = (float*)result->data;
	float* t1_data = (float*)t1->data;
	float* t2_data = (float*)t2->data;
	
	size_t offset1 = 0;
	size_t offset2 = 0;

	for (size_t i = 0; i < result->size; ++i) {
		r_data[i] = op(t1_data[offset1], t2_data[offset2]);

		for (int d = out_dim - 1; d >= 0; --d) {
		    indices[d]++;
		    offset1 += s1[d];
		    offset2 += s2[d];
		    if (indices[d] < out_shape[d]) {
			break;
		    }
		    offset1 -= indices[d] * s1[d]; 
		    offset2 -= indices[d] * s2[d];
		    indices[d] = 0;
		}
	}	

	free(out_shape);
	free(s1);
	free(s2);
	return result;
}

/*--- Int version ---*/
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
		free(out_shape);
		free(s1);
		free(s2);
		return NULL;
	}
	
	broadcasted_stride(s1, s2, out_dim, t1, t2);

	size_t* indices = (size_t*)alloca(out_dim * sizeof(size_t));
	memset(indices, 0, out_dim * sizeof(size_t));

	int* r_data = (int*)result->data;
	int* t1_data = (int*)t1->data;
	int* t2_data = (int*)t2->data;
	
	size_t offset1 = 0;
	size_t offset2 = 0;

	for (size_t i = 0; i < result->size; ++i) {
		r_data[i] = op(t1_data[offset1], t2_data[offset2]);

		for (int d = out_dim - 1; d >= 0; --d) {
		    indices[d]++;
		    offset1 += s1[d];
		    offset2 += s2[d];
		    if (indices[d] < out_shape[d]) {
			break;
		    }
		    offset1 -= indices[d] * s1[d]; 
		    offset2 -= indices[d] * s2[d];
		    indices[d] = 0;
		}
	}

	free(out_shape);
	free(s1);
	free(s2);
	return result;
}

/*
 * Tensor binary operation wrappers
 * @t1: Pointer to the first tensor.
 * @t2: Pointer to the second tensor.
 *
 * These functions provide convenient wrappers for performing common element-wise
 * binary operations (sum, subtraction, multiplication, division) on tensors of
 * different data types: double, float, and int.
 *
 * Returns:
 * Pointer to a newly allocated tensor containing the result of the operation,
 * or NULL if broadcasting fails or memory allocation fails.
 */

// double
Tensor* tensor_sum_double(Tensor* t1, Tensor* t2)      { return tensor_apply_binary_op_double(t1, t2, op_add_double); }
Tensor* tensor_divide_double(Tensor* t1, Tensor* t2)   { return tensor_apply_binary_op_double(t1, t2, op_div_double); }
Tensor* tensor_subtract_double(Tensor* t1, Tensor* t2) { return tensor_apply_binary_op_double(t1, t2, op_sub_double); }
Tensor* tensor_multiply_double(Tensor* t1, Tensor* t2) { return tensor_apply_binary_op_double(t1, t2, op_mul_double); }

// float
Tensor* tensor_sum_float(Tensor* t1, Tensor* t2)       { return tensor_apply_binary_op_float(t1, t2, op_add_float); }
Tensor* tensor_divide_float(Tensor* t1, Tensor* t2)    { return tensor_apply_binary_op_float(t1, t2, op_div_float); }
Tensor* tensor_subtract_float(Tensor* t1, Tensor* t2)  { return tensor_apply_binary_op_float(t1, t2, op_sub_float); }
Tensor* tensor_multiply_float(Tensor* t1, Tensor* t2)  { return tensor_apply_binary_op_float(t1, t2, op_mul_float); }

// int
Tensor* tensor_sum_int(Tensor* t1, Tensor* t2)         { return tensor_apply_binary_op_int(t1, t2, op_add_int); }
Tensor* tensor_divide_int(Tensor* t1, Tensor* t2)      { return tensor_apply_binary_op_int(t1, t2, op_div_int); }
Tensor* tensor_subtract_int(Tensor* t1, Tensor* t2)    { return tensor_apply_binary_op_int(t1, t2, op_sub_int); }
Tensor* tensor_multiply_int(Tensor* t1, Tensor* t2)    { return tensor_apply_binary_op_int(t1, t2, op_mul_int); }

Tensor* tensor_matmul(Tensor* t1, Tensor* t2) {
	if(t1->shape[t1->order - 1] != t2->shape[t2->order - 2]){
		printf("ERROR: matmul is not possible, inner dimensions don't match\n");
		return NULL;
	}
	size_t out_dim; // will hold the number of dimensions after the second 
	size_t extra = t1->remaining_extra > t2->remaining_extra ? t1->remaining_extra : t2->remaining_extra;

	// needs to get the out shape of dimensions that are > 2
	size_t* out_shape = broadcast_shape(t1->shape, t1->order - 2, t2->shape, t2->order - 2, &out_dim);
	if (!out_shape) {
		printf("ERROR: Tensor shapes aren't broadcastable.\n");
		return NULL;
	}
	Tensor* result = tensor_create(out_shape, out_dim + 2, extra, DT_FLOAT);
	if (!result){
		free(out_shape);
		return NULL;
	}

	size_t* s1 = calloc(out_dim, sizeof(size_t)); 
	size_t* s2 = calloc(out_dim, sizeof(size_t));
	if(!s1 || !s2){
		printf("ERROR: Fail creating s1 or s2 arrays.\n");
		free(out_shape);
		free(s1);
		free(s2);
		return NULL;
	}
	broadcasted_stride(s1, s2, out_dim, t1, t2);

	/*
	  m x n @ n x p = m x p

	  [a11, a12],
	  [a21, a22],
	  [a31, a32]
	  [a41, a42]
	  @ 
	  [b11, b12, b13],
	  [b21, b22, b23],
	  = 
	  [a11 * b11 + a12 * b21 | a11 * b12 + a12 * b22 | a11 * b13 + a12 * b23]
	  [a21 * b11 + a22 * b21 | a21 * b12 + a22 * b22 | a21 * b13 + a22 * b23]
	  [a31 * b11 + a32 * b21 | a31 * b12 + a32 * b22 | a31 * b13 + a32 * b23]
	  [a41 * b11 + a42 * b21 | a41 * b12 + a42 * b22 | a41 * b13 + a42 * b23]
	 
	  shape t1 = [l, m, n]
	  shape t2 = [o, n, p]
	 
	  the dimensions above order 2 are used as batch, so
	  do [m, n] @ [n, p] to l, o dimensions
	 
	  multiplication process:
	  based on the example above for matrices m x n and n x p
	  a loop from 1 -> m (do all the a's variations) is needed
	  inside it a loop from 1 -> p (all the b's)
	  inside it a loop from 1 -> n 
	 
	  wow, O(n^3) i need something quicker
	 
	  pseudocode:
	  m1 = matrice 1
	  m2 = matrice 2
	  mr = result matrice
	 
	  for i in 1->m:
	       for j in 1->p:
	 		total = 0
	 		for k in 1->n:
	 	               total += m1[i][k] * m[k][j]
	 	mr[i][j] = total

	would need to do the indices thing in the above dimensions 
	add the offset of the above dimensions to this
	 */

	float* r_data = (float*)result->data;
	float* t1_data = (float*)t1->data;
	float* t2_data = (float*)t2->data;

	size_t m = t1->shape[t1->order - 2]; // outer dimension of t1 tensor 
	size_t n = t2->shape[t2->order - 1]; // outer dimension of t2 tensor
	size_t p = t1->shape[t1->order - 1]; // inner dimension of one of the two tensors (they must be equal in the inner dimension)
	
	// set result's extra shape
	result->shape[result->order - 2] = m;
	result->shape[result->order - 1] = n;

	// re-calculates the new stride for result
	result->stride[result->order - 1] = 1;
	for (size_t i = result->order - 1; i > 0; i--){
		result->stride[i - 1] = result->stride[i] * result->shape[i]; 
	}


	// multi-dimensional indice counter
	size_t* idx = calloc(out_dim, sizeof(size_t));

	int res = 0;
	// offset for result, t1 and t2
	size_t base_res = 0;
	size_t base_t1 = 0;
	size_t base_t2 = 0;
	size_t total_elements = 1;
	for(int i = 0; i < out_dim; i++) total_elements *= result->shape[i]; // Calculates the total number of elements in the array (for dimensions > 2)
	for(int i = 0 ; i < total_elements; i++){
		for(int i = 0; i < m; i++){ // this logic matches the one explained earlier to calculate the matmul, but now the offset and strides are involved
			for(int j = 0; j < n; j++){
				float total = 0;
				for(int k = 0; k < p; k++){
					size_t pos_a = base_t1 + (t1->stride[t1->order - 2] * i) + (t1->stride[t1->order - 1] * k); 
					size_t pos_b = base_t2 + (t2->stride[t2->order - 2] * k) + (t2->stride[t2->order - 1] * j); 
					float a = t1_data[pos_a];
					float b = t2_data[pos_b];
					total += a * b;
				}
				size_t pos_r = result->stride[result->order - 2] * i + result->stride[result->order - 1] * j;
				r_data[base_res + pos_r] = total;
			}

		}
		/*
		 * code responsible to find the current "multidimensional index" offset
		 *
		 * start an array idx with a position for each element, ex: for 2 dimensions [0, 0]
		 * each new element adds a number until the dimension is filled
		 * [0, 0] -> [0, 1] -> [0, 2] -> ...
		 * then the next number (to the left) is counted when the number matches the shape at that position
		 * for each addition, the stride of the current position is added to the offset
		 */
		for(int h = out_dim - 1; h >= 0; h--){
			idx[h]+=1;
			base_res += result->stride[h]; 
			base_t1 += s1[h]; 
			base_t2 += s2[h]; 
			if(idx[h] < result->shape[h]) break;
			idx[h] = 0;
			base_res -= result->stride[h] * result->shape[h];
			base_t1 -= s1[h]            * result->shape[h];
			base_t2 -= s2[h]            * result->shape[h];
		}
	}
	free(out_shape);
	free(idx);
	free(s1);
	free(s2);
	return result;
}

int tensor_fill_random(Tensor* t){
	if(!t){
		return -1;
	}
	switch(t->dtype){
		case DT_DOUBLE: {
			double* d = (double*)t->data;
			for(int i = 0; i < t->size; i++){
				d[i] = xoshiro_next_double();
			}
			break;
		}
		case DT_FLOAT: {
			float* d = (float*)t->data;
			for(int i = 0; i < t->size; i++){
				d[i] = xoshiro_next_float();
			}
			break;
		}
		case DT_INT: {
			int* d = (int*)t->data;
			for(int i = 0; i < t->size; i++){
				d[i] = xoshiro_next();
			}
			break;
		}
	}
	return 0;
}

int tensor_negation(Tensor* t){
	if(!t){
		return -1;
	}
	switch(t->dtype){
		case DT_DOUBLE: {
			double* d = (double*)t->data;
			for(int i = 0; i < t->size; i++){
				d[i] = -d[i];
			}
			break;
		}
		case DT_FLOAT: {
			float* d = (float*)t->data;
			for(int i = 0; i < t->size; i++){
				d[i] = -d[i];
			}
			break;
		}
		case DT_INT: {
			int* d = (int*)t->data;
			for(int i = 0; i < t->size; i++){
				d[i] = -d[i];
			}
			break;
		}
	}
	return 0;
}

int tensor_abs(Tensor* t){
	if(!t){
		return -1;
	}
	switch(t->dtype){
		case DT_DOUBLE: {
			double* d = (double*)t->data;
			for(int i = 0; i < t->size; i++){
				d[i] = fabs(d[i]);
			}
			break;
		}
		case DT_FLOAT: {
			float* d = (float*)t->data;
			for(int i = 0; i < t->size; i++){
				d[i] = fabs(d[i]);
			}
			break;
		}
		case DT_INT: {
			int* d = (int*)t->data;
			for(int i = 0; i < t->size; i++){
				d[i] = abs(d[i]);
			}
			break;
		}
	}
	return 0;
}

int tensor_exp(Tensor* t){
	if(!t){
		return -1;
	}
	switch(t->dtype){
		case DT_DOUBLE: {
			double* d = (double*)t->data;
			for(int i = 0; i < t->size; i++){
				d[i] = exp(d[i]);
			}
			break;
		}
		case DT_FLOAT: {
			float* d = (float*)t->data;
			for(int i = 0; i < t->size; i++){
				d[i] = expf(d[i]);
			}
			break;
		}
		case DT_INT: {
			int* d = (int*)t->data;
			for(int i = 0; i < t->size; i++){
				d[i] = (int)exp(d[i]);
			}
			break;
		}
	}
	return 0;
}


/*
 * tensor reshape - Updates the tensor shape to the one passed to the function
 * @t: Base tensor to apply the operation on.
 * @shape: desired new shape for tensor.
 * @new_order: order of the result tensor (size of the shape array)
 *
 * Returns: 0 operation completed sucessfully, -1 error during the operation, 
 */
int tensor_reshape(Tensor* t, size_t* shape, size_t new_order){
	size_t new_size = 1;
	for(int i = 0; i < new_order; i++){
		new_size *= shape[i];
	}
	if(new_size != t->size){
		printf("ERROR: Reshaping not allowed.\n");
		return -1;
	}

	if(new_order != t->order && t->remaining_extra + t->order < new_order){
		printf("ERROR: Not enough memory for stride and shape.\n");
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

/*
 * tensor squeeze - Removes all dimensions of size 1 from Tensor
 * @t: Base tensor to apply the operation on.
 *
 * Returns: 0 operation completed sucessfully, -1 error during the operation, 
 */
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

/*
 * tensor squeeze - Removes dimension from Tensor at the index if it's 1 
 * @t: Base tensor to apply the operation on.
 * @idx: dimension index to remove 
 *
 * Returns: 0 operation completed sucessfully, -1 error during the operation, 
 */
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
	t->remaining_extra++;
	return 0;
}

/*
 * tensor unsqueeze - Adds a new dimension of 1 at the index
 * @t: Base tensor to apply the operation on.
 * @idx: dimension index to remove 
 *
 * Returns: 0 operation completed sucessfully, -1 error during the operation, 
 */
int tensor_unsqueeze(Tensor* t, size_t idx){
	if(!t || idx > t->order){
		printf("ERROR: Invalid axis for unsqueeze operation.\n");
		return -1;
	}

	if(t->remaining_extra < 1){
		printf("ERROR: Not enough memory for stride and shape.\n");
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

/*
 * tensor unsqueeze - Adds a new dimension of 1 at the index
 * @t: Base tensor to apply the operation on.
 * @idx: dimension index to remove 
 *
 * Returns: 0 operation completed sucessfully, -1 error during the operation,
 */
int tensor_transpose(Tensor* t, size_t from, size_t to){
	if(from > t->order || to > t->order || t->order < 2){
		printf("ERROR: Impossible to transpose.\n");
		return -1;
	}

	size_t temp = t->shape[from];
	t->shape[from] = t->shape[to];
	t->shape[to] = temp;

	return 0;
}

/*
 * tensor permute - Changes the order of the Tensor's shape
 * @t: Base tensor to apply the operation on.
 * @permute_arr: New order for Tensor's shape
 * @permute_arr_size: Size of the permute_arr
 *
 * Returns: 0 for operation completed sucessfully and -1 for error during the operation,
 */
int tensor_permute(Tensor* t, size_t* permute_arr, size_t permute_arr_size){
	if(permute_arr_size != t->order){
		printf("ERROR: Impossible to permute.\n");
		return -1;
	}

	size_t* temp_shape = malloc(sizeof(size_t) * t->order);
	if(!temp_shape){
		printf("ERROR: Temporary shape malloc failed.\n");
		return -1;
	}

	for(size_t i = 0; i < permute_arr_size; i++){
		size_t ith_permute = permute_arr[i];
		if(ith_permute >= t->order){
			printf("ERROR: Impossible to permute.\n");
			free(temp_shape);
			return -1;
		}
		temp_shape[i] = t->shape[ith_permute]; 
	}

	for(int i = 0; i < t->order; i++){
		t->shape[i] = temp_shape[i];
	}

	free(temp_shape);
	return 0;
}


/*
 * tensor clone - Receives a Tensor and new one. Copies the data from the received to created one.
 * @c: Base tensor to be copied 
 *
 * Returns: Tensor pointer to the newly created one.
 */
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

/*
 * tensor_flatten - Reduces all the tensor shapes to just a single contiguous array
 * @t: Tensor to be flatten
 *
 * Returns: 0 for operation completed sucessfully and -1 for error during the operation,
 */
int tensor_flatten(Tensor* t){
	if(!t){
		return -1;
	}
	t->order = 1;
	t->shape[0] = t->size;
	t->stride[0] = 1;
	return 0;
}

/*
 * tensor_print_recursive - Recursion that goes trough each dimension printing elements and [, ]
 * @data: data array as a void pointer
 * @shape: shape array from tensor t  
 * @dtype: tensor idata type 
 * @order: number of dimensions the tensor has
 * @pos: current position in the shape array
 * @offset: memory offset in the data array
 */
void tensor_print_recursive(void* data, size_t* shape, DataType dtype, size_t order, size_t pos, size_t offset){
	if (pos == order - 1) {
		printf("[");
		for (size_t i = 0; i < shape[pos]; i++) {
			switch(dtype){
				case DT_DOUBLE:{
					printf("%lf", ((double*)data)[offset + i]);
					break;
				}
				case DT_FLOAT:{
					printf("%f", ((float*)data)[offset + i]);
					break;
				}
				case DT_INT:{
					printf("%d", ((int*)data)[offset + i]);
					break;
				}
			}
			if (i < shape[pos] - 1){
				printf(", ");
			} 
		}
		printf("]");
		return;
	}

	printf("[");

	size_t inner_size = 1;
        for (size_t j = pos + 1; j < order; j++) inner_size *= shape[j];

	for(size_t i = 0; i < shape[pos]; i++){
		tensor_print_recursive(data, shape, dtype, order, pos + 1, offset + i * inner_size);
	}

	printf("]");
}

void tensor_print(Tensor* t){
	switch(t->dtype){
		case DT_DOUBLE: {
			tensor_print_recursive(t->data, t->shape, t->dtype, t->order, 0, 0);
			break;
		}
		case DT_FLOAT: {
			tensor_print_recursive(t->data, t->shape, t->dtype, t->order, 0, 0);
			break;
		}
		case DT_INT: {
			tensor_print_recursive(t->data, t->shape, t->dtype, t->order, 0, 0);
			break;
		}
	}
	printf("\n");
	return;
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

