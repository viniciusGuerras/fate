/*
 * aqua.c
 * Implements a simple tensor library for numerical operations.
 * Supports creation, manipulation, and arithmetic on tensors.
 * Author: Vinicius Guerra
 * Start-Date: 2025-10-16
 */

#include "tensor.h"
static HashSet* hs = NULL;

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

Tensor* tensor_instantiate(RequestState* rs, char* identifier){
	char* block = (char*)arena_request(rs, identifier);

	Tensor* t = (Tensor*)block;
	Tensor* retrieved = hashset_get(hs, identifier);
	t->order = retrieved->order;
	t->order_max = retrieved->order_max;
	t->dtype = retrieved->dtype;
	t->size  = retrieved->size;

	char* ptr = block + sizeof(Tensor);
	
	t->shape = (size_t*)ptr;

	ptr += sizeof(size_t) * (retrieved->order_max);
	t->stride = (size_t*)ptr; 

	ptr += sizeof(size_t) * (retrieved->order_max);
	t->data = (void*)ptr; 

	for (size_t i = 0; i < retrieved->order; i++) {
		t->shape[i]  = retrieved->shape[i];
		t->stride[i] = retrieved->stride[i];
	}

	switch(t->dtype){
		case DT_DOUBLE: {
			double* d = (double*)t->data;
			for(int i = 0; i < t->size; i++){
				d[i] = 0;
			}
			break;
		}
		case DT_FLOAT: {
			float* d = (float*)t->data;
			for(int i = 0; i < t->size; i++){
				d[i] = 0;
			}
			break;
		}
		case DT_INT: {
			int* d = (int*)t->data;
			for(int i = 0; i < t->size; i++){
				d[i] = 0;
			}
			break;
		}
	}

	return t;
}

// functions for each type and operation 
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


void tensor_request(RequestState* rs, char* identifier, const size_t* shape, size_t order, size_t extra, DataType dtype) {
	size_t total_elements = 1;
	for(int i = 0; i < order; i++){ 
		total_elements *= shape[i];
	}

	size_t type_size = get_dtype_size(dtype);

	//find the total memory of this tensor
	size_t size =  sizeof(Tensor) 
			+ (2 * (sizeof(size_t) * (order + extra))) 
			+ (type_size * total_elements);

	arena_add(rs, identifier, size); //creates an entry to it in the arena

	// initiate the hashset
	if(!hs){
		hs = hashset_create();
	}

	// save's it locally to later instantiate
	Tensor* t = (Tensor*)malloc(sizeof(Tensor));
	if(!t){
		return;
	}
	t->dtype     = dtype;
	t->order     = order;
	t->order_max = order + extra;
	t->shape     = calloc(order + extra, sizeof(size_t));
	t->stride    = calloc(order + extra, sizeof(size_t));
	t->size      = total_elements;
	if(!t->shape){
		free(t);
		return;
	}

	for(int i = 0; i < order; i++){
		t->shape[i] = shape[i];
	}

	// Compute strides from last element to first
	t->stride[order - 1] = 1;
	for (size_t i = order - 1; i > 0; i--) {
		t->stride[i - 1] = t->stride[i] * t->shape[i]; 
	}

	hashset_add(hs, identifier, (void*)t);
}

void tensor_op_elementwise_request(RequestState* rs, char* identifier_r, char* identifier_1, char* identifier_2){
	Tensor* a = hashset_get(hs, identifier_1);
	Tensor* b = hashset_get(hs, identifier_2);

	size_t out_dim;
	size_t extra = a->order_max > b->order_max ? a->order_max - a->order: b->order_max - b->order;
	size_t* out_shape = broadcast_shape(a->shape, a->order, b->shape, b->order, &out_dim);
	if (!out_shape) {
		printf("ERROR: Tensor shapes aren't broadcastable.\n");
		return;
	}

	tensor_request(rs, identifier_r, out_shape, out_dim, extra, DT_FLOAT);
}

void tensor_matmul_request(RequestState* rs, char* identifier,
			const size_t* shape_1, size_t order_1, size_t extra_1, DataType dtype_1,
			const size_t* shape_2, size_t order_2, size_t extra_2, DataType dtype_2) {
	if(shape_1[order_1 - 1] != shape_2[order_2 - 2]){
		printf("ERROR: matmul is not possible, inner dimensions don't match\n");
		return;
	}
	size_t out_dim; // will hold the number of dimensions > 2
	size_t extra = order_1 + extra_1 > order_2 + extra_2 ? extra_1 : extra_2;
	//
	// needs to get the out shape of dimensions that are > 2
	size_t* out_shape = broadcast_shape(shape_1, order_1 - 2, shape_2, order_2 - 2, &out_dim);
	if (!out_shape) {
		printf("ERROR: Tensor shapes aren't broadcastable.\n");
		return;
	}
	tensor_request(rs, identifier, out_shape, out_dim + 2, extra, DT_FLOAT);
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
void tensor_apply_binary_op_double(Tensor* t1, Tensor* t2, Tensor* r, tensor_op_double op) {
	size_t* s1 = malloc(r->order * sizeof(size_t));
	size_t* s2 = malloc(r->order * sizeof(size_t));
	if(!s1 || !s2){
		printf("ERROR: Fail creating s1 or s2 arrays.\n");
		free(r->shape);
		free(s1);
		free(s2);
		return;
	}
	broadcasted_stride(s1, s2, r->order, t1, t2);

	size_t* indices = (size_t*)alloca(r->order * sizeof(size_t));
	memset(indices, 0, r->order * sizeof(size_t));

	double* r_data = (double*)r->data;
	double* t1_data = (double*)t1->data;
	double* t2_data = (double*)t2->data;
	
	size_t offset1 = 0;
	size_t offset2 = 0;

	for (size_t i = 0; i < r->size; ++i) {
		r_data[i] = op(t1_data[offset1], t2_data[offset2]);

		for (int d = r->order - 1; d >= 0; --d) {
		    indices[d]++;
		    offset1 += s1[d];
		    offset2 += s2[d];
		    if (indices[d] < r->shape[d]) {
			break;
		    }
		    offset1 -= indices[d] * s1[d]; 
		    offset2 -= indices[d] * s2[d];
		    indices[d] = 0;
		}
	}	
	free(s1);
	free(s2);
}

/*--- Float version ---*/
void tensor_apply_binary_op_float(Tensor* t1, Tensor* t2, Tensor* r, tensor_op_float op) {
	size_t* s1 = calloc(r->order, sizeof(size_t)); 
	size_t* s2 = calloc(r->order, sizeof(size_t));
	if(!s1 || !s2){
		printf("ERROR: Fail creating s1 or s2 arrays.\n");
		free(r->shape);
		free(s1);
		free(s2);
		return;
	}
	broadcasted_stride(s1, s2, r->order, t1, t2);

	size_t* indices = (size_t*)alloca(r->order * sizeof(size_t));
	memset(indices, 0, r->order * sizeof(size_t));

	float* r_data = (float*)r->data;
	float* t1_data = (float*)t1->data;
	float* t2_data = (float*)t2->data;
	
	size_t offset1 = 0;
	size_t offset2 = 0;

	for (size_t i = 0; i < r->size; ++i) {
		r_data[i] = op(t1_data[offset1], t2_data[offset2]);
		for (int d = r->order - 1; d >= 0; --d) {
		    indices[d]++;
		    offset1 += s1[d];
		    offset2 += s2[d];
		    if (indices[d] < r->shape[d]) {
			break;
		    }
		    offset1 -= indices[d] * s1[d]; 
		    offset2 -= indices[d] * s2[d];
		    indices[d] = 0;
		}
	}	
	free(s1);
	free(s2);
}

/*--- Int version ---*/
void tensor_apply_binary_op_int(Tensor* t1, Tensor* t2, Tensor* r, tensor_op_int op) {
	size_t* s1 = calloc(r->order, sizeof(size_t)); 
	size_t* s2 = calloc(r->order, sizeof(size_t));
	if(!s1 || !s2){
		printf("ERROR: Fail creating s1 or s2 arrays.\n");
		free(r->shape);
		free(s1);
		free(s2);
		return;
	}
	
	broadcasted_stride(s1, s2, r->order, t1, t2);

	size_t* indices = (size_t*)alloca(r->order * sizeof(size_t));
	memset(indices, 0, r->order * sizeof(size_t));

	int* r_data = (int*)r->data;
	int* t1_data = (int*)t1->data;
	int* t2_data = (int*)t2->data;
	
	size_t offset1 = 0;
	size_t offset2 = 0;

	for (size_t i = 0; i < r->size; ++i) {
		r_data[i] = op(t1_data[offset1], t2_data[offset2]);

		for (int d = r->order - 1; d >= 0; --d) {
		    indices[d]++;
		    offset1 += s1[d];
		    offset2 += s2[d];
		    if (indices[d] < r->shape[d]) {
			break;
		    }
		    offset1 -= indices[d] * s1[d]; 
		    offset2 -= indices[d] * s2[d];
		    indices[d] = 0;
		}
	}
	free(s1);
	free(s2);
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
int tensor_sum_double(Tensor* r, Tensor* t1, Tensor* t2)      { tensor_apply_binary_op_double(t1, t2, r, op_add_double); }
int tensor_divide_double(Tensor* r, Tensor* t1, Tensor* t2)   { tensor_apply_binary_op_double(t1, t2, r, op_div_double); }
int tensor_subtract_double(Tensor* r, Tensor* t1, Tensor* t2) { tensor_apply_binary_op_double(t1, t2, r, op_sub_double); }
int tensor_multiply_double(Tensor* r, Tensor* t1, Tensor* t2) { tensor_apply_binary_op_double(t1, t2, r, op_mul_double); }

// float
int tensor_sum_float(Tensor* r, Tensor* t1, Tensor* t2)       { tensor_apply_binary_op_float(t1, t2, r, op_add_float); }
int tensor_divide_float(Tensor* r, Tensor* t1, Tensor* t2)    { tensor_apply_binary_op_float(t1, t2, r, op_div_float); }
int tensor_subtract_float(Tensor* r, Tensor* t1, Tensor* t2)  { tensor_apply_binary_op_float(t1, t2, r, op_sub_float); }
int tensor_multiply_float(Tensor* r, Tensor* t1, Tensor* t2)  { tensor_apply_binary_op_float(t1, t2, r, op_mul_float); }

// int
int tensor_sum_int(Tensor* r, Tensor* t1, Tensor* t2)         { tensor_apply_binary_op_int(t1, t2, r, op_add_int); }
int tensor_divide_int(Tensor* r, Tensor* t1, Tensor* t2)      { tensor_apply_binary_op_int(t1, t2, r, op_div_int); }
int tensor_subtract_int(Tensor* r, Tensor* t1, Tensor* t2)    { tensor_apply_binary_op_int(t1, t2, r, op_sub_int); }
int tensor_multiply_int(Tensor* r, Tensor* t1, Tensor* t2)    { tensor_apply_binary_op_int(t1, t2, r, op_mul_int); }

Tensor* tensor_matmul(Tensor* t1, Tensor* t2, Tensor* r) {
	if(t1->shape[t1->order - 1] != t2->shape[t2->order - 2]){
		printf("ERROR: matmul is not possible, inner dimensions don't match\n");
		return NULL;
	}
	size_t* s1 = calloc(r->order, sizeof(size_t)); 
	size_t* s2 = calloc(r->order, sizeof(size_t));
	if(!s1 || !s2){
		printf("ERROR: Fail creating s1 or s2 arrays.\n");
		free(s1);
		free(s2);
		return NULL;
	}
	broadcasted_stride(s1, s2, r->order, t1, t2);

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
	  do [m, n] @ [n, p] to l and o dimensions
	 
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

	would need to do the indices indexing in the above dimensions 
	add the offset of the above dimensions to this
	 */

	float* r_data = (float*)r->data;
	float* t1_data = (float*)t1->data;
	float* t2_data = (float*)t2->data;

	size_t t1_outer_idx = t1->order - 2;
	size_t t2_outer_idx = t2->order - 1;
	size_t inner_idx    = t1->order - 2;

	size_t m = t1->shape[t1_outer_idx]; // outer dimension of t1 tensor 
	size_t n = t2->shape[t2_outer_idx]; // outer dimension of t2 tensor
	size_t p = t1->shape[inner_idx];    // inner dimension of one of the two tensors (they must be equal in the inner dimension)
	
	// set result's extra shape
	r->shape[r->order - 2] = m;
	r->shape[r->order - 1] = n;

	// re-calculates the new stride for result
	r->stride[r->order - 1] = 1;
	for (size_t i = r->order - 1; i > 0; i--){
		r->stride[i - 1] = r->stride[i] * r->shape[i]; 
	}

	size_t* idx = calloc(r->order, sizeof(size_t)); // multi-dimensional indice counter
	int res = 0;

	// offset for result, t1 and t2
	size_t base_res = 0;
	size_t base_t1 = 0;
	size_t base_t2 = 0;
	size_t total_elements = 1;
	for(int i = 0; i < r->order; i++) total_elements *= r->shape[i]; // Calculates the total number of elements in the array (for dimensions > 2)
	for(int l = 0 ; l < total_elements; l++){
		for(int i = 0; i < m; i++){ // this logic matches the one explained earlier to calculate the matmul, but now the offset and strides are involved
			for(int j = 0; j < n; j++){
				float total = 0;
				for(int k = 0; k < p; k++){
					total = t1_data[base_t1 + (t1->stride[t1_outer_idx] * i) + (t1->stride[inner_idx] * k)] + 
						t2_data[base_t2 + (t2->stride[inner_idx] * k) + (t2->stride[t2_outer_idx] * j)]; 
				}
				r_data[base_res + r->stride[r->order - 2] * i + r->stride[r->order - 1] * j] = total;
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
		for(int h = r->order - 1; h >= 0; h--){
			idx[h]++;
			base_res += r->stride[h]; 
			base_t1  += s1[h]; 
			base_t2  += s2[h]; 
			if(idx[h] < r->shape[h]) break;
			idx[h] = 0;
			base_res -= r->stride[h] * r->shape[h];
			base_t1  -= s1[h]        * r->shape[h];
			base_t2  -= s2[h]        * r->shape[h];
		}
	}
	free(idx);
	free(s1);
	free(s2);
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

	if(new_order != t->order && t->order_max < new_order){
		printf("ERROR: Not enough memory for stride and shape.\n");
		return -1;
	}
	else{
		t->order_max -= new_order;
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

	if((t->order_max - t->order) < 1){
		printf("ERROR: Not enough memory for stride and shape.\n");
		return -1;
	}

	for(size_t i = t->order; i > idx; i--){
		t->shape[i] =  t->shape[i - 1];
		t->stride[i] = t->stride[i - 1];
	 }

	t->shape[idx] = 1;
	t->stride[idx] = (idx < t->order) ? t->stride[idx + 1] : 1;
	t->order++;
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
Tensor* tensor_clone(Tensor* c){
	Tensor* t = tensor_create(c->shape, c->order, c->order_max, c->dtype);
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
*/

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

