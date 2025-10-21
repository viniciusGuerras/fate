#include <sys/sysctl.h>
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>

int thread_count = -1;

typedef struct {
	float*  data;
	int*   shape;
	int* strides;
	int     ndim;
	int     size;
} Tensor;

typedef double (*tensor_op)(double, double);

typedef struct {
	float* t1;
	float* t2;
	float* tr;
	int len;
	tensor_op op;
} pthread_tensor_operation_args;

typedef struct {
	pthread_t* threads;
	int n_threads;
	int n_splits;
	int extra;
} pthread_info;

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

void tensor_info(Tensor* t) {
    printf("Shape: (");
    for (int i = 0; i < t->ndim; i++) {
        printf("%d", t->shape[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf(")\n");

    printf("Strides: (");
    for (int i = 0; i < t->ndim; i++) {
        printf("%d", t->strides[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf(")\n");

    printf("Total size: %d elements\n", t->size);
    printf("Data preview: [");
    int limit = (t->size < 10) ? t->size : 10;
    for (int i = 0; i < limit; i++) {
        printf("%.2f", t->data[i]);
        if (i < limit - 1) printf(", ");
    }
    if (t->size > 10) printf(", ...");
    printf("]\n");
}
	
int get_cores(){
	if(thread_count != -1){
		return thread_count;
	}
	int logical_cores;
	size_t len = sizeof(logical_cores);

	if (sysctlbyname("hw.logicalcpu", &logical_cores, &len, NULL, 0) == -1) {
		perror("sysctlbyname hw.logicalcpu");
		return 0;
	}

	thread_count = logical_cores;
	return logical_cores;
}

pthread_info* get_thread_separation(int size){
	pthread_info* pthread_info = malloc(sizeof(pthread_info));

	int thread_count = get_cores();
	int splits = size / thread_count; 
	int extra  = size % thread_count;

	pthread_info->threads = malloc(sizeof(pthread_t) * thread_count);
	pthread_info->n_threads = thread_count;
	pthread_info->n_splits = splits;
	pthread_info->extra = extra;

	return pthread_info;
}

void* multithreaded_tensor_operation(void* arg) {
    pthread_tensor_operation_args* a = (pthread_tensor_operation_args *)arg;
    tensor_op op = a->op;

    for (int i = 0; i < a->len; i++) {
        a->tr[i] = op(a->t1[i], a->t2[i]);
    }

    free(a);
    return NULL;
}
/*
int get_offset(int* shape, int* strides, int ndim, int* indices) {
	int offset = 0;
	for (int i = 0; i < ndim; i++){
		int idx = indices[i];
		if (shape[i] == 1) idx = 0; 
		offset += idx * strides;
	}
	return offset;
}
*/

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

    printf("Broadcasted shape: [");
    for(int i = 0; i < out_dim; i++) printf("%d%s", out_shape[i], i==out_dim-1?"":", ");
    printf("]\n");

    int t1_broadcasted_strides[out_dim];
    int t2_broadcasted_strides[out_dim];

    for(int i = 0; i < out_dim; i++){
        int t1_idx = i - (out_dim - t1->ndim);
        int t2_idx = i - (out_dim - t2->ndim);
        t1_broadcasted_strides[i] = (t1_idx < 0 || t1->shape[t1_idx] == 1) ? 0 : t1->strides[t1_idx];
        t2_broadcasted_strides[i] = (t2_idx < 0 || t2->shape[t2_idx] == 1) ? 0 : t2->strides[t2_idx];
    }

    printf("t1 broadcasted strides: [");
    for(int i = 0; i < out_dim; i++) printf("%d%s", t1_broadcasted_strides[i], i==out_dim-1?"":", ");
    printf("]\n");

    printf("t2 broadcasted strides: [");
    for(int i = 0; i < out_dim; i++) printf("%d%s", t2_broadcasted_strides[i], i==out_dim-1?"":", ");
    printf("]\n");

    Tensor* result = tensor_create(out_shape, out_dim);

    int* idx = calloc(out_dim, sizeof(int));

    for(int i = 0; i < result->size; i++){
        int offset1 = 0, offset2 = 0;

        // Compute offsets
        for(int d = 0; d < out_dim; d++){
            offset1 += idx[d] * t1_broadcasted_strides[d];
            offset2 += idx[d] * t2_broadcasted_strides[d];
        }

        // Print debug info
        printf("i=%d, idx=[", i);
        for(int d = 0; d < out_dim; d++) printf("%d%s", idx[d], d==out_dim-1?"":", ");
        printf("], offset1=%d, offset2=%d, t1=%f, t2=%f, ", 
               offset1, offset2, t1->data[offset1], t2->data[offset2]);

        result->data[i] = op(t1->data[offset1], t2->data[offset2]);
        printf("result=%f\n", result->data[i]);

        // Increment idx like an odometer
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

int main() {
    // Tiny shapes
    int shape1[2] = {2, 1};   // 2 rows, 1 column
    int shape2[2] = {1, 3};   // 1 row, 3 columns

    Tensor* t1 = tensor_create(shape1, 2);
    Tensor* t2 = tensor_create(shape2, 2);

    // Fill t1
    t1->data[0] = 10;
    t1->data[1] = 20;

    // Fill t2
    t2->data[0] = 1;
    t2->data[1] = 2;
    t2->data[2] = 3;

    printf("=== Tensor Apply Binary Op Debug ===\n");
    Tensor* t3 = tensor_sum(t1, t2);

    // Print info
    tensor_info(t1);
    tensor_info(t2);
    tensor_info(t3);

    tensor_free(t1);
    tensor_free(t2);
    tensor_free(t3);

    return 0;
}
