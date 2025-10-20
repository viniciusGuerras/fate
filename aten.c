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

	for(int i = 1; i < ndim; i++){
		t->strides[i] = t->strides[i - 1] * t->shape[i - 1];
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
// [5, 3, 2], [1]
// last dimension is equal? yes
// [5, 3, 2], [1, 1, 1]
Tensor* broadcaster(Tensor* t1, Tensor* t2){
	int* tensor_dims = t1->shape;
	int  dim_n       = t1->ndim;
	for(int i = dim_n; i >= 0; i++){
		if(t1->shape[i] != t2->shape[i] && (t1->shape[i]!= 1 || t2->shape[i]!=1){
			//error
			printf("Error broadcasting");
		}
	}
}
*/

int* broadcast_shape(int* s1, int n1, int* s2, int n2, int* out_ndim) {
	int max_dim = n1 > n2 ? n1 : n2;
	int* out_shape = malloc(sizeof(int) * max_dim);

	for(int i = 0; i < max_dim; i++){
		int dim1 = (i < max_dim - n1) ? 1 : s1[i - (max_dim - n1)]; 
		int dim2 = (i < max_dim - n2) ? 1 : s2[i - (max_dim - n2)]; 

		out_shape[i] = (dim1 > dim2) ? dim1 : dim2;
		*out_ndim++;
	}
	return out_shape;
}


Tensor* tensor_apply_binary_op(Tensor* t1, Tensor* t2, tensor_op op){
	if(shape_match(t1->shape, t1->ndim, t2->shape, t2->ndim) != 1){
		printf("Tensor shapes must match.\n");
		return NULL;
	}

	//shape is assured by the shape-match (soon broadcasting)
	Tensor* result = tensor_create(t1->shape, t1->ndim);
	pthread_info* thread_i = get_thread_separation(t1->size);
	int thread_count = thread_i->n_threads;

	int start = 0;
	for(int i = 0; i < thread_count; i++){
		int end = start + thread_i->n_splits + (i == thread_count - 1 ? thread_i->extra : 0);

		pthread_tensor_operation_args* args = malloc(sizeof(pthread_tensor_operation_args));
		args->t1 = t1->data + start;
		args->t2 = t2->data + start;
		args->len = end - start;
		args->tr = result->data + start;
		args->op = op;

		pthread_create(&thread_i->threads[i], NULL, multithreaded_tensor_operation, (void *) args);
		start = end;
	}

	for(int i = 0; i < thread_count; i++){
		pthread_join(thread_i->threads[i], NULL);
	}

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

int main(){
	int s1c[] = {3, 2};
	int s2c[] = {2, 3};
	int ndim;
	int* res = broadcast_shape(s1c, 2, s2c, 2, &ndim);
	if (res) {
	printf("Broadcasted shape: [");
	for(int i=0; i<ndim; i++) printf("%d ", res[i]);
		printf("]\n");
		free(res);
	} else {
		printf("Shapes incompatible\n");
	}
	printf("\n");

	int s1b[] = {8, 1, 6, 1};
	int s2b[] = {7, 1, 5};
	res = broadcast_shape(s1b, 4, s2b, 3, &ndim);
	if (res) {
		printf("Broadcasted shape: [");
	for(int i=0; i<ndim; i++) printf("%d ", res[i]);
		printf("]\n");
		free(res);
	} else {
		printf("Shapes incompatible\n");
	}
/*
	Tensor* t1 = tensor_create(shape, 3);	
	Tensor* t2 = tensor_create(shape, 3);	
	t1->data[4] = 2.0;
	t1->data[1] = 5.0;
	t2->data[2] = 1.4;
	t2->data[10] = 9;
	t2->data[4] = 9;
	Tensor* t3 = tensor_sum(t1, t2);	
	tensor_info(t1);
	tensor_info(t2);
	tensor_info(t3);
	t3 = tensor_sum(t1, t2);
	tensor_info(t3);
	t3 = tensor_subtract(t1, t2);
	tensor_info(t3);
	t3 = tensor_multiply(t1, t2);
	tensor_info(t3);
	t3 = tensor_divide(t1, t2);
	tensor_info(t3);

	tensor_free(t1);
	tensor_free(t2);
	tensor_free(t3);
*/
}
