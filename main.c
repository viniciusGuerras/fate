#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "tensor.h"

int main(void) {
    RequestState* rs = arena_create();
    if (!rs) {
        return -1;
    }

    size_t shape1[4] = {16, 128, 256, 512};
    size_t shape2[4] = {1, 128, 1, 512};
    size_t shape3[4] = {16, 1, 256, 1};

    tensor_request(rs, "t1", shape1, 4, 0, DT_FLOAT);
    tensor_request(rs, "t2", shape2, 4, 0, DT_FLOAT);
    tensor_request(rs, "t3", shape3, 4, 0, DT_FLOAT);

    tensor_op_elementwise_request(rs, "sum12", "t1", "t2");
    tensor_op_elementwise_request(rs, "prod23", "t2", "t3");
    tensor_op_elementwise_request(rs, "sub31", "t3", "t1");

    arena_initialize(rs);

    Tensor* t1 = tensor_instantiate(rs, "t1");
    Tensor* t2 = tensor_instantiate(rs, "t2");
    Tensor* t3 = tensor_instantiate(rs, "t3");
    Tensor* sum12 = tensor_instantiate(rs, "sum12");
    Tensor* prod23 = tensor_instantiate(rs, "prod23");
    Tensor* sub31 = tensor_instantiate(rs, "sub31");

    if (!t1 || !t2 || !t3 || !sum12 || !prod23 || !sub31) {
        return -1;
    }

    xoshiro_seed((uint64_t)time(NULL));
    tensor_fill_random(t1);
    tensor_fill_random(t2);
    tensor_fill_random(t3);

   struct timespec start, end;
    double elapsed;

    // Sum
    clock_gettime(CLOCK_MONOTONIC, &start);
    tensor_sum_float(sum12, t1, t2);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("t1 + t2 finished in %.6f s\n", elapsed);

    // Multiply
    clock_gettime(CLOCK_MONOTONIC, &start);
    tensor_multiply_float(prod23, t2, t3);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("t2 * t3 finished in %.6f s\n", elapsed);

    // Subtract
    clock_gettime(CLOCK_MONOTONIC, &start);
    tensor_subtract_float(sub31, t3, t1);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("t3 - t1 finished in %.6f s\n", elapsed);

    float* s = (float*)sum12->data;
    printf("\nSample values from t1 + t2: [%.5f, %.5f, %.5f, ...]\n", s[0], s[1], s[2]);
    return -1;
}

