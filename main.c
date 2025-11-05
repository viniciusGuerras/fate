#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "tensor.h"

int main(void) {
    printf("🚀 Starting Giant Tensor Velocity Test (Arena-based)\n");

    /* 1️⃣ Create a request state / arena */
    RequestState* rs = arena_create();
    if (!rs) {
        fprintf(stderr, "❌ Failed to create RequestState (arena)\n");
        return EXIT_FAILURE;
    }

    /* 2️⃣ Declare tensor shapes */
    size_t shape1[4] = {16, 128, 256, 512};
    size_t shape2[4] = {1, 128, 1, 512};
    size_t shape3[4] = {16, 1, 256, 1};

    printf("Declaring tensors...\n");

    /* 3️⃣ Request tensors (deferred allocation) */
    tensor_request(rs, "t1", shape1, 4, 0, DT_FLOAT);
    tensor_request(rs, "t2", shape2, 4, 0, DT_FLOAT);
    tensor_request(rs, "t3", shape3, 4, 0, DT_FLOAT);

    /* Declare results */
    tensor_op_elementwise_request(rs, "sum12", "t1", "t2");
    tensor_op_elementwise_request(rs, "prod23", "t2", "t3");
    tensor_op_elementwise_request(rs, "sub31", "t3", "t1");

    /* 4️⃣ Initialize arena (actual memory allocation) */
    arena_initialize(rs);

    /* 5️⃣ Instantiate tensors */
    Tensor* t1 = tensor_instantiate(rs, "t1");
    Tensor* t2 = tensor_instantiate(rs, "t2");
    Tensor* t3 = tensor_instantiate(rs, "t3");
    Tensor* sum12 = tensor_instantiate(rs, "sum12");
    Tensor* prod23 = tensor_instantiate(rs, "prod23");
    Tensor* sub31 = tensor_instantiate(rs, "sub31");

    if (!t1 || !t2 || !t3 || !sum12 || !prod23 || !sub31) {
        fprintf(stderr, "❌ Failed to instantiate one or more tensors\n");
        return EXIT_FAILURE;
    }

    /* 6️⃣ Fill tensors with random data */
    printf("Filling tensors with random data...\n");
    xoshiro_seed((uint64_t)time(NULL));
    tensor_fill_random(t1);
    tensor_fill_random(t2);
    tensor_fill_random(t3);

    /* 7️⃣ Print shapes */
    printf("Tensor shapes:\n");
    tensor_print_shape(t1);
    tensor_print_shape(t2);
    tensor_print_shape(t3);

    /* 8️⃣ Perform operations and time them */
    struct timespec start, end;
    double elapsed;

    printf("\n⚙️ Performing tensor operations...\n");

    // Sum
    clock_gettime(CLOCK_MONOTONIC, &start);
    tensor_sum_float(sum12, t1, t2);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("✅ t1 + t2 finished in %.3f s\n", elapsed);

    // Multiply
    clock_gettime(CLOCK_MONOTONIC, &start);
    tensor_multiply_float(prod23, t2, t3);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("✅ t2 * t3 finished in %.3f s\n", elapsed);

    // Subtract
    clock_gettime(CLOCK_MONOTONIC, &start);
    tensor_subtract_float(sub31, t3, t1);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("✅ t3 - t1 finished in %.3f s\n", elapsed);

    /* 9️⃣ Print sample outputs */
    float* s = (float*)sum12->data;
    printf("\nSample values from t1 + t2: [%.5f, %.5f, %.5f, ...]\n", s[0], s[1], s[2]);


    printf("\n🏁 Giant Tensor Velocity Test Completed Successfully!\n");
    return EXIT_SUCCESS;
}

