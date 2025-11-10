#include "test.h"
#include "tensor.h"
#include "arena.h"
#include "structures/hashset.h"
#include "utils.h"
#include "rng.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

// Arena Tests
void test_arena_init() {
    printf("\n=== Testing arena_init ===\n");
    int result = arena_init(1024);
    ASSERT_EQUAL(result, 0);

    // Test with zero size (should fail)
    arena_free();
    result = arena_init(0);
    // This might succeed or fail depending on implementation
    arena_free();
}

void test_arena_alloc() {
    printf("\n=== Testing arena_alloc ===\n");
    arena_init(1024);

    void* ptr1 = arena_alloc(1, 100);
    ASSERT_BOOL(ptr1 != NULL, 1);

    void* ptr2 = arena_alloc(1, 200);
    ASSERT_BOOL(ptr2 != NULL, 1);

    // Test allocation beyond limit
    arena_init(100);
    void* ptr3 = arena_alloc(1, 200);
    ASSERT_BOOL(ptr3 == NULL, 1);

    arena_free();
}

void test_arena_free() {
    printf("\n=== Testing arena_free ===\n");
    arena_init(1024);
    int result = arena_free();
    ASSERT_EQUAL(result, 0);

    // Test double free
    result = arena_free();
    ASSERT_EQUAL(result, -1);
}

// Tensor Creation Tests 
void test_tensor_create() {
    printf("\n=== Testing tensor_create ===\n");
    arena_init(10000);

    size_t shape[] = {2, 3, 4};
    Tensor* t = tensor_create(shape, 3, 2, DT_INT);
    ASSERT_BOOL(t != NULL, 1);
    ASSERT_EQUAL(t->order, 3);
    ASSERT_EQUAL(t->size, 24);
    ASSERT_EQUAL(t->dtype, DT_INT);
    ASSERT_EQUAL(t->shape[0], 2);
    ASSERT_EQUAL(t->shape[1], 3);
    ASSERT_EQUAL(t->shape[2], 4);

    arena_free();
}

void test_scalar_tensor() {
    printf("\n=== Testing scalar_tensor ===\n");
    arena_init(10000);

    ScalarType v;
    v.i = 42;
    Tensor* t = scalar_tensor(v, DT_INT, 2);
    ASSERT_BOOL(t != NULL, 1);
    ASSERT_EQUAL(t->order, 1);
    ASSERT_EQUAL(t->size, 1);
    ASSERT_EQUAL(((int*)t->data)[0], 42);

    v.f = 3.14f;
    Tensor* t2 = scalar_tensor(v, DT_FLOAT, 2);
    ASSERT_BOOL(t2 != NULL, 1);
    ASSERT_EQUAL(((float*)t2->data)[0] == 3.14f, 1);

    v.d = 2.718;
    Tensor* t3 = scalar_tensor(v, DT_DOUBLE, 2);
    ASSERT_BOOL(t3 != NULL, 1);
    ASSERT_EQUAL(((double*)t3->data)[0] == 2.718, 1);

    arena_free();
}

void test_tensor_clone() {
    printf("\n=== Testing tensor_clone ===\n");
    arena_init(10000);

    size_t shape[] = {2, 3};
    Tensor* t1 = tensor_create(shape, 2, 2, DT_INT);
    int* data1 = (int*)t1->data;
    for (int i = 0; i < 6; i++) {
        data1[i] = i * 10;
    }

    Tensor* t2 = tensor_clone(t1);
    ASSERT_BOOL(t2 != NULL, 1);
    ASSERT_EQUAL(t2->order, t1->order);
    ASSERT_EQUAL(t2->size, t1->size);
    ASSERT_EQUAL(t2->dtype, t1->dtype);

    int* data2 = (int*)t2->data;
    for (int i = 0; i < 6; i++) {
        ASSERT_EQUAL(data1[i], data2[i]);
    }

    arena_free();
}

// Tensor Binary Operations Tests 
void test_tensor_add_int() {
    printf("\n=== Testing tensor_add_int ===\n");
    arena_init(10000);

    size_t shape[] = {2, 2};
    Tensor* t1 = tensor_create(shape, 2, 2, DT_INT);
    Tensor* t2 = tensor_create(shape, 2, 2, DT_INT);

    int* data1 = (int*)t1->data;
    int* data2 = (int*)t2->data;
    data1[0] = 1; data1[1] = 2; data1[2] = 3; data1[3] = 4;
    data2[0] = 5; data2[1] = 6; data2[2] = 7; data2[3] = 8;

    Tensor* result = tensor_sum_int(t1, t2);
    ASSERT_BOOL(result != NULL, 1);

    int* r_data = (int*)result->data;
    ASSERT_EQUAL(r_data[0], 6);
    ASSERT_EQUAL(r_data[1], 8);
    ASSERT_EQUAL(r_data[2], 10);
    ASSERT_EQUAL(r_data[3], 12);

    arena_free();
}

void test_tensor_subtract_float() {
    printf("\n=== Testing tensor_subtract_float ===\n");
    arena_init(10000);

    size_t shape[] = {3};
    Tensor* t1 = tensor_create(shape, 1, 2, DT_FLOAT);
    Tensor* t2 = tensor_create(shape, 1, 2, DT_FLOAT);

    float* data1 = (float*)t1->data;
    float* data2 = (float*)t2->data;
    data1[0] = 10.0f; data1[1] = 20.0f; data1[2] = 30.0f;
    data2[0] = 3.0f; data2[1] = 5.0f; data2[2] = 7.0f;

    Tensor* result = tensor_subtract_float(t1, t2);
    ASSERT_BOOL(result != NULL, 1);

    float* r_data = (float*)result->data;
    ASSERT_BOOL(fabs(r_data[0] - 7.0f) < 0.001f, 1);
    ASSERT_BOOL(fabs(r_data[1] - 15.0f) < 0.001f, 1);
    ASSERT_BOOL(fabs(r_data[2] - 23.0f) < 0.001f, 1);

    arena_free();
}

void test_tensor_multiply_double() {
    printf("\n=== Testing tensor_multiply_double ===\n");
    arena_init(10000);

    size_t shape[] = {2, 2};
    Tensor* t1 = tensor_create(shape, 2, 2, DT_DOUBLE);
    Tensor* t2 = tensor_create(shape, 2, 2, DT_DOUBLE);

    double* data1 = (double*)t1->data;
    double* data2 = (double*)t2->data;
    data1[0] = 2.0; data1[1] = 3.0; data1[2] = 4.0; data1[3] = 5.0;
    data2[0] = 1.5; data2[1] = 2.0; data2[2] = 2.5; data2[3] = 3.0;

    Tensor* result = tensor_multiply_double(t1, t2);
    ASSERT_BOOL(result != NULL, 1);

    double* r_data = (double*)result->data;
    ASSERT_BOOL(fabs(r_data[0] - 3.0) < 0.001, 1);
    ASSERT_BOOL(fabs(r_data[1] - 6.0) < 0.001, 1);
    ASSERT_BOOL(fabs(r_data[2] - 10.0) < 0.001, 1);
    ASSERT_BOOL(fabs(r_data[3] - 15.0) < 0.001, 1);

    arena_free();
}

void test_tensor_divide_int() {
    printf("\n=== Testing tensor_divide_int ===\n");
    arena_init(10000);

    size_t shape[] = {4};
    Tensor* t1 = tensor_create(shape, 1, 2, DT_INT);
    Tensor* t2 = tensor_create(shape, 1, 2, DT_INT);

    int* data1 = (int*)t1->data;
    int* data2 = (int*)t2->data;
    data1[0] = 10; data1[1] = 20; data1[2] = 30; data1[3] = 40;
    data2[0] = 2; data2[1] = 4; data2[2] = 5; data2[3] = 8;

    Tensor* result = tensor_divide_int(t1, t2);
    ASSERT_BOOL(result != NULL, 1);

    int* r_data = (int*)result->data;
    ASSERT_EQUAL(r_data[0], 5);
    ASSERT_EQUAL(r_data[1], 5);
    ASSERT_EQUAL(r_data[2], 6);
    ASSERT_EQUAL(r_data[3], 5);

    arena_free();
}

void test_tensor_broadcasting() {
    printf("\n=== Testing tensor broadcasting ===\n");
    arena_init(10000);

    // Test scalar broadcasting
    size_t shape1[] = {1};
    size_t shape2[] = {3, 3};
    Tensor* t1 = tensor_create(shape1, 1, 2, DT_INT);
    Tensor* t2 = tensor_create(shape2, 2, 2, DT_INT);

    ((int*)t1->data)[0] = 5;
    int* data2 = (int*)t2->data;
    for (int i = 0; i < 9; i++) {
        data2[i] = i + 1;
    }

    Tensor* result = tensor_sum_int(t1, t2);
    ASSERT_BOOL(result != NULL, 1);
    ASSERT_EQUAL(result->order, 2);
    ASSERT_EQUAL(result->shape[0], 3);
    ASSERT_EQUAL(result->shape[1], 3);

    int* r_data = (int*)result->data;
    ASSERT_EQUAL(r_data[0], 6);  // 5 + 1
    ASSERT_EQUAL(r_data[4], 10); // 5 + 5

    arena_free();
}

// Tensor Unary Operations Tests 
void test_tensor_negation() {
    printf("\n=== Testing tensor_negation ===\n");
    arena_init(10000);

    size_t shape[] = {2, 2};
    Tensor* t = tensor_create(shape, 2, 2, DT_INT);
    int* data = (int*)t->data;
    data[0] = 1; data[1] = -2; data[2] = 3; data[3] = -4;

    int result = tensor_negation(t);
    ASSERT_EQUAL(result, 0);

    ASSERT_EQUAL(data[0], -1);
    ASSERT_EQUAL(data[1], 2);
    ASSERT_EQUAL(data[2], -3);
    ASSERT_EQUAL(data[3], 4);
 
    arena_free();
}

void test_tensor_abs() {
    printf("\n=== Testing tensor_abs ===\n");
    arena_init(10000);

    size_t shape[] = {4};
    Tensor* t = tensor_create(shape, 1, 2, DT_INT);
    int* data = (int*)t->data;
    data[0] = -5; data[1] = 10; data[2] = -15; data[3] = 20;

    int result = tensor_abs(t);
    ASSERT_EQUAL(result, 0);

    ASSERT_EQUAL(data[0], 5);
    ASSERT_EQUAL(data[1], 10);
    ASSERT_EQUAL(data[2], 15);
    ASSERT_EQUAL(data[3], 20);

    arena_free();
}

void test_tensor_exp() {
    printf("\n=== Testing tensor_exp ===\n");
    arena_init(10000);

    size_t shape[] = {3};
    Tensor* t = tensor_create(shape, 1, 2, DT_FLOAT);
    float* data = (float*)t->data;
    data[0] = 0.0f; data[1] = 1.0f; data[2] = 2.0f;

    int result = tensor_exp(t);
    ASSERT_EQUAL(result, 0);

    ASSERT_BOOL(fabs(data[0] - 1.0f) < 0.001f, 1);  // exp(0) = 1
    ASSERT_BOOL(fabs(data[1] - 2.718f) < 0.1f, 1);  // exp(1) ≈ e
    ASSERT_BOOL(fabs(data[2] - 7.389f) < 0.1f, 1);  // exp(2) ≈ e^2

    arena_free();
}

// Tensor Manipulation Tests 
void test_tensor_reshape() {
    printf("\n=== Testing tensor_reshape ===\n");
    arena_init(10000);

    size_t shape1[] = {2, 3};
    Tensor* t = tensor_create(shape1, 2, 2, DT_INT);

    size_t shape2[] = {3, 2};
    int result = tensor_reshape(t, shape2, 2);
    ASSERT_EQUAL(result, 0);
    ASSERT_EQUAL(t->order, 2);
    ASSERT_EQUAL(t->shape[0], 3);
    ASSERT_EQUAL(t->shape[1], 2);

    size_t shape3[] = {6};
    result = tensor_reshape(t, shape3, 1);
    ASSERT_EQUAL(result, 0);
    ASSERT_EQUAL(t->order, 1);
    ASSERT_EQUAL(t->shape[0], 6);

    // Test invalid reshape
    size_t shape4[] = {5};
    result = tensor_reshape(t, shape4, 1);
    ASSERT_EQUAL(result, -1);

    arena_free();
}

void test_tensor_transpose() {
    printf("\n=== Testing tensor_transpose ===\n");
    arena_init(10000);

    size_t shape[] = {2, 3};
    Tensor* t = tensor_create(shape, 2, 2, DT_INT);

    int result = tensor_transpose(t, 0, 1);
    ASSERT_EQUAL(result, 0);
    ASSERT_EQUAL(t->shape[0], 3);
    ASSERT_EQUAL(t->shape[1], 2);

    // Test invalid transpose
    result = tensor_transpose(t, 0, 5);
    ASSERT_EQUAL(result, -1);

    arena_free();
}

void test_tensor_permute() {
    printf("\n=== Testing tensor_permute ===\n");
    arena_init(10000);

    size_t shape[] = {2, 3, 4};
    Tensor* t = tensor_create(shape, 3, 2, DT_INT);

    size_t permute[] = {2, 0, 1};
    int result = tensor_permute(t, permute, 3);
    ASSERT_EQUAL(result, 0);
    ASSERT_EQUAL(t->shape[0], 4);
    ASSERT_EQUAL(t->shape[1], 2);
    ASSERT_EQUAL(t->shape[2], 3);

    // Test invalid permute
    size_t permute2[] = {0, 1, 5};
    result = tensor_permute(t, permute2, 3);
    ASSERT_EQUAL(result, -1);

    arena_free();
}

void test_tensor_squeeze() {
    printf("\n=== Testing tensor_squeeze ===\n");
    arena_init(10000);

    size_t shape[] = {1, 3, 1, 4, 1};
    Tensor* t = tensor_create(shape, 5, 2, DT_INT);

    int result = tensor_squeeze(t);
    ASSERT_EQUAL(result, 0);
    ASSERT_EQUAL(t->order, 2);
    ASSERT_EQUAL(t->shape[0], 3);
    ASSERT_EQUAL(t->shape[1], 4);

    arena_free();
}

void test_tensor_squeeze_at() {
    printf("\n=== Testing tensor_squeeze_at ===\n");
    arena_init(10000);

    size_t shape[] = {1, 3, 1, 4};
    Tensor* t = tensor_create(shape, 4, 2, DT_INT);

    int result = tensor_squeeze_at(t, 0);
    ASSERT_EQUAL(result, 0);
    ASSERT_EQUAL(t->order, 3);
    ASSERT_EQUAL(t->shape[0], 3);

    // Test invalid squeeze
    result = tensor_squeeze_at(t, 0);
    ASSERT_EQUAL(result, -1);

    arena_free();
}

void test_tensor_unsqueeze() {
    printf("\n=== Testing tensor_unsqueeze ===\n");
    arena_init(10000);

    size_t shape[] = {3, 4};
    Tensor* t = tensor_create(shape, 2, 2, DT_INT);

    int result = tensor_unsqueeze(t, 0);
    ASSERT_EQUAL(result, 0);
    ASSERT_EQUAL(t->order, 3);
    ASSERT_EQUAL(t->shape[0], 1);
    ASSERT_EQUAL(t->shape[1], 3);
    ASSERT_EQUAL(t->shape[2], 4);

    result = tensor_unsqueeze(t, 3);
    ASSERT_EQUAL(result, 0);
    ASSERT_EQUAL(t->order, 4);
    ASSERT_EQUAL(t->shape[3], 1);

    // Test invalid unsqueeze
    result = tensor_unsqueeze(t, 10);
    ASSERT_EQUAL(result, -1);

    arena_free();
}

void test_tensor_flatten() {
    printf("\n=== Testing tensor_flatten ===\n");
    arena_init(10000);

    size_t shape[] = {2, 3, 4};
    Tensor* t = tensor_create(shape, 3, 2, DT_INT);

    int result = tensor_flatten(t);
    ASSERT_EQUAL(result, 0);
    ASSERT_EQUAL(t->order, 1);
    ASSERT_EQUAL(t->shape[0], 24);
    ASSERT_EQUAL(t->stride[0], 1);

    arena_free();
}

// Tensor Matrix Multiplication Tests
void test_tensor_matmul() {
    printf("\n=== Testing tensor_matmul ===\n");
    arena_init(10000);

    size_t shape1[] = {2, 3};
    size_t shape2[] = {3, 2};
    Tensor* t1 = tensor_create(shape1, 2, 2, DT_FLOAT);
    Tensor* t2 = tensor_create(shape2, 2, 2, DT_FLOAT);

    float* data1 = (float*)t1->data;
    float* data2 = (float*)t2->data;

    // t1 = [[1, 2, 3], [4, 5, 6]]
    data1[0] = 1.0f; data1[1] = 2.0f; data1[2] = 3.0f;
    data1[3] = 4.0f; data1[4] = 5.0f; data1[5] = 6.0f;

    // t2 = [[1, 2], [3, 4], [5, 6]]
    data2[0] = 1.0f; data2[1] = 2.0f;
    data2[2] = 3.0f; data2[3] = 4.0f;
    data2[4] = 5.0f; data2[5] = 6.0f;

    Tensor* result = tensor_matmul(t1, t2);
    ASSERT_BOOL(result != NULL, 1);
    ASSERT_EQUAL(result->order, 2);
    ASSERT_EQUAL(result->shape[0], 2);
    ASSERT_EQUAL(result->shape[1], 2);

    float* r_data = (float*)result->data;
    // Result should be [[22, 28], [49, 64]]
    ASSERT_BOOL(fabs(r_data[0] - 22.0f) < 0.1f, 1);
    ASSERT_BOOL(fabs(r_data[1] - 28.0f) < 0.1f, 1);
    ASSERT_BOOL(fabs(r_data[2] - 49.0f) < 0.1f, 1);
    ASSERT_BOOL(fabs(r_data[3] - 64.0f) < 0.1f, 1);

    // Test invalid matmul
    size_t shape3[] = {2, 3};
    Tensor* t3 = tensor_create(shape3, 2, 2, DT_FLOAT);
    Tensor* result2 = tensor_matmul(t1, t3);
    ASSERT_BOOL(result2 == NULL, 1);

    arena_free();
}

// Tensor Fill Random Tests 
void test_tensor_fill_random() {
    printf("\n=== Testing tensor_fill_random ===\n");
    arena_init(10000);
    xoshiro_seed(12345);

    size_t shape[] = {3, 3};
    Tensor* t = tensor_create(shape, 2, 2, DT_FLOAT);

    int result = tensor_fill_random(t);
    ASSERT_EQUAL(result, 0);

    float* data = (float*)t->data;
    // Check that values are filled (not all zeros)
    int all_zero = 1;
    for (int i = 0; i < 9; i++) {
        if (data[i] != 0.0f) {
            all_zero = 0;
            break;
        }
    }
    ASSERT_BOOL(all_zero == 0, 1);

    arena_free();
}

// Utility Tests
void test_get_dtype_size() {
    printf("\n=== Testing get_dtype_size ===\n");
    ASSERT_EQUAL(get_dtype_size(DT_INT), sizeof(int));
    ASSERT_EQUAL(get_dtype_size(DT_FLOAT), sizeof(float));
    ASSERT_EQUAL(get_dtype_size(DT_DOUBLE), sizeof(double));
}

int main() {
    // Arena tests
    TEST_FUNC(test_arena_init);
    TEST_FUNC(test_arena_alloc);
    TEST_FUNC(test_arena_free);

    // Utility tests
    TEST_FUNC(test_get_dtype_size);

    // Tensor creation tests
    TEST_FUNC(test_tensor_create);
    TEST_FUNC(test_scalar_tensor);
    TEST_FUNC(test_tensor_clone);

    // Tensor binary operation tests
    TEST_FUNC(test_tensor_add_int);
    TEST_FUNC(test_tensor_subtract_float);
    TEST_FUNC(test_tensor_multiply_double);
    TEST_FUNC(test_tensor_divide_int);
    TEST_FUNC(test_tensor_broadcasting);

    // Tensor unary operation tests
    TEST_FUNC(test_tensor_negation);
    TEST_FUNC(test_tensor_abs);
    TEST_FUNC(test_tensor_exp);

    // Tensor manipulation tests
    TEST_FUNC(test_tensor_reshape);
    TEST_FUNC(test_tensor_transpose);
    TEST_FUNC(test_tensor_permute);
    TEST_FUNC(test_tensor_squeeze);
    TEST_FUNC(test_tensor_squeeze_at);
    TEST_FUNC(test_tensor_unsqueeze);
    TEST_FUNC(test_tensor_flatten);

    // Tensor matrix multiplication tests
    TEST_FUNC(test_tensor_matmul);

    // Tensor fill random tests
    TEST_FUNC(test_tensor_fill_random);
    return 0;
}
