#include "aqua.h"
#include "test_utils.h"
#include <math.h>

void test_tensor_create_and_free() {
    size_t shape[2] = {2, 3};
    Tensor* t = tensor_create(shape, 2, 0, DT_DOUBLE);
    if (!t) {
        printf("[FAIL] tensor_create returned NULL\n");
        return;
    }
    ASSERT_BOOL((t->dtype == DT_DOUBLE), 1);
    ASSERT_BOOL((t->order == 2), 1);
    ASSERT_BOOL((t->shape[0] == 2), 1);
    ASSERT_BOOL((t->shape[1] == 3), 1);
    tensor_free(t);
}

void test_scalar_tensor() {
    ScalarType v;
    v.d = 42.0;
    Tensor* t = scalar_tensor(v, DT_DOUBLE, 0);
    if (!t) {
        printf("[FAIL] scalar_tensor returned NULL\n");
        return;
    }
    ASSERT_BOOL((t->order == 1), 1);
    ASSERT_BOOL((t->dtype == DT_DOUBLE), 1);
    double* data = (double*)t->data;
    ASSERT_BOOL((*data == 42.0), 1);
    tensor_free(t);
}

void test_tensor_clone() {
    size_t shape[2] = {2, 2};
    Tensor* t = tensor_create(shape, 2, 0, DT_INT);
    int* data = (int*)t->data;
    for (int i = 0; i < 4; i++) data[i] = i + 1;
    Tensor* c = tensor_clone(t);
    ASSERT_ARRAY_EQUAL((int*)t->data, (int*)c->data, 4);
    tensor_free(t);
    tensor_free(c);
}

 void test_tensor_sum_double() {
    size_t shape[2] = {2, 2};
    Tensor* t1 = tensor_create(shape, 2, 0, DT_DOUBLE);
    Tensor* t2 = tensor_create(shape, 2, 0, DT_DOUBLE);
    double* d1 = (double*)t1->data;
    double* d2 = (double*)t2->data;
    for (int i = 0; i < 4; i++) {
        d1[i] = i + 1;
        d2[i] = (i + 1) * 10;
    }
    Tensor* out = tensor_sum_double(t1, t2);
    double* result = (double*)out->data;
    double expected[4] = {11, 22, 33, 44};
    ASSERT_ARRAY_EQUAL(result, expected, 4);
    tensor_free(t1);
    tensor_free(t2);
    tensor_free(out);
}

 void test_tensor_broadcast_shape() {
    size_t a[2] = {2, 3};
    size_t b[1] = {3};
    size_t out_order = 0;
    size_t* out = broadcast_shape(a, 2, b, 1, &out_order);
    if (!out) {
        printf("[FAIL] broadcast_shape returned NULL\n");
        return;
    }
    ASSERT_BOOL((out_order == 2), 1);
    ASSERT_BOOL((out[0] == 2), 1);
    ASSERT_BOOL((out[1] == 3), 1);
    free(out);
}

void test_tensor_reshape_correct() {
    size_t shape[2] = {2, 3};
    Tensor* t = tensor_create(shape, 2, 2, DT_INT);
    size_t new_shape[3] = {3, 1, 2};
    int res = tensor_reshape(t, new_shape, 3);
    ASSERT_BOOL((t->order == 3), 1);
    ASSERT_EQUAL(res, CORRECT);
    tensor_free(t);
}

void test_tensor_reshape_fail() {
    size_t shape[2] = {2, 3};
    Tensor* t = tensor_create(shape, 2, 0, DT_INT);
    size_t new_shape[3] = {3, 1, 2};
    int res = tensor_reshape(t, new_shape, 3);
    ASSERT_BOOL((t->order == 3), 0);
    ASSERT_EQUAL(res, ERROR);
    tensor_free(t);
}

void test_tensor_squeeze_unsqueeze() {
    size_t shape[3] = {1, 2, 3};
    Tensor* t = tensor_create(shape, 3, 0, DT_FLOAT);
    ASSERT_BOOL(tensor_squeeze_at(t, 0), CORRECT);
    ASSERT_BOOL((t->order == 2), 1);
    ASSERT_BOOL(tensor_unsqueeze(t, 0), CORRECT);
    ASSERT_BOOL((t->order == 3), 1);
    tensor_free(t);
}

void test_tensor_flatten() {
    size_t shape[2] = {2, 3};
    Tensor* t = tensor_create(shape, 2, 0, DT_INT);
    ASSERT_BOOL(tensor_flatten(t), CORRECT);
    ASSERT_BOOL((t->order == 1), 1);
    ASSERT_BOOL((t->shape[0] == 6), 1);
    tensor_free(t);
}

void test_tensor_transpose() {
    size_t shape[2] = {2, 3};
    Tensor* t = tensor_create(shape, 2, 0, DT_INT);
    int* data = (int*)t->data;
    for (int i = 0; i < 6; i++) data[i] = i + 1;
    int res = tensor_transpose(t, 0, 1);
    ASSERT_EQUAL(res, 0);
    ASSERT_ARRAY_EQUAL(t->shape, ((size_t[]){3, 2}), 2);
    tensor_free(t);
}

void test_tensor_permute() {
    size_t shape[3] = {2, 3, 4};
    Tensor* t = tensor_create(shape, 3, 0, DT_INT);
    int* data = (int*)t->data;
    for (int i = 0; i < 24; i++) data[i] = i + 1;
    size_t perm[3] = {2, 0, 1};
    int res = tensor_permute(t, perm, 3);
    ASSERT_EQUAL(res, 0);
    ASSERT_ARRAY_EQUAL(t->shape, ((size_t[]){4, 2, 3}), 3);
    tensor_free(t);
}

void test_tensor_negation() {
    size_t shape_int[2] = {2, 2};
    Tensor* t_int = tensor_create(shape_int, 2, 0, DT_INT);
    int* data_int = (int*)t_int->data;
    for (int i = 0; i < 4; i++) data_int[i] = i + 1;
    ASSERT_EQUAL(tensor_negation(t_int), 0);
    int expected_int[4] = {-1, -2, -3, -4};
    ASSERT_ARRAY_EQUAL(data_int, expected_int, 4);
    tensor_free(t_int);
    size_t shape_float[2] = {2, 2};
    Tensor* t_float = tensor_create(shape_float, 2, 0, DT_FLOAT);
    float* data_float = (float*)t_float->data;
    for (int i = 0; i < 4; i++) data_float[i] = (float)(i + 1);
    ASSERT_EQUAL(tensor_negation(t_float), 0);
    float expected_float[4] = {-1.0f, -2.0f, -3.0f, -4.0f};
    for (int i = 0; i < 4; i++) ASSERT_BOOL(data_float[i] == expected_float[i], 1);
    tensor_free(t_float);
    size_t shape_double[2] = {2, 2};
    Tensor* t_double = tensor_create(shape_double, 2, 0, DT_DOUBLE);
    double* data_double = (double*)t_double->data;
    for (int i = 0; i < 4; i++) data_double[i] = (double)(i + 1);
    ASSERT_EQUAL(tensor_negation(t_double), 0);
    double expected_double[4] = {-1.0, -2.0, -3.0, -4.0};
    for (int i = 0; i < 4; i++) ASSERT_BOOL(data_double[i] == expected_double[i], 1);
    tensor_free(t_double);
    ASSERT_EQUAL(tensor_negation(NULL), -1);
}

int main() {
    printf("=== Running Aqua Tensor Tests ===\n");

    TEST_FUNC(test_tensor_create_and_free);
    TEST_FUNC(test_scalar_tensor);
    TEST_FUNC(test_tensor_clone);
    TEST_FUNC(test_tensor_sum_double);
    TEST_FUNC(test_tensor_broadcast_shape);
    TEST_FUNC(test_tensor_reshape_correct);
    TEST_FUNC(test_tensor_reshape_fail);
    TEST_FUNC(test_tensor_squeeze_unsqueeze);
    TEST_FUNC(test_tensor_flatten);
    TEST_FUNC(test_tensor_transpose);
    TEST_FUNC(test_tensor_permute);
    TEST_FUNC(test_tensor_negation);

    printf("=== All tests finished ===\n");
    return 0;
}

