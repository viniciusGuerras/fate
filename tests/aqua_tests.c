#include <stdio.h>
#include <assert.h>
#include "tensor.h"

void test_tensor_sum() {
    // Test 1: 2x1 tensor + scalar
    int shape1[2] = {2, 1};
    Tensor* t1 = tensor_create(shape1, 2);
    t1->data[0] = 4; t1->data[1] = 5;
    Tensor* t2 = scalar_tensor(3.0);
    Tensor* t3 = tensor_sum(t1, t2);
    assert(t3->data[0] == 7.0);
    assert(t3->data[1] == 8.0);
    tensor_free(t1); tensor_free(t2); tensor_free(t3);

    // Test 2: 2x2 tensor + 2x2 tensor
    int shape2[2] = {2, 2};
    Tensor* t4 = tensor_create(shape2, 2);
    Tensor* t5 = tensor_create(shape2, 2);
    t4->data[0] = 1; t4->data[1] = 2; t4->data[2] = 3; t4->data[3] = 4;
    t5->data[0] = 5; t5->data[1] = 6; t5->data[2] = 7; t5->data[3] = 8;
    Tensor* t6 = tensor_sum(t4, t5);
    assert(t6->data[0] == 6.0);
    assert(t6->data[1] == 8.0);
    assert(t6->data[2] == 10.0);
    assert(t6->data[3] == 12.0);
    tensor_free(t4); tensor_free(t5); tensor_free(t6);

    // Test 3: 3x1 tensor + scalar (negative numbers)
    int shape3[2] = {3, 1};
    Tensor* t7 = tensor_create(shape3, 2);
    t7->data[0] = -1; t7->data[1] = 0; t7->data[2] = 2;
    Tensor* t8 = scalar_tensor(5.0);
    Tensor* t9 = tensor_sum(t7, t8);
    assert(t9->data[0] == 4.0);
    assert(t9->data[1] == 5.0);
    assert(t9->data[2] == 7.0);
    tensor_free(t7); tensor_free(t8); tensor_free(t9);

    // Test 4: 1x0 tensor (empty tensor)
    int shape4[2] = {1, 0};
    Tensor* t10 = tensor_create(shape4, 2);
    Tensor* t11 = scalar_tensor(10.0);
    Tensor* t12 = tensor_sum(t10, t11);
    assert(t12->size == 0);  // Assuming Tensor has size field
    tensor_free(t10); tensor_free(t11); tensor_free(t12);

    printf("All tensor tests passed!\n");
}

int main() {
    test_tensor_sum();
    return 0;
}

