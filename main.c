#include "aqua.h"
#include <stdio.h>

int main() {
    // Create a simple 2D tensor with shape [2, 3]
    size_t shape[2] = {2, 3};
    Tensor* t = tensor_create(shape, 2, DT_FLOAT);
    float* data = (float*)t->data;

    // Fill tensor with sequential values
    for (size_t i = 0; i < 6; i++) {
        data[i] = (float)(i + 1);
    }

    printf("Original tensor (shape [2, 3]):\n");
    tensor_print(t);

    // Unsqueeze (add new dimension at the front)
    tensor_unsqueeze(t);

    printf("\nAfter unsqueeze (expected shape [1, 2, 3]):\n");
    tensor_print(t);

    // Print shape and stride arrays for verification
    printf("\nNew shape: ");
    for (size_t i = 0; i < t->order; i++) {
        printf("%zu ", t->shape[i]);
    }
    printf("\nNew stride: ");
    for (size_t i = 0; i < t->order; i++) {
        printf("%zu ", t->stride[i]);
    }
    printf("\n");

    // Cleanup
    tensor_free(t);
    return 0;
}
