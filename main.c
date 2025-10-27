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
    tensor_print_shape(t);
    tensor_print_stride(t);
    tensor_print(t);

    // Unsqueeze (add new dimension at the front)
    tensor_unsqueeze(t, 0);

    tensor_print_shape(t);
    tensor_print_stride(t);
    tensor_print(t);

    tensor_squeeze_at(t, 0);

    tensor_print_shape(t);
    tensor_print_stride(t);
    tensor_print(t);

    tensor_flatten(t);

    tensor_print_shape(t);
    tensor_print_stride(t);
    tensor_print(t);

    tensor_free(t);
    return 0;
}
