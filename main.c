#include "aqua.h"

int main() {
    size_t shape[2] = {2, 1};
    Tensor* t1 = tensor_create(shape, 2);
    t1->data[0] = 1.4f;

    Tensor* t2 = scalar_tensor(4.9f);
    Tensor* t3 = tensor_sum(t1, t2);

    tensor_print(t1);
    tensor_print(t2);
    tensor_print(t3);
    Tensor* copied = tensor_clone(t2);
    tensor_print(copied);

    tensor_free(t1);
    tensor_free(t2);
    tensor_free(t3);
    return 0;
}

