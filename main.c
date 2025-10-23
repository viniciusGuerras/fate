#include "aqua.h"

int main() {
    size_t shape[2] = {2, 1};
    Tensor* t1 = tensor_create(shape, 2);
    t1->data[0] = 1.4f;

    Tensor* t2 = scalar_tensor(4.9f);
    Tensor* t3 = tensor_sum(t1, t2);

    for (size_t i = 0; i < t3->size; i++) {
        printf("%f\n", t3->data[i]);
    }

    tensor_free(t1);
    tensor_free(t2);
    tensor_free(t3);
    return 0;
}

