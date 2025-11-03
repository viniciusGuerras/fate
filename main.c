#include "aqua.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int main() {
	size_t shape_1[3] = {1, 2, 4};
	size_t shape_2[2] = {2, 4};
	xoshiro_seed(time(NULL));
	Tensor* t_1 = tensor_create(shape_1, 3t:!;, 0, DT_FLOAT);
	Tensor* t_2 = tensor_create(shape_2, 2, 0, DT_FLOAT);
	tensor_fill_random(t_1);
	tensor_fill_random(t_2);
	tensor_print(t_1);
	tensor_print(t_2);
	Tensor* r = tensor_sum_float(t_1, t_2);
	tensor_print(r);
	tensor_free(t_1);
	tensor_free(t_2);
	tensor_free(r);
}



