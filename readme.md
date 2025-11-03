# Fate (Fast Tensors)

**Fate** is a lightweight C library for fast tensor operations, providing multi-dimensional arrays (tensors) with support for various data types, broadcasting, and common mathematical operations. It is designed for high-performance numeric computations in a simple and flexible interface.

## Features

- Multi-dimensional tensors with dynamic shapes and strides
- Supports `int`, `float`, and `double` data types
- Random tensor initialization
- Scalar tensors for easy numeric operations
- Element-wise unary operations (`negation`, `abs`, `exp`)
- Element-wise binary operations with broadcasting support
- Matrix multiplication
- Tensor manipulation: `reshape`, `transpose`, `squeeze`, `unsqueeze`, `flatten`
- Utilities for printing tensor shapes, strides, and contents
- Memory management functions for tensors

## Installation

No special installation required. Include the `aqua.h` header and link your source code with the implementation.

```c
#include "aqua.h"

