#ifndef ERRORS_H
#define ERRORS_H

/*
 * errors.h
 * Implements error codes
 * Author: Vinicius Guerra
 * Start-Date: 2025-11-16
 */

typedef enum {
    TENSOR_OK = 0,                // No error
    TENSOR_ERR_ALLOC,             // Memory allocation failed
    TENSOR_ERR_SHAPE,             // Invalid tensor shape
    TENSOR_ERR_INDEX,             // Index out of bounds
    TENSOR_ERR_DIM,               // Dimension mismatch
    TENSOR_ERR_RNG,               // RNG-related error
    TENSOR_ERR_INVALID_ARGS,      // Invalid arguments passed
    TENSOR_ERR_NOT_ENOUGH_MEMORY, // Not enough shape/stride memory
    TENSOR_ERR_UNKNOWN            // Catch-all
} TensorError;

typedef enum {
    ARENA_OK = 0,
    ARENA_ERR_NOT_INITIALIZED,
} ArenaError;

#endif // ERRORS_H

