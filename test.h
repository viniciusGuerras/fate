#ifndef TEST_H
#define TEST_H

/*
 * rng.h
 * Implements Random Number Generators 
 * Author: Vinicius Guerra
 * Start-Date: 2025-11-02
 */

#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#include "tensor.h"
#include "arena.h"
#include <stdio.h>
#include "utils.h"
#include <math.h>
#include "rng.h"

#define ERROR -1
#define CORRECT 0

// --- Function Test Macro --- //
#define TEST_FUNC(func, ...) do { \
    printf("[TESTING] %s()\n", #func); fflush(stdout); \
    func(__VA_ARGS__); \
    printf("------------\n"); fflush(stdout); \
} while(0)

// --- Boolean Expression Test --- //
#define ASSERT_BOOL(expression, expected) do { \
    int __test_result = (expression); \
    if (__test_result == (expected)) { \
        printf("[PASS] %s working as expected at line %d\n", #expression, __LINE__); \
    } else { \
        printf("[FAIL] %s got result: %d (expected %d) at line %d\n", #expression, __test_result, (expected), __LINE__); \
    } \
    fflush(stdout); \
} while(0)

// --- Array Equality Assertion --- //
#define ASSERT_ARRAY_EQUAL(a, b, n) do { \
    size_t __fail_index = (n); \
    for (size_t i = 0; i < (n); i++) { \
        if ((a)[i] != (b)[i]) { \
            printf("[FAIL] Arrays differ at index %zu: %d != %d (line %d)\n", i, (a)[i], (b)[i], __LINE__); \
            __fail_index = i; \
            break; \
        } \
    } \
    if (__fail_index == (n)) { \
        printf("[PASS] Arrays are equal (%zu elements)\n", (size_t)(n)); \
    } \
    fflush(stdout); \
} while(0)

// --- Equality Assertion --- //
#define ASSERT_EQUAL(a, b) do { \
    if ((a) == (b)) { \
        printf("[PASS] %s == %s (%d)\n", #a, #b, (int)(a)); \
    } else { \
        printf("[FAIL] %s (%d) != %s (%d) at line %d\n", #a, (int)(a), #b, (int)(b), __LINE__); \
    } \
    fflush(stdout); \
} while(0)

#endif 
