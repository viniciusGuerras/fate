#ifndef ARENA_H
#define ARENA_H

/*
 * arena.h
 * Implements a arena (large block of memory)
 * Author: Vinicius Guerra
 * Start-Date: 2025-11-04
 */

#include <stdlib.h>
#include "errors.h"
#include <stdio.h>

// Global variables
extern char*   global_init;
extern size_t  global_offset;
extern size_t  global_limit;

int   arena_init(size_t memory_size);
void* arena_alloc(size_t number, size_t memory_size);
int   arena_free();

#endif
