#ifndef ARENA_H
#define ARENA_H

/*
 * arena.h
 * Implements a memory arena
 * Supports addition, requesting of elements and initalization/freeing of memory
 * Author: Vinicius Guerra
 * Start-Date: 2025-11-04
 */

#include "structures/hashset.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define MAX_SIZE 128

extern char*   global_init;
extern size_t  global_offset;

typedef struct {
	HashSet* hashset;
} RequestState;

typedef struct {
	char*    identifier;   // string
	char*    local_init;   // pointer to initial memory address 
	size_t   local_offset; // size it'll take on memory
} Request;

RequestState* arena_create();
int   arena_add(RequestState* rs, char* identifier, size_t memory_size);
void* arena_request(RequestState* rs, char* identifier);
int   arena_initialize(RequestState* rs);
int   arena_free();

#endif
