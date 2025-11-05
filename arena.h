#ifndef ARENA_H
#define ARENA_H

#include "structures/hashset.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define MAX_SIZE 128

extern char* global_init;
extern size_t  global_offset;

typedef struct {
	HashSet* hashset;
} RequestState;

typedef struct {
	char*    identifier;
	char*  local_init;
	size_t   local_offset;
} Request;

int arena_add(RequestState* rs, char* identifier, size_t memory_size);
void* arena_request(RequestState* rs, char* identifier);
int arena_initialize(RequestState* rs);
RequestState* arena_create();
int arena_free();

#endif
