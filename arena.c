#include "arena.h"

char*  global_init   = NULL; // Pointer to the start of the allocated memory block (the arena)
size_t global_offset = 0;    // Current offset (how much memory has been allocated so far)
size_t global_limit;         // Total size (limit) of the arena in bytes

// Initializes the arena with a given total memory size.
int arena_init(size_t memory_size){
	global_init  = malloc(memory_size); // Allocate a single large block of memory
	global_limit = memory_size;
	global_offset = 0;
	if(!global_init){
		return ARENA_ERR_NOT_INITIALIZED;
	}
	return ARENA_OK;
}

// Allocates a block of memory within the arena.
void* arena_alloc(size_t number, size_t memory_size){
	size_t total = number * memory_size;     // Total bytes requested
	if((global_offset + total) > global_limit){
		return NULL;
	}
	void* ptr = global_init + global_offset; // Get a pointer to the next free space in the arena
	global_offset += total;                  // Advance the offset for the next allocation
	return ptr;
}

// Frees the entire arena (all allocations at once).
int arena_free(){
	if(!global_init){
		return ARENA_ERR_NOT_INITIALIZED;
	}
	free(global_init);
	global_init   = NULL;
	global_offset = 0;
	global_limit  = 0;
	return ARENA_OK;
}

