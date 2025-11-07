#include "arena.h"

char*    global_init       = NULL;
size_t   global_offset     = 0;
size_t   global_limit;

int arena_init(size_t memory_size){
	global_init  = malloc(memory_size);
	global_limit = memory_size;

	if(!global_init){
		printf("ERROR: couldn't initialize area.\n");
		return -1;
	}

	global_offset = 0;
	return 0;
}

void* arena_alloc(size_t number, size_t memory_size){
	size_t total = number * memory_size;

	if((global_offset + total) > global_limit){
		printf("ERROR: not enough memory.\n");
		return NULL;
	}

	void* ptr = global_init + global_offset;
	global_offset += total;

	return ptr;
}

int arena_free(){
	if(!global_init){
		printf("ERROR: arena not initialized.\n");
		return -1;
	}

	free(global_init);
	global_init   = NULL;
	global_offset = 0;
	global_limit  = 0;

	return 0;
}

