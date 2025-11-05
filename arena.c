
#include "arena.h"

char* global_init = NULL;
size_t  global_offset = 0;

Request* allocate_list[MAX_SIZE];
size_t   allocate_list_pos = 0;
size_t   total_memory = 0;

RequestState* arena_create(){
	RequestState* state = (RequestState*)malloc(sizeof(RequestState));
	if(!state){
		printf("ERROR: failed allocating state.\n");
		return NULL;
	}
	state->hashset = hashset_create();
	return state;
}

int arena_add(RequestState* rs, char* identifier, size_t memory_size){
	Request* request = malloc(sizeof(Request));
	if(!request){
		printf("ERROR: failed allocating request.\n");
		return -1;
	}
	request->identifier      = strdup(identifier);
	request->local_init      = NULL;
	request->local_offset    = memory_size;

	allocate_list[allocate_list_pos++] = request;
	total_memory += memory_size;
	return 0;
}

int arena_initialize(RequestState* rs){
	global_init = malloc(total_memory);
	for(int i = 0; i < allocate_list_pos; i++){
		allocate_list[i]->local_init = global_init + global_offset;
		global_offset += allocate_list[i]->local_offset;
		hashset_add(rs->hashset, 
			    allocate_list[i]->identifier,
			    allocate_list[i]->local_init);
	}
	if(!global_init){
		printf("ERROR: couldn't initialize area.\n");
		return -1;
	}
	return 0;
}

void* arena_request(RequestState* rs, char* identifier){
	void* found = hashset_get(rs->hashset, identifier); 
	if(!found){
		printf("ERROR: identifier not found.\n");
		return NULL;
	}
	return found;
}

int arena_free(){
	if(!global_init){
		printf("ERROR: arena not initialized.\n");
		return -1;
	}
	free(global_init);
	return 0;
}

