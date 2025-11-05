
#include "hashset.h"
#include <string.h>

unsigned int MurmurHash2A(const void *key, int len, unsigned int seed){
	const unsigned int m = 0x5bd1e995;
	const int r = 24;
	unsigned int l = len;

	const unsigned char *data = (const unsigned char *)key;

	unsigned int h = seed;
	unsigned int k;

	while (len >= 4){
		k = *(unsigned int*)data;
		mmix(h, k);
		data += 4;
		len -= 4;
	}

	unsigned int t = 0;

	switch (len)
	{
	case 3: t ^= data[2] << 16;
	case 2: t ^= data[1] << 8;
	case 1: t ^= data[0];
	}

	mmix(h, t);
	mmix(h, l);

	h ^= h >> 13;
	h *= m;
	h ^= h >> 15;

	return h;
}

HashSet* hashset_create(){
	HashSet* h = malloc(sizeof(HashSet));
	if(!h){
		return NULL;
	}
	for (int i = 0; i < HASH_SET_SIZE; i++){
		h->array[i] = NULL;
	}
	return h;
}

int hashset_add(HashSet* h, char* k, void* v) {
	unsigned int hash = MurmurHash2A((void*)k, strlen(k), FIXED_HASH_SEED);
	unsigned int idx = hash % HASH_SET_SIZE;

	Item* new_item = (Item*)malloc(sizeof(Item));
	if (!new_item){
		return -1;
	}

	new_item->k = strdup(k);
	new_item->v = v;
	new_item->next = NULL;

	// check if the key already exists — update if found
	Item* current = h->array[idx];
	while (current) {
		if (strcmp(current->k, k) == 0) {
		    current->v = v;
		    free(new_item->k);
		    free(new_item);
		    return 0;
		}
		current = current->next;
	}

	// insert at head (seems faster)
	new_item->next = h->array[idx];
	h->array[idx] = new_item;

	return 0;
}

void* hashset_get(HashSet* h, char* k) {
	unsigned int hash = MurmurHash2A((void*)k, strlen(k), FIXED_HASH_SEED);
	unsigned int idx = hash % HASH_SET_SIZE;

	Item* temp = h->array[idx];
	while (temp != NULL) {
		if (strcmp(temp->k, k) == 0){
		    return temp->v;
		}
		temp = temp->next;
	}
	return NULL;
}

