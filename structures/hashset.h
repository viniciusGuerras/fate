#ifndef HASHSET_H
#define HASHSET_H

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define HASH_SET_SIZE 128
#define FIXED_HASH_SEED 0x9747b28c
#define mmix(h,k) { k *= m; k ^= k >> r; k *= m; h *= m; h ^= k; }

typedef struct Item {
	char* k;
	void* v;
	struct Item* next;
} Item;

typedef struct HashSet {
    Item* array[HASH_SET_SIZE];
} HashSet;

unsigned int MurmurHash2A(const void *key, int len, unsigned int seed);
int   hashset_add(HashSet* h, char* k, void* v);
void* hashset_get(HashSet* h, char* k);
HashSet*  hashset_create();

#endif

