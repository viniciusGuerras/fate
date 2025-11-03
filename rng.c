/*
 * random.c
 * Implements Random Number Generators
 * Current implementations: SplitMix64 and Xoshiro
 * Author: Vinicius Guerra
 * Start-Date: 2025-10-29
 */

#include "rng.h"

static uint64_t splitmix64_s;
static uint64_t xoshiro_s[4];

/*------- SplitMix64 -------*/

/* 
 * splitmix64_next_int - Generates the next 64-bit pseudorandom integer.
 * @sm Pointer to a SplitMix64 instance
 *
 * SplitMix64 uses bit mixing and large constants to produce
 * high-quality 64-bit pseudorandom numbers.
 * 
 * Returns: uint64_t Pseudorandom 64-bit integer
 */
uint64_t splitmix64_next_int(){
	uint64_t z = (splitmix64_s += 0x9e3779b97f4a7c15ULL);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
	z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
	return z ^ (z >> 31);
}

/* 
 * splitmix64_next_float - Generates a pseudorandom float in the range [0,1).
 *
 * Uses the top bits of next_int to produce a 24-bit precision float.
 * SplitMix64 uses bit mixing and large constants to produce
 * high-quality 64-bit pseudorandom numbers.
 * 
 * Returns: uint64_t Pseudorandom 64-bit integer
 */
float splitmix64_next_float(){
	return (splitmix64_next_int() >> 40) / (float)(1ULL << 24);
}

/* 
 * splitmix64_seed - Sets the state's seed
 *
 * Initializes the internal state array splitmix64_s using a 64-bit seed.
 */
void splitmix64_seed(uint64_t seed){
	splitmix64_s = seed;
}

/*------- Xoshiro256+ (xor, shift, rotate) -------*/

/* rotl - Rotates a 64 bit integer
 *
 * left side shifts the int's 64 bits by k to the left, 
 * right side shifts the int's 64 bits by (64 - k) 
 * after that the result go trough an "or" logic.
 * 
 */
static inline uint64_t xoshiro_rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

/*
 * xoshiro_next
 *
 * Generates the next 64-bit pseudorandom number using the xoshiro256** algorithm.
 *
 * Output calculation:
 *   - result = rotl(s[1] * 5, 7) * 9
 *     * Multiply s[1] by 5: spreads bits and ensures low bits are mixed.
 *     * Rotate left by 7: moves bits across positions for diffusion.
 *     * Multiply by 9: further scrambles bits for high-quality output.
 *
 * State update:
 *   - t = s[1] << 17
 *     * Temporary variable to mix s[1] into s[2].
 *   - XOR operations:
 *       s[2] ^= s[0]
 *       s[3] ^= s[1]
 *       s[1] ^= s[2]
 *       s[0] ^= s[3]
 *     * Mixes the four state variables together.
 *   - s[2] ^= t
 *     * Introduces the temporary value to further diffuse bits.
 *   - s[3] = rotl(s[3], 45)
 *     * Rotates s[3] to spread its bits across the state.
 *
 * Constants explanation:
 *   - 5, 9: Odd multipliers chosen to maximize bit mixing and randomness.
 *   - 7, 45: Carefully chosen rotation distances to avoid correlations.
 *   - 17: Shift distance for state diffusion.
 *
 * Notes:
 *   - The internal state array `xoshiro_s[0..3]` must be initialized with
 *     non-zero values. The algorithm has a period of 2^256-1.
 *   - This function is very fast and produces high-quality 64-bit pseudorandom numbers.
 */
uint64_t xoshiro_next(void) {
	const uint64_t result = xoshiro_rotl(xoshiro_s[1] * 5, 7) * 9;

	const uint64_t t = xoshiro_s[1] << 17;

	xoshiro_s[2] ^= xoshiro_s[0];
	xoshiro_s[3] ^= xoshiro_s[1];
	xoshiro_s[1] ^= xoshiro_s[2];
	xoshiro_s[0] ^= xoshiro_s[3];

	xoshiro_s[2] ^= t;

	xoshiro_s[3] = xoshiro_rotl(xoshiro_s[3], 45);

	return result;
}

/*
 * xoshiro_next_float
 *
 * Generates a pseudorandom float in the range [0, 1).
 *
 * Implementation:
 *   - Calls xoshiro_next() to get a 64-bit random integer.
 *   - Shifts right by 40 bits to keep the top 24 bits.
 *   - Divides by 2^24 (1 << 24) to normalize into [0, 1).
 *
 * Notes:
 *   - Produces 24-bit precision floats (single precision).
 */
float xoshiro_next_float(void){
	return (xoshiro_next() >> 40) / (float)(1ULL << 24);
}

/*
 * xoshiro_next_double
 *
 * Generates a pseudorandom double in the range [0, 1).
 *
 * Implementation:
 *   - Calls xoshiro_next() to get a 64-bit random integer.
 *   - Shifts right by 11 bits to keep the top 53 bits.
 *   - Multiplies by 1 / 2^53 (9007199254740992.0) to normalize into [0, 1).
 *
 * Notes:
 *   - Produces 53-bit precision doubles (double precision).
 */
double xoshiro_next_double(void){
	return (xoshiro_next() >> 11) * (1.0 / 9007199254740992.0);
}

/*
 * xoshiro_seed
 *
 * Initializes the internal state array xoshiro_s[0..3] using a 64-bit seed.
 *
 * Implementation:
 *   - Calls splitmix64_seed(seed) to initialize the splitmix64 generator.
 *   - Fills xoshiro_s[0..3] by calling splitmix64_next_int() four times.
 *
 * Notes:
 *   - All four state variables must be non-zero for proper operation.
 *   - Ensures the xoshiro256** generator has a full period of 2^256-1.
 */
void xoshiro_seed(uint64_t seed){
	splitmix64_seed(seed);
	for(int i = 0; i < 4; i++){
		xoshiro_s[i] = splitmix64_next_int();
	}
}
