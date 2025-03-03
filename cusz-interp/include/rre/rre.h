#pragma once

void RRE1_COMPRESS(void* input, size_t insize, uint8_t** output, size_t* outsize, float* time);
void RRE1_DECOMPRESS(void* input, uint8_t** output, int size_before_rre1, float* time);