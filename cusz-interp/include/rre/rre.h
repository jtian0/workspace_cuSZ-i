#pragma once

void RRE2_COMPRESS(uint32_t* input, size_t insize, uint8_t** output, size_t* outsize, float* time);
void RRE2_DECOMPRESS(uint8_t* input, void** output, float* time);