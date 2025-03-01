#pragma once

void RRE2_COMPRESS(uint32_t* input, size_t insize, uint8_t** output, size_t* outsize, size_t* rre2_padding_bytes, float* time);
void RRE2_DECOMPRESS(uint8_t* input, void** output, size_t rre2_padding_bytes, float* time);