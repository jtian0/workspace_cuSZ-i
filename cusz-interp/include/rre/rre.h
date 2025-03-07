#pragma once

void DIFFBIT_COMPRESS(void* input, size_t insize, uint8_t** output, size_t* outsize, size_t* diffbit_padding_bytes, float* time);
void DIFFBIT_DECOMPRESS(uint8_t* input, void** output, size_t diffbit_padding_bytes, float* time);
