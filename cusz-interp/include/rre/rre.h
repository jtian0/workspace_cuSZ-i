#pragma once

void DIFFLOG_COMPRESS(void* input, size_t insize, uint8_t** output, size_t* outsize, size_t* difflog_padding_bytes, float* time);
void DIFFLOG_DECOMPRESS(uint8_t* input, void** output, size_t difflog_padding_bytes, float* time);
