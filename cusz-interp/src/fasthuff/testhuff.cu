/**
 * @file testhuff.cu
 * @author Cody Rivera
 * @brief
 * @version 0.0
 * @date 2022-04-13
 * (created) 2022-04-13

 * @copyright (C) 2022 by Washington State University, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <iostream>
#include <fstream>
#include <limits>
#include <string>
#include <cstdlib>
#include <sys/stat.h>
#include "common.hh"
#include "utils.hh"
#include "huffman_coarse.cuh"
#include "huffman_parbook.cuh"
#include "../../include/fasthuff/fasthuff.h"

using UInt64 = unsigned long long;

size_t file_size(std::string filename) {
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return (size_t) (rc == 0 ? stat_buf.st_size : 0);
}

void huff_compress(uint8_t* d_in_symbols, int n_syms, size_t len, uint8_t** d_out_codewords, size_t* out_bytes) {
    using namespace cusz;
    using T = uint8_t;
    using H = uint32_t;
    using M = uint32_t;
    
    int num_syms = n_syms;

    Capsule<T> in_symbols(len, "Input symbols");
    Capsule<H> codebook(num_syms, "Codebook");
    Capsule<uint8_t> revbook(HuffmanCoarse<T, H>::get_revbook_nbyte(num_syms), "Reverse codebook");

    in_symbols.template alloc<cusz::LOC::HOST_DEVICE>();
    codebook.template alloc<cusz::LOC::DEVICE>();
    revbook.template alloc<cusz::LOC::HOST_DEVICE>();

    in_symbols.template set<cusz::LOC::DEVICE>(d_in_symbols);

    HuffmanCoarse<T, H> codec;
    int sublen, pardeg;
    AutoconfigHelper::autotune(len, sublen, pardeg);
    codec.allocate_workspace(len, num_syms, pardeg);

    uint8_t* d_out;
    size_t out_len;
    codec.encode(
        in_symbols.template get<cusz::LOC::DEVICE>(),
        len,
        num_syms,
        sublen,
        pardeg,
        d_out,
        out_len
    );
    // Allocate GPU memory for d_out_codewords
    uint8_t* temp_out;
    cudaMalloc((void**)&temp_out, out_len);
    
    // Copy the compressed data to the newly allocated memory
    cudaMemcpy(temp_out, d_out, out_len, cudaMemcpyDeviceToDevice);
    
    // Set the output pointer to point to the newly allocated memory
    *d_out_codewords = temp_out;
    *out_bytes = out_len;

    in_symbols.template free<cusz::LOC::HOST_DEVICE>();
    codebook.template free<cusz::LOC::DEVICE>();
    revbook.template free<cusz::LOC::HOST_DEVICE>();
}

void huff_decompress(uint8_t* d_in_codewords, size_t in_len, uint8_t* d_out_symbols, size_t len) {
    using namespace cusz;
    using T = uint8_t;
    using H = uint32_t;
    using M = uint32_t;
    
    Capsule<uint8_t> in_codewords(in_len, "Input codewords");
    Capsule<T> out_symbols(len, "Output symbols");

    in_codewords.template alloc<cusz::LOC::DEVICE>();
    in_codewords.template set<cusz::LOC::DEVICE>(d_in_codewords);

    out_symbols.template alloc<cusz::LOC::DEVICE>();

    HuffmanCoarse<T, H> codec;
    codec.decode(
        in_codewords.template get<cusz::LOC::DEVICE>(),
        out_symbols.template get<cusz::LOC::DEVICE>()
    );

    cudaMemcpy(d_out_symbols, out_symbols.template get<cusz::LOC::DEVICE>(), len, cudaMemcpyDeviceToHost);

    in_codewords.template free<cusz::LOC::DEVICE>();
    out_symbols.template free<cusz::LOC::DEVICE>();
}
