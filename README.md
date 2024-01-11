# Bitcomp Lossless Example

## Introduction
Bitcomp is a powerful tool for lossless file compression and decompression. It efficiently compresses files, ensuring that your data is stored in a compact format without any loss of information.

## Installation

```bash
## python setup.py [CUDA/NVCC VERSION: 11 or 12]
## Can be identified using `nvcc --version`
python setup.py 12
```

Before running, we need to setup `LD_LIBRARY_PATH`


```bash
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH    
```

### Compression
To compress a file with Bitcomp, use the following command:

```bash
./bitcomp_example -c /path/to/file
```

This will compress the file located at `/path/to/file` using Bitcomp's lossless compression algorithm. The compressed file name is `/path/to/file.bitcomp`.

### Decompression
To decompress a file that has been compressed using Bitcomp, use the command:

```bash
./bitcomp_example -d /path/to/compressed/file
```

Here, `/path/to/compressed/file` is the path to the file that you want to decompress. The decompressed file name is `/path/to/compressed/file.decompressed`.

### Roundtrip Verification
For a roundtrip process (compress and then decompress a file), and to verify the integrity and correctness of the process, use:

```bash
./bitcomp_example -r /path/to/file
```

This command performs both compression and decompression on `/path/to/file`, allowing you to verify that the original file and the decompressed file are identical.

## Case Study with cuSZ-Interp

Please refer to the [artifact of cuSZ with interpolation](https://github.com/Meso272/cusz-I). Please also refer to [our arXiv'ed paper in submission](https://arxiv.org/pdf/2312.05492.pdf).

In the case study, we perform `cuszi` compression to have the scientific data encoded with Huffman codec. Then, the output of `cuszi` is the input of `bitcomp_example`. The final compress ratio is `CR-cusz` multiplied by `CR-bitcomp`. 

It is also worth noting that 1) the input size of `bitcomp` is sufficiently small, so that appending `bitcomp` will not noticeably decrease the end-to-end throughput; 2) while `cuszi` may not be the fastest in processing, its high compression ratio offers significant benefits in data transfer rates, making it a preferred choice in certain scientific applications with frequent data movements. 

We are also working on a substitution dictionary-like codec to propriatery bitcomp, which is supposed to enable better integration.