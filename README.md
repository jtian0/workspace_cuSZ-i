# Bitcomp Lossless Example

Bitcomp is a powerful tool for lossless file compression and decompression. It efficiently compresses files, ensuring that your data is stored in a compact format without any loss of information.

## Installation

```bash
git clone https://github.com/jtian0/bitcomp_lossless_example.git
cd bitcomp_lossless_example

## python setup.py [CUDA/NVCC VERSION: 11 or 12]
## Can be identified using `nvcc --version`
python setup.py 12
```

Before running, we need to setup `LD_LIBRARY_PATH`

```bash
# if it is `python setup.py 12`
export LD_LIBRARY_PATH=$(pwd)/nvcomp3.0.5-cuda12/lib:$LD_LIBRARY_PATH    
# if it is `python setup.py 11`
#export LD_LIBRARY_PATH=$(pwd)/nvcomp3.0.5-cuda11/lib:$LD_LIBRARY_PATH    
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

Please refer to the [artifact of cuSZ with interpolation](https://github.com/Meso272/cusz-I). Please also refer to [our arXiv'ed paper in submission](https://arxiv.org/pdf/2312.05492.pdf). The synopisis of setting up cuSZ-I is as follows. 
```bash
# In the desired root directory
git clone https://github.com/Meso272/cusz-I.git cusz-interp
cd cusz-interp && mkdir build && cd build

cmake .. \
    -DPSZ_BACKEND=cuda \
    -DPSZ_BUILD_EXAMPLES=on \
    -DCMAKE_CUDA_ARCHITECTURES="70;80;86" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_COLOR_DIAGNOSTICS=on \
#    -DCMAKE_INSTALL_PREFIX=[/path/to/install/dir]
make -j

# type cuszi (inside its build dir) for the quick help
```

In the case study, we perform `cuszi` compression to have the scientific data encoded with Huffman codec. Then, the output of `cuszi` is the input of `bitcomp_example`. The final compress ratio is `CR-cusz` multiplied by `CR-bitcomp`. 

```bash
## using default Spline predictor
cuszi -t f32 -m r2r -e [ErrorBound] -i [/PATH/TO/DATA] -l [X]x[Y]x[Z] -z --report time
cuszi -i [/PATH/TO/DATA].cusza -x --report time --compare [/PATH/TO/DATA]

## using Lorenzo predictor for comparison
cuszi -t f32 -m r2r -e [ErrorBound] -i [/PATH/TO/DATA] -l [X]x[Y]x[Z] -z --report time -- predictor lorenzo
cuszi -i [/PATH/TO/DATA].cusza -x --report time --compare [/PATH/TO/DATA]
```

- The output of `cuszi`-compress is `<original filname>.cusza`
- The output of `cuszi`-decompress is `<original filname>.cuszx`
- To see the complete pipeline demonstration (`cuszi` + `bitcomp`) 

  ```bash
  `bitcomp_example -c /path/to/<original filename>.cusza`
  ```



It is also worth noting that 1) the input size of `bitcomp` is sufficiently small, so that appending `bitcomp` will not noticeably decrease the end-to-end throughput; 2) while `cuszi` may not be the fastest in processing, its high compression ratio offers significant benefits in data transfer rates, making it a preferred choice in certain scientific applications with frequent data movements. 

We are also working on a substitution dictionary-like codec to propriatery bitcomp, which is supposed to enable better integration.
