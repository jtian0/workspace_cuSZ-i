# cuSZ-i Artifacts for SC '24

This repo contains the full artifacts of cuSZ-i work accepted in SC '24 conference.

## Preface
<details>
<summary>
Paper's Main Contributions
</summary>

- `C_1` We develop a GPU-optimized interpolation-based data predictor *G-Interp* with highly parallelized efficient interpolation, which can present excellent data prediction accuracy.
- `C_2` We design a lightweight interpolation auto-tuning kernel for GPU interpolation to optimize both the performance and compression quality of cuSZ-*i*.
- `C_3` We improve the implementation of GPU-based Huffman encoding and import a new lossless module to reduce its encoding redundancy further.
- `C_4` cuSZ-*i* improves compression ratio over other state-of-the-art GPU-based scientific lossy compressors by up to 476\% under the same error bound or PSNR. Meanwhile, it preserves a compression throughput of the same magnitude as other GPU compressors.

</details>

<details>
<summary>
Expected Results
</summary>
With the provided setup, the artifacts reproduce the experimental results reported in the paper, verifying cuSZ-*i*'s high compression ratio and quality and moderate throughput.
</details>

<details>
<summary>
Expected Reproduction Time
</summary>
Normally,

- The setup can be completed in 10 minutes.
- The executions should take 1 hour.
- The analysis can take a few minutes.

In case of the compatibility issue, we *alternatively* provide the `spack` installation instruction to replicate our tested environment:

- Please refer to the installation section for details.
- The \emph{alternative} Spack installation/deployment significantly increases the setup time to one hour due to building everything from the source code.
- The time for executions and the analysis remains unchanged.

</details>

## Artifact Setup

### Hardware

We require NVIDIA A100 GPU (40-GB, i.e., the common variant) to cover the essential functionality and, optionally, NVIDIA A40 GPU to cover the throughput scalability.

### Software

- We require an up-to-date mainstream Linux distro as the base environment.
  - e.g., CentOS 7 onward, Ubuntu 22.04.
- We require CUDA SDK of version 11.4 onward but lower than 12.5 (i.e., 11.4 to 12.4, inclusively).
  - corresponding to CUDA driver of version 470 onward.
  - CUDA 12.5 was tested not compatible.
- We require C++17-compliant host compiler.
  - e.g., GCC 9.3 onward.
- We require a modern cmake build system.
  - e.g., 3.18 onward. 


### Datasets/Inputs

The data setup will be done in setting up the workplace. 

<details>
<summary>
The details are folded here.
</summary>

- JHTDB 
  - Though hosted on https://turbulence.pha.jhu.edu/ as open data, it requires a token to access the data, which prohibits us from automating the data preprocessing. Thus, we don't include JHTDB datafields for the artifacts.
- Miranda, Nyx, QMCPack, S3D 
  - hosted on https://sdrbench.github.io
- RTM data are from proprietary simulations
  - which are not open to the public.
  - We exclude the use of RTM in this artifact.

</details>  

### Setup Compilers

To use `module-load` to setup the toolchain:

```bash
## Please change the version accordingly.
module load cuda/11.4
module load gcc/9.3
````

<details>
<summary>
Alternative Compiler Setup using Spack 
</summary>

```bash
cd $HOME
git clone -c feature.manyFiles=true \
https://github.com/spack/spack.git
## Now, initialize Spack on terminal start
## It is recommended to add the next line to
## "$HOME/.bashrc" or "$HOME/.zshrc"
. $HOME/spack/share/spack/setup-env.sh
## For other shells, please refer to the
## instruction by typing (quotes not included)
## "$HOME/spack/bin/spack load"
spack compiler find
spack install gcc@9.3.0
spack install cuda@12.4.4%gcc@9.3.0

spack load gcc@9.3.0 cuda@12.4.4
export LD_LIBRARY_PATH=$(dirname $(which nvcc))/../lib64:$LD_LIBRARY_PATH
```

</details>

### Setup: Workspace

```bash
## (1) get the artifacts repo
cd $HOME ## It can be anywhere.
git clone --recursive \
  https://github.com/jtian0/24_SC_artifacts.git \
  sc24cuszi
cd sc24cuszi

## (2) setup
## If you use CUDA 11
source setup-all.sh 11 <WHERE_TO_PUT_DATA_DIRS>
## If you use CUDA 12
# source setup-all.sh 12 <WHERE_TO_PUT_DATA_DIRS>

## (!!) clear build cache without removing data
bash setup-all.sh purge

## (3) prepare the data
bash setup-data.sh
```

## Artifact Execution

Navigate back to the workplace using `cd $WORKSPACE`. Then, run for each dataset.

```bash
## $DATAPATH is set in setup-all.sh

## Nyx
THIS_DATADIR=SDRBENCH-EXASKY-NYX-512x512x512
python script_data_collection.py  \
  --input ${DATAPATH}/${THIS_DATADIR} \
  --output $DATAPATH/${THIS_DATADIR}_log \
  --dims 512 512 512

## Miranda
THIS_DATADIR=SDRBENCH-Miranda-256x384x384
python script_data_collection.py  \
  --input ${DATAPATH}/${THIS_DATADIR} \
  --output $DATAPATH/${THIS_DATADIR}_log \
  --dims 384 384 256

## QMC
THIS_DATADIR=SDRBENCH-SDRBENCH-QMCPack
python script_data_collection.py  \
  --input ${DATAPATH}/${THIS_DATADIR} \
  --output $DATAPATH/${THIS_DATADIR}_log \
  --dims 69 69 33120

## S3D
THIS_DATRADIR=SDRBENCH-S3D
python script_data_collection.py  \
  --input ${DATAPATH}/${THIS_DATADIR} \
  --output $DATAPATH/${THIS_DATADIR}_log \
  --dims 500 500 500
```


## Artifact Analysis

```bash
## $DATAPATH is set in setup-all.sh

## Nyx
THIS_DATADIR=SDRBENCH-EXASKY-NYX-512x512x512
python script_data_analysis.py.py  \
  --input ${DATAPATH}/${THIS_DATADIR}_log \
  --output $DATAPATH/${THIS_DATADIR}_csv \
  --dims 512 512 512

## Miranda
THIS_DATADIR=SDRBENCH-Miranda-256x384x384
python script_data_analysis.py.py  \
  --input ${DATAPATH}/${THIS_DATADIR}_log \
  --output $DATAPATH/${THIS_DATADIR}_csv \
  --dims 384 384 256

## QMC
THIS_DATADIR=SDRBENCH-SDRBENCH-QMCPack
python script_data_analysis.py.py  \
  --input ${DATAPATH}/${THIS_DATADIR}_log \
  --output $DATAPATH/${THIS_DATADIR}_csv \
  --dims 69 69 33120

## S3D
THIS_DATRADIR=SDRBENCH-S3D
python script_data_analysis.py.py  \
  --input ${DATAPATH}/${THIS_DATADIR}_log \
  --output $DATAPATH/${THIS_DATADIR}_csv \
  --dims 500 500 500
```

<!-- ## Installation

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

We are also working on a substitution dictionary-like codec to propriatery bitcomp, which is supposed to enable better integration. -->
