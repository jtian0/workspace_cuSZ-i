import sys
import subprocess as sp
import os

def setup(ver_cuda="12"):
    print(f"Setting up for CUDA {ver_cuda}...")

    ver_nvcomp = "3.0.5"
    ver_nvcomp_ = ver_nvcomp.replace(".", "")

    nvcomp_dir = f"nvcomp{ver_nvcomp}-cuda{ver_cuda}"
    nvcomp_tgz = f"nvcomp_{ver_nvcomp}_x86_64_{ver_cuda}.x.tgz"
    proj_dir = os.getcwd()

    cmd_setup = f"""if [ ! -d {nvcomp_dir} ]; then 
    mkdir {nvcomp_dir} 
fi
cd {nvcomp_dir}
if [ ! -f {nvcomp_tgz} ]; then 
    wget https://developer.download.nvidia.com/compute/nvcomp/{ver_nvcomp}/local_installers/{nvcomp_tgz} 
fi
tar zxvf {nvcomp_tgz}
cd ..
nvcc -L{proj_dir}/{nvcomp_dir}/lib -I{proj_dir}/{nvcomp_dir}/include/ bitcomp_example.cu -lcuda -lnvcomp_bitcomp -o bitcomp_example"""

    # print(cmd_all)
    sp.check_call(cmd_setup, shell=True, stdout=sp.DEVNULL)


if __name__ == "__main__":
    #print(len(sys.argv))
    if len(sys.argv) < 2:
        print("help: `python setup-nvcomp.py 12` for CUDA 12, or `11` in place of the last argument for CUDA 11.")
    else:
        setup(sys.argv[-1])
