import os
import sys
import copy
import argparse
import subprocess
from tqdm import tqdm
import pandas as pd


def compute_throughput(elapsed_time, data_size):
    return float(data_size[0]) * float(data_size[1]) * float(data_size[2]) * 4 / 1024.0/ 1024.0/ 1024.0 / (elapsed_time * 1e-9)


def update_command(cmp, data_path, data_size, error_bound="1e-2", bit_rate="2", cuszx_block_size=64, nsys_result_path="./nsys_result", proc_id=1):
    work_path = os.getenv('WORK_PATH')
    print(cmp, data_size[0], data_size[1], data_size[2] )
    try:
        nbEle = int(data_size[0]) * int(data_size[1]) * int(data_size[2])
    except:
        assert 0
    print(nbEle)
    # nsys
    if cmp == "FZGPU":
        cmd = [
                    ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "fz-gpu",
                    data_path, 
                    data_size[0], 
                    data_size[1], 
                    data_size[2], 
                    error_bound,
                    ],
                    ["compareData",
                    "-f",  data_path, data_path+'.fzgpux',],
                    ["rm",
                    data_path + '.fzgpua', data_path+'.fzgpux', data_path + '.fzgpua.bitcomp',data_path + '.fzgpua.bitcomp.decompressed',  f"./{nsys_result_path}/result.nsys-rep",  f"./{nsys_result_path}/result.sqlite"]]
    elif cmp == "cuSZ":
        cmd = [["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cusz-interp_24/build/cuszi", 
                    "-t", "f32",
                    "-m", "r2r",
                    "-i", data_path,
                    "-e", error_bound,
                    "-l", f"{data_size[0]}x{data_size[1]}x{data_size[2]}",
                    "-z", 
                    "--predictor", "lorenzo",
                    "--report", "time,cr",
                    "-a", "0",],
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cusz-interp_24/build/cuszi", 
                    "-i", data_path+".cusza",
                    "-x",
                    "--report", "time",
                    "--compare", data_path,
                    ],
                ["compareData",
                    "-f",  data_path, data_path+'.cuszx',],
                ["rm",
                    data_path + '.cusza', data_path+'.cuszx', data_path + '.cusza.bitcomp',data_path + '.cusza.bitcomp.decompressed',  f"./{nsys_result_path}/result.nsys-rep",  f"./{nsys_result_path}/result.sqlite",]]
    elif cmp == "cuSZ_24":
        cmd = [["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cusz-interp_24/build/cuszi", 
                    "-t", "f32",
                    "-m", "r2r",
                    "-i", data_path,
                    "-e", error_bound,
                    "-l", f"{data_size[0]}x{data_size[1]}x{data_size[2]}",
                    "-z", 
                    "--predictor", "lorenzo",
                    "--report", "time,cr",
                    "-a", "0",],
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cusz-interp_24/build/cuszi", 
                    "-i", data_path+".cusza",
                    "-x",
                    "--report", "time",
                    "--compare", data_path,
                    ],
                ["compareData",
                    "-f",  data_path, data_path+'.cuszx',],
                ["rm",
                    data_path + '.cusza', data_path+'.cuszx', data_path + '.cusza.bitcomp',data_path + '.cusza.bitcomp.decompressed',  f"./{nsys_result_path}/result.nsys-rep",  f"./{nsys_result_path}/result.sqlite",]]
    elif cmp == "cuSZp_outlier":
        cmd = [
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cuSZp", "-i",
                    data_path, "-t", "f32", "-m", "outlier",
                    # data_path, "-t", "f32", "-m", "outlier",
                    "-eb", "rel", error_bound, "-x", f"{data_path}.cuszpa", "-o", f"{data_path}.cuszpx"],
                ["compareData",
                    "-f",  data_path, data_path+'.cuszpx',],
                ["rm",
                    data_path + '.cuszpa', data_path+'.cuszpx', data_path + '.cuszpa.bitcomp',data_path + '.cuszpa.bitcomp.decompressed',  f"./{nsys_result_path}/result.nsys-rep",  f"./{nsys_result_path}/result.sqlite",]]
    elif cmp == "cuSZp_plain":
        cmd = [
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cuSZp", "-i",
                    data_path, "-t", "f32", "-m", "plain",
                    # data_path, "-t", "f32", "-m", "outlier",
                    "-eb", "rel", error_bound, "-x", f"{data_path}.cuszpa", "-o", f"{data_path}.cuszpx"],
                ["compareData",
                    "-f",  data_path, data_path+'.cuszpx',],
                ["rm",
                    data_path + '.cuszpa', data_path+'.cuszpx', data_path + '.cuszpa.bitcomp',data_path + '.cuszpa.bitcomp.decompressed',  f"./{nsys_result_path}/result.nsys-rep",  f"./{nsys_result_path}/result.sqlite",]]
    elif cmp == "cuzfp":
        cmd = [["nsys", "profile",  "--stats=true", "-o", nsys_result_path + bit_rate, "zfp",
                    "-i", data_path,
                    "-z", data_path+'.cuzfpa',
                    "-x", "cuda",
                    "-f", 
                    "-3", 
                    data_size[0], 
                    data_size[1], 
                    data_size[2], 
                    "-r", bit_rate],
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path + bit_rate, "zfp",
                    "-z", data_path+'.cuzfpa',
                    "-o", data_path+'.cuzfpx',
                    "-x", "cuda",
                    "-f", 
                    "-3", 
                    data_size[0], 
                    data_size[1], 
                    data_size[2], 
                    "-r", bit_rate],
                # ~/qcat-1.3-install/bin/compareData -f $DATA $DATA.cuszx
                ["compareData",
                    "-f",  data_path, data_path+'.cuzfpx',],
                ["rm",
                    data_path + '.cuzfpa', data_path+'.cuzfpx', data_path + '.cuzfpa.bitcomp',data_path + '.cuzfpa.bitcomp.decompressed'
                    ,  f"./{nsys_result_path}/result.nsys-rep",  f"./{nsys_result_path}/result.sqlite"]]
    elif cmp == "cuSZx":
        cmd = [["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "szx_testfloat_compress_fastmode2",
                    data_path, f"{cuszx_block_size}", error_bound, "--cuda"],
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "szx_testfloat_decompress_fastmode2",
                    data_path+".szx", f"{nbEle}", "--cuda"],
                ["compareData",
                    "-f",  data_path, data_path+'.szx.out',],
                ["rm",
                    data_path + '.szx', data_path+'.szx.out', data_path + '.szx.bitcomp',data_path + '.szx.bitcomp.decompressed'
                    ,  f"./{nsys_result_path}/result.nsys-rep",  f"./{nsys_result_path}/result.sqlite"],
        ]
    elif cmp == "cuSZi_24":
        cmd = [
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cusz-interp_24/build/cuszi", 
                    "-t", "f32",
                    "-m", "r2r",
                    "-i", data_path,
                    "-e", error_bound,
                    "-l", f"{data_size[0]}x{data_size[1]}x{data_size[2]}",
                    "-z", 
                    "-a", "2",
                    "--predictor", "spline3",
                    "--report", "time,cr"
                    ],
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cusz-interp_24/build/cuszi", 
                    "-i", data_path+".cusza",
                    "-x",
                    "--report", "time",
                    "--compare", data_path,
                    ],
                ["compareData",
                    "-f",  data_path, data_path+'.cuszx',],
                    ["rm",
                    data_path + '.cusza', data_path+'.cuszx', data_path + '.cusza.bitcomp',data_path + '.cusza.bitcomp.decompressed',  f"./{nsys_result_path}/result.nsys-rep",  f"./{nsys_result_path}/result.sqlite"]
                ]
    elif cmp == "cuSZi_a2_Huff_1":
        cmd = [
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cuszi", 
                    "-t", "f32",
                    "-m", "r2r",
                    "-i", data_path,
                    "-e", error_bound,
                    "-l", f"{data_size[0]}x{data_size[1]}x{data_size[2]}",
                    "-z", 
                    "-a", "2",
                    "-s", "cr",
                    "--predictor", "spline3",
                    "--report", "time,cr"
                    ],
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cuszi", 
                    "-i", data_path+".cusza",
                    "-x",
                    "--report", "time",
                    "--compare", data_path,
                    ],
                ["compareData",
                    "-f",  data_path, data_path+'.cuszx',],
                    ["rm",
                    data_path + '.cusza', data_path+'.cuszx', data_path + '.cusza.bitcomp',data_path + '.cusza.bitcomp.decompressed',  f"./{nsys_result_path}/result.nsys-rep",  f"./{nsys_result_path}/result.sqlite"]
                ]
    elif cmp == "cuSZi_a3_Huff_1":
        cmd = [
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cuszi", 
                    "-t", "f32",
                    "-m", "r2r",
                    "-i", data_path,
                    "-e", error_bound,
                    "-l", f"{data_size[0]}x{data_size[1]}x{data_size[2]}",
                    "-z", 
                    "-a", "3",
                    "-s", "cr",
                    "--predictor", "spline3",
                    "--report", "time,cr"
                    ],
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cuszi", 
                    "-i", data_path+".cusza",
                    "-x",
                    "--report", "time",
                    "--compare", data_path,
                    ],
                ["compareData",
                    "-f",  data_path, data_path+'.cuszx',],
                    ["rm",
                    data_path + '.cusza', data_path+'.cuszx', data_path + '.cusza.bitcomp',data_path + '.cusza.bitcomp.decompressed',  f"./{nsys_result_path}/result.nsys-rep",  f"./{nsys_result_path}/result.sqlite"]
                ]
    elif cmp == "cuSZi_a6_Huff_1":
        cmd = [
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cuszi", 
                    "-t", "f32",
                    "-m", "r2r",
                    "-i", data_path,
                    "-e", error_bound,
                    "-l", f"{data_size[0]}x{data_size[1]}x{data_size[2]}",
                    "-z", 
                    "-a", "6",
                    "-s", "cr",
                    "--predictor", "spline3",
                    "--report", "time,cr"
                    ],
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cuszi", 
                    "-i", data_path+".cusza",
                    "-x",
                    "--report", "time",
                    "--compare", data_path,
                    ],
                ["compareData",
                    "-f",  data_path, data_path+'.cuszx',],
                    ["rm",
                    data_path + '.cusza', data_path+'.cuszx', data_path + '.cusza.bitcomp',data_path + '.cusza.bitcomp.decompressed',  f"./{nsys_result_path}/result.nsys-rep",  f"./{nsys_result_path}/result.sqlite", ]
                ]
    elif cmp == "cuSZi_a2_Huff_0":
        cmd = [
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cuszi", 
                    "-t", "f32",
                    "-m", "r2r",
                    "-i", data_path,
                    "-e", error_bound,
                    "-l", f"{data_size[0]}x{data_size[1]}x{data_size[2]}",
                    "-z", 
                    "-a", "2",
                    "-s", "tp",
                    "--predictor", "spline3",
                    "--report", "time,cr"
                    ],
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cuszi", 
                    "-i", data_path+".cusza",
                    "-x",
                    "--report", "time",
                    "--compare", data_path,
                    ],
                ["compareData",
                    "-f",  data_path, data_path+'.cuszx',],
                    ["rm",
                    data_path + '.cusza', data_path+'.cuszx', data_path + '.cusza.bitcomp',data_path + '.cusza.bitcomp.decompressed',  f"./{nsys_result_path}/result.nsys-rep",  f"./{nsys_result_path}/result.sqlite"]
                ]
    elif cmp == "cuSZi_a3_Huff_0":
        cmd = [
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cuszi", 
                    "-t", "f32",
                    "-m", "r2r",
                    "-i", data_path,
                    "-e", error_bound,
                    "-l", f"{data_size[0]}x{data_size[1]}x{data_size[2]}",
                    "-z", 
                    "-a", "3",
                    "-s", "tp",
                    "--predictor", "spline3",
                    "--report", "time,cr"
                    ],
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cuszi", 
                    "-i", data_path+".cusza",
                    "-x",
                    "--report", "time",
                    "--compare", data_path,
                    ],
                ["compareData",
                    "-f",  data_path, data_path+'.cuszx',],
                    ["rm",
                    data_path + '.cusza', data_path+'.cuszx', data_path + '.cusza.bitcomp',data_path + '.cusza.bitcomp.decompressed',  f"./{nsys_result_path}/result.nsys-rep",  f"./{nsys_result_path}/result.sqlite"]
                ]
    elif cmp == "cuSZi_a6_Huff_0":
        cmd = [
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cuszi", 
                    "-t", "f32",
                    "-m", "r2r",
                    "-i", data_path,
                    "-e", error_bound,
                    "-l", f"{data_size[0]}x{data_size[1]}x{data_size[2]}",
                    "-z", 
                    "-a", "6",
                    "-s", "tp",
                    "--predictor", "spline3",
                    "--report", "time,cr"
                    ],
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cuszi", 
                    "-i", data_path+".cusza",
                    "-x",
                    "--report", "time",
                    "--compare", data_path,
                    ],
                ["compareData",
                    "-f",  data_path, data_path+'.cuszx',],
                    ["rm",
                    data_path + '.cusza', data_path+'.cuszx', data_path + '.cusza.bitcomp',data_path + '.cusza.bitcomp.decompressed',  f"./{nsys_result_path}/result.nsys-rep",  f"./{nsys_result_path}/result.sqlite", ]
                ]
    elif cmp == "cuSZi_interp_16_4steps":
        cmd = [
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cusz-interp_16_4steps/build/cuszi", 
                    "-t", "f32",
                    "-m", "r2r",
                    "-i", data_path,
                    "-e", error_bound,
                    "-l", f"{data_size[0]}x{data_size[1]}x{data_size[2]}",
                    "-z", 
                    "-a", "2",
                    "--predictor", "spline3",
                    "--report", "time,cr"
                    ],
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cusz-interp_16_4steps/build/cuszi", 
                    "-i", data_path+".cusza",
                    "-x",
                    "--report", "time",
                    "--compare", data_path,
                    ],
                ["compareData",
                    "-f",  data_path, data_path+'.cuszx',],
                    ["rm",
                    data_path + '.cusza', data_path+'.cuszx', data_path + '.cusza.bitcomp',data_path + '.cusza.bitcomp.decompressed',  f"./{nsys_result_path}/result.nsys-rep",  f"./{nsys_result_path}/result.sqlite", ]
                ]
    elif cmp == "cuSZi_interp_16_4steps_reorder":
        cmd = [
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cusz-interp_16_4steps_reorder/build/cuszi", 
                    "-t", "f32",
                    "-m", "r2r",
                    "-i", data_path,
                    "-e", error_bound,
                    "-l", f"{data_size[0]}x{data_size[1]}x{data_size[2]}",
                    "-z", 
                    "-a", "2",
                    "--predictor", "spline3",
                    "--report", "time,cr"
                    ],
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cusz-interp_16_4steps_reorder/build/cuszi", 
                    "-i", data_path+".cusza",
                    "-x",
                    "--report", "time",
                    "--compare", data_path,
                    ],
                ["compareData",
                    "-f",  data_path, data_path+'.cuszx',],
                    ["rm",
                    data_path + '.cusza', data_path+'.cuszx', data_path + '.cusza.bitcomp',data_path + '.cusza.bitcomp.decompressed',  f"./{nsys_result_path}/result.nsys-rep",  f"./{nsys_result_path}/result.sqlite", ]
                ]
    elif cmp == "cuSZi_interp_16_4steps_reorder_att_balance_a3":
        cmd = [
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cusz-interp_16_4steps_reorder_att_balance/build/cuszi", 
                    "-t", "f32",
                    "-m", "r2r",
                    "-i", data_path,
                    "-e", error_bound,
                    "-l", f"{data_size[0]}x{data_size[1]}x{data_size[2]}",
                    "-z", 
                    "-a", "3",
                    "--predictor", "spline3",
                    "--report", "time,cr"
                    ],
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cusz-interp_16_4steps_reorder_att_balance/build/cuszi", 
                    "-i", data_path+".cusza",
                    "-x",
                    "--report", "time",
                    "--compare", data_path,
                    ],
                ["compareData",
                    "-f",  data_path, data_path+'.cuszx',],
                    ["rm",
                    data_path + '.cusza', data_path+'.cuszx', data_path + '.cusza.bitcomp',data_path + '.cusza.bitcomp.decompressed',  f"./{nsys_result_path}/result.nsys-rep",  f"./{nsys_result_path}/result.sqlite", ]
                ]
    elif cmp == "cuSZi_interp_16_4steps_reorder_att_balance_a6":
        cmd = [
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cusz-interp_16_4steps_reorder_att_balance/build/cuszi", 
                    "-t", "f32",
                    "-m", "r2r",
                    "-i", data_path,
                    "-e", error_bound,
                    "-l", f"{data_size[0]}x{data_size[1]}x{data_size[2]}",
                    "-z", 
                    "-a", "6",
                    "--predictor", "spline3",
                    "--report", "time,cr"
                    ],
                ["nsys", "profile",  "--stats=true", "-o", nsys_result_path, "cusz-interp_16_4steps_reorder_att_balance/build/cuszi", 
                    "-i", data_path+".cusza",
                    "-x",
                    "--report", "time",
                    "--compare", data_path,
                    ],
                ["compareData",
                    "-f",  data_path, data_path+'.cuszx',],
                    ["rm",
                    data_path + '.cusza', data_path+'.cuszx', data_path + '.cusza.bitcomp',data_path + '.cusza.bitcomp.decompressed',  f"./{nsys_result_path}/result.nsys-rep",  f"./{nsys_result_path}/result.sqlite", ]
                ]
    
    cmd[-1].append(os.path.join(nsys_result_path, "*.sqlite") )
    cmd[-1].append(os.path.join(nsys_result_path, "*.nsys-rep"))
    cmd_nvcomp = [
        "nsys", "profile",  "--stats=true", "-o", nsys_result_path, "benchmark_bitcomp_chunked",
        "-f", data_path, "-a", "0"
    ]
    cmd_bitcomp = [
        "nsys", "profile",  "--stats=true", "-o", nsys_result_path, "bitcomp_example",
        "-r", data_path,
    ]
    
    return cmd, cmd_nvcomp, cmd_bitcomp


# Define the DataFrame with MultiIndex
# index = pd.MultiIndex.from_product(
#     [['FZ-GPU', 'cuSZ', 'cuSZp', 'cuzfp', 'cuSZx'], ['1e-2', '5e-3', '1e-3','5e-4', '1e-4', '5e-5', '1e-5',]],
#     names=['Method', 'Error_Bound']
# )
# columns = ['CR', 'PSNR', 'Comp_Throughput (GB/s)', 'Decomp_Throughput (GB/s)', 'Comp_Throughput_nsys (GB/s)', 'Decomp_Throughput_nsys (GB/s)']
# df = pd.DataFrame(index=index, columns=columns).sort_index()

def run_FZGPU(command, bitcomp_cmd_nv, bitcomp_cmd, file_path):
    result = subprocess.run(command[0], capture_output=True, text=True)
    qcat_result = subprocess.run(command[1], capture_output=True, text=True)
    nvcomp = copy.deepcopy(bitcomp_cmd_nv)
    nvcomp[-3] += '.fzgpua'
    bitcomp = copy.deepcopy(bitcomp_cmd)
    bitcomp[-1] += '.fzgpua'
    nvcomp_result = subprocess.run(nvcomp, capture_output=True, text=True)
    bitcomp_result = ""
    # bitcomp_result = subprocess.run(bitcomp, capture_output=True, text=True)
    
    with open(file_path, 'w') as file:
        file.write("-fzgpu-\n" + result.stdout + result.stderr + "-fzgpu-\n" + 
                   "-nvcomp-\n" + nvcomp_result.stdout + nvcomp_result.stderr + "-nvcomp-\n" + 
                   "-bitcomp-\n" + "-bitcomp-\n" + 
                   "-compareData-\n" + qcat_result.stdout + qcat_result.stderr + "-compareData-\n")
    result = subprocess.run(command[-1], capture_output=True, text=True, shell=True)

def run_cuSZ(command, bitcomp_cmd_nv, bitcomp_cmd, file_path):
    result = subprocess.run(command[0], capture_output=True, text=True)
    decomp_result = subprocess.run(command[1], capture_output=True, text=True)
    qcat_result = subprocess.run(command[2], capture_output=True, text=True)
    nvcomp = copy.deepcopy(bitcomp_cmd_nv)
    nvcomp[-3] += '.cusza'
    bitcomp = copy.deepcopy(bitcomp_cmd)
    bitcomp[-1] += '.cusza'
    nvcomp_result = subprocess.run(nvcomp, capture_output=True, text=True)
    # bitcomp_result = ""
    bitcomp_result = subprocess.run(bitcomp, capture_output=True, text=True)
    with open(file_path, 'w') as file:
        file.write("-cusz_compress-\n" + result.stdout + result.stderr + "-cusz_compress-\n" + 
                   "-cusz_decompress-\n" + decomp_result.stdout + decomp_result.stderr + "-cusz_decompress-\n" + 
                   "-nvcomp-\n" + nvcomp_result.stdout + nvcomp_result.stderr + "-nvcomp-\n" + 
                    "-bitcomp-\n" + bitcomp_result.stdout + bitcomp_result.stderr + "-bitcomp-\n" + 
                   "-compareData-\n" + qcat_result.stdout + qcat_result.stderr + '-compareData-\n')
    result = subprocess.run(command[-1], capture_output=True, text=True, shell=True)

def run_cuSZp_outlier(command, bitcomp_cmd_nv, bitcomp_cmd, file_path):
    result = subprocess.run(command[0], capture_output=True, text=True)
    qcat_result = subprocess.run(command[1], capture_output=True, text=True)
    nvcomp_result = subprocess.run(bitcomp_cmd_nv, capture_output=True, text=True)
    # bitcomp_result = subprocess.run(bitcomp_cmd, capture_output=True, text=True)
    nvcomp = copy.deepcopy(bitcomp_cmd_nv)
    nvcomp[-3] += '.cuszpa'
    bitcomp = copy.deepcopy(bitcomp_cmd)
    bitcomp[-1] += '.cuszpa'
    nvcomp_result = subprocess.run(nvcomp, capture_output=True, text=True)
    bitcomp_result = ""
    # bitcomp_result = subprocess.run(bitcomp, capture_output=True, text=True)
    with open(file_path, 'w') as file:
        file.write( "-cuszp_compress-\n" + result.stdout + result.stderr + "-cuszp_compress-\n" + 
                   "-nvcomp-\n" + nvcomp_result.stdout + nvcomp_result.stderr + "-nvcomp-\n" + 
                   "-bitcomp-\n" + "-bitcomp-\n" + 
                   "-compareData-\n" + qcat_result.stdout + qcat_result.stderr + "-compareData-\n" )
    result = subprocess.run(command[-1], capture_output=True, text=True, shell=True)

def run_cuSZp_plain(command, bitcomp_cmd_nv, bitcomp_cmd, file_path):
    result = subprocess.run(command[0], capture_output=True, text=True)
    qcat_result = subprocess.run(command[1], capture_output=True, text=True)
    nvcomp_result = subprocess.run(bitcomp_cmd_nv, capture_output=True, text=True)
    # bitcomp_result = subprocess.run(bitcomp_cmd, capture_output=True, text=True)
    nvcomp = copy.deepcopy(bitcomp_cmd_nv)
    nvcomp[-3] += '.cuszpa'
    bitcomp = copy.deepcopy(bitcomp_cmd)
    bitcomp[-1] += '.cuszpa'
    nvcomp_result = subprocess.run(nvcomp, capture_output=True, text=True)
    bitcomp_result = ""
    # bitcomp_result = subprocess.run(bitcomp, capture_output=True, text=True)
    with open(file_path, 'w') as file:
        file.write( "-cuszp_compress-\n" + result.stdout + result.stderr + "-cuszp_compress-\n" + 
                   "-nvcomp-\n" + nvcomp_result.stdout + nvcomp_result.stderr + "-nvcomp-\n" + 
                   "-bitcomp-\n" + "-bitcomp-\n" + 
                   "-compareData-\n" + qcat_result.stdout + qcat_result.stderr + "-compareData-\n" )
    result = subprocess.run(command[-1], capture_output=True, text=True, shell=True)
        
def run_cuSZx(command, bitcomp_cmd_nv, bitcomp_cmd, file_path):
    result = subprocess.run(command[0], capture_output=True, text=True)
    decomp_result = subprocess.run(command[1], capture_output=True, text=True)
    psnr_result = subprocess.run(command[2], capture_output=True, text=True)
    nvcomp = copy.deepcopy(bitcomp_cmd_nv)
    nvcomp[-3] += '.szx'
    bitcomp = copy.deepcopy(bitcomp_cmd)
    bitcomp[-1] += '.szx'
    nvcomp_result = subprocess.run(nvcomp, capture_output=True, text=True)
    bitcomp_result = ""
    # bitcomp_result = subprocess.run(bitcomp, capture_output=True, text=True)
    with open(file_path, 'w') as file:
        file.write("-cuszx_compress-\n" + result.stdout + result.stderr + "-cuszx_compress-\n" + 
                   "-cuszx_decompress-\n" + decomp_result.stdout + decomp_result.stderr + "-cuszx_decompress-\n" + 
                   "-compareData-\n" + psnr_result.stdout + psnr_result.stderr +  "-compareData-\n" + 
                   "-nvcomp-\n" + nvcomp_result.stdout + nvcomp_result.stderr + "-nvcomp-\n" + 
                   "-bitcomp-\n" + "-bitcomp-\n" )
    result = subprocess.run(command[-1], capture_output=True, text=True, shell=True)

def run_cuzfp(command, bitcomp_cmd_nv, bitcomp_cmd, file_path):
    result = subprocess.run(command[0], capture_output=True, text=True)
    decomp_result = subprocess.run(command[1], capture_output=True, text=True)
    psnr_result = subprocess.run(command[2], capture_output=True, text=True)
    
    nvcomp = copy.deepcopy(bitcomp_cmd_nv)
    nvcomp[-3] += '.cuzfpa'
    bitcomp = copy.deepcopy(bitcomp_cmd)
    bitcomp[-1] += '.cuzfpa'
    nvcomp_result = subprocess.run(nvcomp, capture_output=True, text=True)
    bitcomp_result = ""
    # bitcomp_result = subprocess.run(bitcomp, capture_output=True, text=True)
    with open(file_path, 'w') as file:
        file.write("-cuzfp_compress\n" + result.stderr + "-cuzfp_compress\n" + 
                   "-cuzfp_decompress-\n" + decomp_result.stderr + "-cuzfp_decompress-\n" + 
                   "-compareData-\n" + psnr_result.stdout + "-compareData-\n" + 
                   "-nsys compress-\n" + result.stdout + "-nsys compress-\n" + 
                   "-nsys decompress-\n" + decomp_result.stdout + "-nsys decompress-\n" + 
                   "-nvcomp-\n" + nvcomp_result.stdout + "-nvcomp-\n" + 
                   "-bitcomp-\n" + "-bitcomp-\n")
    result = subprocess.run(command[-1], capture_output=True, text=True, shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help="(MANDATORY) input data folder", type=str)
    parser.add_argument('--output', '-o', help="(MANDATORY) output folder for logs", type=str)
    parser.add_argument('--dim', '-d', help="data dimension", type=int,default=3)
    parser.add_argument('--dims', '-m', help="(MANDATORY) data dimension", type=str,nargs="+")
    parser.add_argument('--cmp', '-c', help="specify a list of compressors", type=str,nargs="*")
    parser.add_argument('--eb', '-e', help="specify a list of error bounds", type=str,nargs="*")
    parser.add_argument('--br', '-b', help="specify a list of bit rates", type=str,nargs="*")
    parser.add_argument('--nsys', '-n', help="specify nsys profile result dir", type=str, default="./nsys_results")
    parser.add_argument("--num_splits", type=int, default=1, help="Total number of splits (i.e., number of processes)")
    parser.add_argument("--proc_id", type=int, default=0, help="Current process ID (0-based index)")
    args = parser.parse_args()
    
    datafolder   = args.input
    outputfolder = args.output
    data_size    = args.dims
    cmp_list     = args.cmp
    eb_list      = args.eb
    br_list      = args.br
    num_splits      = args.num_splits
    proc_id      = args.proc_id
    nsys_result_path         = args.nsys + f"{proc_id}"

    if any(e is None for e in [args.input, args.output, args.dims]):
        print()
        print("need to specify MANDATORY arguments")
        print()
        parser.print_help()
        sys.exit(1)
    
    

    method_list = [
                #     'FZGPU', 
                #    'cuSZp_outlier',
                #     'cuSZp_plain', 
                    # 'cuzfp', 
                    #   'cuSZ_24', 
                    'cuSZi_24',
                    #   'cuSZi_a3_Huff_1', 'cuSZi_a6_Huff_1', 
                    #   'cuSZi_a3_Huff_0', 'cuSZi_a6_Huff_0', 
                    # 'cuSZi_interp_16_4steps',
                    # 'cuSZi_interp_16_4steps_reorder',
                    # 'cuSZi_interp_16_4steps_reorder_att_balance_a3',
                    # 'cuSZi_interp_16_4steps_reorder_att_balance_a6',
                    ]
    error_bound_list = ['1e-2', '5e-3', '1e-3','5e-4', '1e-4', '5e-5', '1e-5']
    bit_rate_list = ['0.5', '1', '2', '4', '6', '8', '12', '16']
    run_func_dict = {"FZGPU":run_FZGPU, "cuSZ":run_cuSZ, "cuSZp_outlier":run_cuSZp_outlier, 
                "cuSZp_plain":run_cuSZp_plain, "cuzfp":run_cuzfp, "cuZFP":run_cuzfp ,
                 "cuSZi_a2_Huff_1":run_cuSZ,
                 "cuSZi_a3_Huff_1":run_cuSZ,
                 "cuSZi_a6_Huff_1":run_cuSZ,
                 "cuSZi_a2_Huff_0":run_cuSZ,
                 "cuSZi_a3_Huff_0":run_cuSZ,
                 "cuSZi_a6_Huff_0":run_cuSZ,
                 'cuSZi_interp_16_4steps':run_cuSZ ,
                'cuSZi_interp_16_4steps_reorder':run_cuSZ,
                'cuSZi_interp_16_4steps_reorder_att_balance_a3':run_cuSZ,
                'cuSZi_interp_16_4steps_reorder_att_balance_a6':run_cuSZ,
                 "cuSZi_24":run_cuSZ,
                 "cuSZ_24":run_cuSZ,
                }
    
    cmp_list = method_list      if cmp_list is None else cmp_list
    eb_list  = error_bound_list if eb_list is None else eb_list
    br_list  = bit_rate_list    if br_list is None else br_list
    
    
    datafiles=os.listdir(datafolder)
    logfiles=os.listdir(outputfolder)
    print(logfiles)
    # assert 0
    datafiles=[file for file in datafiles if file.split(".")[-1]=="dat" or file.split(".")[-1]=="f32" or file.split(".")[-1]=="bin"]
    # 按照 num_splits 进行划分
    datafiles.sort()  # 先排序，确保所有进程分到的文件顺序一致
    chunk_size = (len(datafiles) + num_splits - 1) // num_splits
    remainder = len(datafiles) % num_splits

    # 计算当前进程应该处理的文件范围
    start_idx = proc_id * chunk_size
    end_idx = start_idx + chunk_size
    datafiles = datafiles[start_idx:end_idx]
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    
    if not os.path.exists(nsys_result_path):
        os.makedirs(nsys_result_path)
        
    echo_cmd = lambda cmd: print("    ", " ".join(cmd))
    
    for cmp in cmp_list:    
        if cmp != 'cuzfp':
            for file in tqdm(datafiles):
                for eb in eb_list:
                    data_path = os.path.join(datafolder, file)
                    file_path = os.path.join(outputfolder, file)
                    if (file + "_eb=" + eb + "_" + cmp) in logfiles:
                        print("pass", file + "_eb=" + eb + "_" + cmp) 
                        continue
                    print(file + "_eb=" + eb + "_" + cmp)
                    # assert 0
                    cmd, cmd_nvcomp, cmd_bitcomp = update_command(cmp, data_path, data_size, error_bound=eb, nsys_result_path=nsys_result_path, proc_id=proc_id)
                    for i in cmd: 
                        echo_cmd(i)
                    echo_cmd(cmd_nvcomp); echo_cmd(cmd_bitcomp)
                    run_func_dict[cmp](cmd, cmd_nvcomp, cmd_bitcomp, file_path + "_eb=" + eb + "_" + cmp)
                
                    
                    
        else:
            for file in tqdm(datafiles):
                for br in br_list:
                    data_path = os.path.join(datafolder, file)
                    file_path = os.path.join(outputfolder, file)
                    if (file + "_br=" + br + "_" + cmp) in logfiles:
                        print("pass", file + "_br=" + br + "_" + cmp) 
                        continue
                    print(file + "_eb=" + eb + "_" + cmp)
                    cmd, cmd_nvcomp, cmd_bitcomp = update_command(cmp, data_path, data_size, bit_rate=br, nsys_result_path=nsys_result_path, proc_id=proc_id)
                    for i in cmd: 
                        echo_cmd(i)
                    echo_cmd(cmd_nvcomp); echo_cmd(cmd_bitcomp)
                    run_func_dict[cmp](cmd, cmd_nvcomp, cmd_bitcomp, file_path + "_br=" + br + "_" + cmp)
