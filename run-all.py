import os
import sys
import copy
import argparse
import subprocess
from tqdm import tqdm
import pandas as pd


def compute_throughput(elapsed_time, data_size):
    return float(data_size[0]) * float(data_size[1]) * float(data_size[2]) * 4 / 1024.0/ 1024.0/ 1024.0 / (elapsed_time * 1e-9)


def update_command(cmp, data_path, data_size, error_bound="1e-2", bit_rate="2", cuszx_block_size=64):
    work_path = os.getenv('WORK_PATH')
    print(cmp, data_size[0], data_size[1], data_size[2] )
    try:
        nbEle = int(data_size[0]) * int(data_size[1]) * int(data_size[2])
    except:
        assert 0
    print(nbEle)
    
    if cmp == "FZGPU":
        cmd = [
                    ["nsys", "profile", "--stats=true", "-o", "nsys_result_" + error_bound, os.path.join( work_path, "FZ-GPU/fz-gpu"),
                    data_path, 
                    data_size[0], 
                    data_size[1], 
                    data_size[2], 
                    error_bound,
                    ],
                    ["compareData",
                    "-f",  data_path, data_path+'.fzgpux',]]
    elif cmp == "cuSZ":
        cmd = [["nsys", "profile", "--stats=true", "-o", "nsys_result_" + error_bound, os.path.join( work_path, "cusz-canary/build/cusz"), 
                    "-t", "f32",
                    "-m", "r2r",
                    "-i", data_path,
                    "-e", error_bound,
                    "-l", f"{data_size[0]}x{data_size[1]}x{data_size[2]}",
                    "-z", 
                    "--predictor", "lorenzo",
                    "--report", "time,cr",
                    "-a", "0",],
                ["nsys", "profile", "--stats=true", "-o", "nsys_result_" + error_bound, os.path.join( work_path, "cusz-canary/build/cusz"), 
                    "-i", data_path+".cusza",
                    "-x",
                    "--report", "time",
                    "--compare", data_path,],
                ["compareData",
                    "-f",  data_path, data_path+'.cuszx',]]
    elif cmp == "cuSZp":
        cmd = [
                ["nsys", "profile", "--stats=true", "-o", "nsys_result_" + error_bound, os.path.join( work_path, "cuSZp/build/examples/bin/cuSZp_gpu_f32_api"),
                    data_path,
                    "REL", error_bound,],
                ["compareData",
                    "-f",  data_path, data_path+'.cuszpx',]]
    elif cmp == "cuzfp":
        cmd = [["nsys", "profile", "--stats=true", "-o", "nsys_result_" + bit_rate, os.path.join( work_path, "zfp/build/bin/zfp"),
                    "-i", data_path,
                    "-z", data_path+'.cuzfpa',
                    "-x", "cuda",
                    "-f", 
                    "-3", 
                    data_size[0], 
                    data_size[1], 
                    data_size[2], 
                    "-r", bit_rate],
                ["nsys", "profile", "--stats=true", "-o", "nsys_result_" + bit_rate, os.path.join( work_path, "zfp/build/bin/zfp"),
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
                    "-f",  data_path, data_path+'.cuzfpx',],]
    elif cmp == "cuSZx":
        cmd = [["nsys", "profile", "--stats=true", "-o", "nsys_result_" + error_bound, os.path.join( work_path, "SZx-main/SZx-main/build/example/szx_testfloat_compress_fastmode2"),
                    data_path, f"{cuszx_block_size}", error_bound, "--cuda"],
                ["nsys", "profile", "--stats=true", "-o", "nsys_result_" + error_bound, os.path.join( work_path, "SZx-main/SZx-main/build/example/szx_testfloat_decompress_fastmode2"),
                    data_path+".szx", f"{nbEle}", "--cuda"],
                ["compareData",
                    "-f",  data_path, data_path+'.szx.out',],
        ]
    elif cmp == "cuSZi":
        cmd = [
                ["nsys", "profile", "--stats=true", "-o", "nsys_result_" + error_bound, os.path.join( work_path, "cusz-canary/build/cusz"), 
                    "-t", "f32",
                    "-m", "r2r",
                    "-i", data_path,
                    "-e", error_bound,
                    "-l", f"{data_size[0]}x{data_size[1]}x{data_size[2]}",
                    "-z", 
                    "-a", "2",
                    "--predictor", "spline3",
                    "--report", "time,cr"],
                ["nsys", "profile", "--stats=true", "-o", "nsys_result_" + error_bound, os.path.join( work_path, "cusz-canary/build/cusz"), 
                    "-i", data_path+".cusza",
                    "-x",
                    "--report", "time",
                    "--compare", data_path,],
                ["compareData",
                    "-f",  data_path, data_path+'.cuszx',]
                ]
    cmd_nvcomp = [
        "nsys", "profile", "--stats=true", "-o", "nsys_result_" + error_bound, os.path.join( work_path, "nvcomp/bin/benchmark_bitcomp_chunked"),
        "-f", data_path, "-a", "0"
    ]
    cmd_bitcomp = [
        "nsys", "profile", "--stats=true", "-o", "nsys_result_" + error_bound, os.path.join( work_path, "bitcomp_lossless_example/bitcomp_example"),
        "-r", data_path,
    ]
    
    return cmd, cmd_nvcomp, cmd_bitcomp


# Define the DataFrame with MultiIndex
index = pd.MultiIndex.from_product(
    [['FZ-GPU', 'cuSZ', 'cuSZp', 'cuzfp', 'cuSZx'], ['1e-2', '5e-3', '1e-3','5e-4', '1e-4', '5e-5', '1e-5',]],
    names=['Method', 'Error_Bound']
)
columns = ['CR', 'PSNR', 'Comp_Throughput (GB/s)', 'Decomp_Throughput (GB/s)', 'Comp_Throughput_nsys (GB/s)', 'Decomp_Throughput_nsys (GB/s)']
df = pd.DataFrame(index=index, columns=columns).sort_index()

def run_FZGPU(command, bitcomp_cmd_nv, bitcomp_cmd, file_path):
    result = subprocess.run(command[0], capture_output=True, text=True)
    qcat_result = subprocess.run(command[1], capture_output=True, text=True)
    nvcomp = copy.deepcopy(bitcomp_cmd_nv)
    nvcomp[-3] += '.fzgpua'
    bitcomp = copy.deepcopy(bitcomp_cmd)
    bitcomp[-1] += '.fzgpua'
    nvcomp_result = subprocess.run(nvcomp, capture_output=True, text=True)
    bitcomp_result = subprocess.run(bitcomp, capture_output=True, text=True)
    
    with open(file_path, 'w') as file:
        file.write("-fzgpu-\n" + result.stdout + "-fzgpu-\n" + 
                   "-nvcomp-\n" + nvcomp_result.stdout + "-nvcomp-\n" + 
                   "-bitcomp-\n" + bitcomp_result.stdout + "-bitcomp-\n" + 
                   "-compareData-\n" + qcat_result.stdout + "-compareData-\n")

def run_cuSZ(command, bitcomp_cmd_nv, bitcomp_cmd, file_path):
    result = subprocess.run(command[0], capture_output=True, text=True)
    decomp_result = subprocess.run(command[1], capture_output=True, text=True)
    qcat_result = subprocess.run(command[2], capture_output=True, text=True)
    nvcomp = copy.deepcopy(bitcomp_cmd_nv)
    nvcomp[-3] += '.cusza'
    bitcomp = copy.deepcopy(bitcomp_cmd)
    bitcomp[-1] += '.cusza'
    nvcomp_result = subprocess.run(nvcomp, capture_output=True, text=True)
    bitcomp_result = subprocess.run(bitcomp, capture_output=True, text=True)
    with open(file_path, 'w') as file:
        file.write("-cusz_compress-\n" + result.stdout + "-cusz_compress-\n" + 
                   "-cusz_decompress-\n" + decomp_result.stdout + "-cusz_decompress-\n" + 
                   "-nvcomp-\n" + nvcomp_result.stdout +  "-nvcomp-\n" + 
                   "-bitcomp-\n" + bitcomp_result.stdout + "-bitcomp-\n" +
                   "-compareData-\n" + qcat_result.stdout + '-compareData-\n')

def run_cuSZp(command, bitcomp_cmd_nv, bitcomp_cmd, file_path):
    result = subprocess.run(command[0], capture_output=True, text=True)
    qcat_result = subprocess.run(command[1], capture_output=True, text=True)
    nvcomp_result = subprocess.run(bitcomp_cmd_nv, capture_output=True, text=True)
    bitcomp_result = subprocess.run(bitcomp_cmd, capture_output=True, text=True)
    nvcomp = copy.deepcopy(bitcomp_cmd_nv)
    nvcomp[-3] += '.cuszpa'
    bitcomp = copy.deepcopy(bitcomp_cmd)
    bitcomp[-1] += '.cuszpa'
    with open(file_path, 'w') as file:
        file.write( "-cuszp_compress-\n" + result.stdout + "-cuszp_compress-\n" + 
                   "-nvcomp-\n" + nvcomp_result.stdout + "-nvcomp-\n" + 
                   "-bitcomp-\n" + bitcomp_result.stdout + "-bitcomp-\n" + 
                   "-compareData-\n" + qcat_result.stdout + "-compareData-\n" )
        
def run_cuSZx(command, bitcomp_cmd_nv, bitcomp_cmd, file_path):
    result = subprocess.run(command[0], capture_output=True, text=True)
    decomp_result = subprocess.run(command[1], capture_output=True, text=True)
    psnr_result = subprocess.run(command[2], capture_output=True, text=True)
    nvcomp = copy.deepcopy(bitcomp_cmd_nv)
    nvcomp[-3] += '.szx'
    bitcomp = copy.deepcopy(bitcomp_cmd)
    bitcomp[-1] += '.szx'
    nvcomp_result = subprocess.run(nvcomp, capture_output=True, text=True)
    bitcomp_result = subprocess.run(bitcomp, capture_output=True, text=True)
    with open(file_path, 'w') as file:
        file.write("-cuszx_compress-\n" + result.stdout + result.stderr + "-cuszx_compress-\n" + 
                   "-cuszx_decompress-\n" + decomp_result.stdout + decomp_result.stderr + "-cuszx_decompress-\n" + 
                   "-compareData-\n" + psnr_result.stdout + psnr_result.stderr +  "-compareData-\n" + 
                   "-nvcomp-\n" + nvcomp_result.stdout + nvcomp_result.stderr + "-nvcomp-\n" + 
                   "-bitcomp-\n" + bitcomp_result.stdout + bitcomp_result.stderr + "-bitcomp-\n" )

def run_cuzfp(command, bitcomp_cmd_nv, bitcomp_cmd, file_path):
    result = subprocess.run(command[0], capture_output=True, text=True)
    decomp_result = subprocess.run(command[1], capture_output=True, text=True)
    psnr_result = subprocess.run(command[2], capture_output=True, text=True)
    
    nvcomp = copy.deepcopy(bitcomp_cmd_nv)
    nvcomp[-3] += '.cuzfpa'
    bitcomp = copy.deepcopy(bitcomp_cmd)
    bitcomp[-1] += '.cuzfpa'
    nvcomp_result = subprocess.run(nvcomp, capture_output=True, text=True)
    bitcomp_result = subprocess.run(bitcomp, capture_output=True, text=True)
    with open(file_path, 'w') as file:
        file.write("-cuzfp_compress\n" + result.stderr + "-cuzfp_compress\n" + 
                   "-cuzfp_decompress-\n" + decomp_result.stderr + "-cuzfp_decompress-\n" + 
                   "-compareData-\n" + psnr_result.stdout + "-compareData-\n" + 
                   "-nsys compress-\n" + result.stdout + "-nsys compress-\n" + 
                   "-nsys decompress-\n" + decomp_result.stdout + "-nsys decompress-\n" + 
                   "-nvcomp-\n" + nvcomp_result.stdout + "-nvcomp-\n" + 
                   "-bitcomp-\n" + bitcomp_result.stdout + "-bitcomp-\n")
    
    
def evaluate_FZGPU(command, error_bound):
    #FZ-GPU
    result = subprocess.run(command, capture_output=True, text=True)
    output_lines = result.stdout.splitlines()
    method = 'FZ-GPU'
    
    for output_line in output_lines:
        # print(output_line)
        if "compression ratio" in output_line:
            output_line_split = output_line.split()
            index = output_line_split.index("ratio:") + 1
            compression_ratio_value = float(output_line_split[index])
            df.loc[(method, error_bound), 'CR'] = compression_ratio_value
        if "PSNR" in output_line:
            output_line_split = output_line.split()
            index = output_line_split.index("PSNR:") + 1
            PSNR_value = float(output_line_split[index])
            df.loc[(method, error_bound), 'PSNR'] = PSNR_value
        if "compression e2e throughput" in output_line and "decompression e2e throughput" not in output_line:
            output_line_split = output_line.split()
            index = output_line_split.index("throughput:") + 1
            compression_throughput = float(output_line_split[index])
            df.loc[(method, error_bound), 'Comp_Throughput (GB/s)'] = compression_throughput
        if "decompression e2e throughput" in output_line:
            output_line_split = output_line.split()
            index = output_line_split.index("throughput:") + 1
            decompression_throughput = float(output_line_split[index])
            df.loc[(method, error_bound), 'Decomp_Throughput (GB/s)'] = decompression_throughput
        if "decompressionFusedKernel" in output_line:
            output_line_split = output_line.split()
            decompressed_time = float(output_line_split[1].replace(",",""))
        if "compressionFusedKernel" in output_line and "decompressionFusedKernel" not in output_line:
            output_line_split = output_line.split()
            compressed_time = float(output_line_split[1].replace(",",""))
        if "cusz::experimental::c_lorenzo_3d1l_32x8x8data_mapto32x1x8" in output_line:
            compressed_time += float(output_line_split[1].replace(",",""))
            compression_throughput = compute_throughput(compressed_time)
            df.loc[(method, error_bound), 'Comp_Throughput_nsys (GB/s)'] = compression_throughput
        if "cusz::experimental::x_lorenzo_3d1l_32x8x8data_mapto32x1x8" in output_line:
            decompressed_time += float(output_line_split[1].replace(",",""))
            decompression_throughput = compute_throughput(decompressed_time)
            df.loc[(method, error_bound), 'Decomp_Throughput_nsys (GB/s)'] = decompression_throughput
       
            

def evaluate_cuSZ(command_comp, command_decomp, error_bound):
    # cuSZ
    method = 'cuSZ'
    
    # compression
    result = subprocess.run(command_comp, capture_output=True, text=True)
    output_lines = result.stdout.splitlines()
    
    for output_line in output_lines:
        # print(output_line)
        compressed_time = 0
        output_line_split = output_line.split()
        if "(total)" in output_line:
            index = output_line_split.index("(total)") + 2
            compression_throughput = float(output_line_split[index])
            df.loc[(method, error_bound), 'Comp_Throughput (GB/s)'] = compression_throughput
        if "cusz::c_spline3d_infprecis_32x8x8data" in output_line:
            compressed_time += float(output_line_split[1].replace(",","")) / (float(output_line_split[0]) * 1e-2)
        if "psz::extrema_kernel<float>" in output_line:
            compressed_time -= float(output_line_split[1].replace(",",""))
        compression_throughput = compute_throughput(compressed_time)
        df.loc[(method, error_bound), 'Comp_Throughput_nsys (GB/s)'] = compression_throughput
    # decompression
    decomp_result = subprocess.run(command_decomp, capture_output=True, text=True)
    output_lines = decomp_result.stdout.splitlines()
    for output_line in output_lines:
        decompressed_time = 0
        output_line_split = output_line.split()
        if 'metrics' in output_line and output_line_split[0] == 'metrics':
            # CR
            index = output_line_split.index("metrics") + 1
            compression_ratio_value = float(output_line_split[index])
            df.loc[(method, error_bound), 'CR'] = compression_ratio_value

            # PSNR
            index = output_line_split.index("metrics") + 4
            PSNR_value = float(output_line_split[index])
            df.loc[(method, error_bound), 'PSNR'] = PSNR_value
        if "(total)" in output_line:
            index = output_line_split.index("(total)") + 2
            decompression_throughput = float(output_line_split[index])
            df.loc[(method, error_bound), 'Decomp_Throughput (GB/s)'] = decompression_throughput
        if "cusz::x_spline3d_infprecis_32x8x8data" in output_line:
            decompressed_time += float(output_line_split[1].replace(",",""))
        if "hf_decode_kernel" in output_line:
            decompressed_time += float(output_line_split[1].replace(",",""))
    decompression_throughput = compute_throughput(decompressed_time)
    df.loc[(method, error_bound), 'Decomp_Throughput_nsys (GB/s)'] = decompression_throughput

def evaluate_cuSZp(command, error_bound):
    #cuSZp
    method = 'cuSZp'
    result = subprocess.run(command, capture_output=True, text=True)
    output_lines = result.stdout.splitlines()
    
    for output_line in output_lines:
        if "compression ratio" in output_line:
            output_line_split = output_line.split()
            index = output_line_split.index("ratio:") + 1
            compression_ratio_value = float(output_line_split[index])
            df.loc[(method, error_bound), 'CR'] = compression_ratio_value
        if "PSNR" in output_line:
            output_line_split = output_line.split()
            index = output_line_split.index("PSNR:") + 1
            PSNR_value = float(output_line_split[index])
            df.loc[(method, error_bound), 'PSNR'] = PSNR_value
        if "cuSZp compression   end-to-end speed" in output_line and "decompression e2e throughput" not in output_line:
            output_line_split = output_line.split()
            index = output_line_split.index("speed:") + 1
            compression_throughput = float(output_line_split[index])
            df.loc[(method, error_bound), 'Comp_Throughput (GB/s)'] = compression_throughput
        if "cuSZp decompression end-to-end speed" in output_line:
            output_line_split = output_line.split()
            index = output_line_split.index("speed:") + 1
            decompression_throughput = float(output_line_split[index])
            df.loc[(method, error_bound), 'Decomp_Throughput (GB/s)'] = decompression_throughput

def evaluate_cuSZx(command, error_bound):
    #cuSZx
    method = 'cuSZx'
    result = subprocess.run(command, capture_output=True, text=True)
    output_lines = result.stdout.splitlines()
    
    for output_line in output_lines:
        # print(output_line)
        if "CR" in output_line:
            output_line_split = output_line.split()
            index = output_line_split.index("CR") + 2
            compression_ratio_value = float(output_line_split[index])
            df.loc[(method, error_bound), 'CR'] = compression_ratio_value
        if "szx::compress_float" in output_line:
            output_line_split = output_line.split()
            compressed_time = float(output_line_split[1].replace(",","")) / (float(output_line_split[0]) * 1e-2)
            compression_throughput = compute_throughput(compressed_time)
            df.loc[(method, error_bound), 'Comp_Throughput_nsys (GB/s)'] = compression_throughput
            

def evaluate_cuzfp(command_comp, command_decomp, command_psnr, error_bound):
    # cuSZ
    method = 'cuzfp'
    
    # compression
    result = subprocess.run(command_comp, capture_output=True, text=True)
    output_lines = result.stderr.splitlines()
    nsys_lines = result.stdout.splitlines()
    
    for line in output_lines:
        if "ratio=" in line:
            output_line_split = line.split()
            index = 7
            CR = float(output_line_split[index][6:])
            df.loc[(method, error_bound), 'CR'] = CR
    
    for line in nsys_lines:
        # print(output_line)
        # print(line)
        if "cuZFP::cudaEncode<float>" in line:
            nsys_line_split = line.split()
            compressed_time = float(nsys_line_split[1].replace(",","")) / (float(nsys_line_split[0]) * 1e-2)
            compression_throughput = compute_throughput(compressed_time)
            df.loc[(method, error_bound), 'Comp_Throughput_nsys (GB/s)'] = compression_throughput
    # decompression
    decomp_result = subprocess.run(command_decomp, capture_output=True, text=True)
    output_lines = decomp_result.stderr.splitlines()
    nsys_lines = decomp_result.stdout.splitlines()
    
    for line in nsys_lines:
        # print(line)
        if "cuZFP::cudaDecode3" in line:
            nsys_line_split = line.split()
            decompressed_time = float(nsys_line_split[1].replace(",","")) / (float(nsys_line_split[0]) * 1e-2)
            decompression_throughput = compute_throughput(decompressed_time)
            df.loc[(method, error_bound), 'Decomp_Throughput_nsys (GB/s)'] = decompression_throughput
    psnr_result = subprocess.run(command_psnr, capture_output=True, text=True)
    output_lines = psnr_result.stdout.splitlines()
    for output_line in output_lines:
        # print(output_line)
        output_line_split = output_line.split()
        if 'PSNR' in output_line:
            output_line_split = output_line.split()
            index = output_line_split.index("PSNR") + 2
            PSNR_value = float(output_line_split[index].replace(",", ""))
            df.loc[(method, error_bound), 'PSNR'] = PSNR_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--output', '-o', type=str)
    parser.add_argument('--dim', '-d', type=int,default=3)
    parser.add_argument('--dims', '-m', type=str,nargs="+")
    parser.add_argument('--cmp', '-c', type=str,nargs="*")
    parser.add_argument('--eb', '-e', type=str,nargs="*")
    parser.add_argument('--br', '-b', type=str,nargs="*")
    args = parser.parse_args()
    
    datafolder   = args.input
    outputfolder = args.output
    data_size    = args.dims
    cmp_list     = args.cmp
    eb_list      = args.eb
    br_list      = args.br
    
    
    method_list = ['FZGPU', 'cuSZ', 'cuSZp', 'cuzfp', 'cuSZx', 'cuSZi']
    error_bound_list = ['1e-2', '5e-3', '1e-3','5e-4', '1e-4', '5e-5', '1e-5']
    bit_rate_list = ['0.5', '1', '2', '4', '6', '8', '12', '16']
    run_func_dict = {"FZGPU":run_FZGPU, "cuSZ":run_cuSZ, "cuSZp":run_cuSZp, "cuSZx":run_cuSZx, "cuzfp":run_cuzfp, "cuSZi":run_cuSZ,}
    
    cmp_list = method_list      if cmp_list is None else cmp_list
    eb_list  = error_bound_list if eb_list is None else eb_list
    br_list  = bit_rate_list    if br_list is None else br_list
    
    
    datafiles=os.listdir(datafolder)
    datafiles=[file for file in datafiles if file.split(".")[-1]=="dat" or file.split(".")[-1]=="f32" or file.split(".")[-1]=="bin"]
    
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    
    
    for cmp in cmp_list:    
        if cmp != 'cuzfp':
            for file in tqdm(datafiles):
                for eb in eb_list:
                    data_path = os.path.join(datafolder, file)
                    file_path = os.path.join(outputfolder, file)
                    cmd, cmd_nvcomp, cmd_bitcomp = update_command(cmp, data_path, data_size, error_bound=eb)
                    run_func_dict[cmp](cmd, cmd_nvcomp, cmd_bitcomp, file_path + "_eb=" + eb + "_" + cmp)
                    
        else:
            for file in tqdm(datafiles):
                for br in br_list:
                    data_path = os.path.join(datafolder, file)
                    file_path = os.path.join(outputfolder, file)
                    cmd, cmd_nvcomp, cmd_bitcomp = update_command(cmp, data_path, data_size, bit_rate=br)
                    run_func_dict[cmp](cmd, cmd_nvcomp, cmd_bitcomp, file_path + "_br=" + br + "_" + cmp)
                    # run_cuzfp(command['cuzfp'], command['nvcomp_bitcomp'], command['boyuan_bitcomp'], file_path + "_bitrate=" + bit_rate + '_' + method_list[3])
           
            # print(error_bound)
            # command = update_command(data_path, error_bound)
            # evaluate_FZGPU(command[0], error_bound)
            # print("fZ_FINISHED")
            # evaluate_cuSZ(command[1], command[2], error_bound)
            # print("CUSZ_FINISHED")
            # evaluate_cuSZp(command[3], error_bound)
            # print("CUSZp_FINISHED")
            # evaluate_cuzfp(command[4],command[5], command[6], error_bound)
            # print("CUZFP_FINISHED")
            # evaluate_cuSZx(command[7], error_bound)
            # print("CUSZX_FINISHED")
        # df.to_csv(fwork_path + "csv/aramco-snapshot-{snapshot:04d}.f32.csv", index=True)