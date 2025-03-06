import os
import sys
import copy
import argparse
import subprocess
from tqdm import tqdm
import pandas as pd
from scipy.stats import hmean
from statistics import mean
from math import sqrt, log10


class Analysis:
    def __init__(self, data_folder, output_folder, data_dims, data_type='f32', data_type_size=4, cmp_list=None, eb_list=None, br_list=None,
                 dataset=None, machine=None):
        """
        Initialize the Analysis class with specified parameters.
        
        :param data_folder: Folder containing the data to be analyzed
        :param data_dims: Dimensions of the data
        :param data_type: Data type
        :param data_type_size: Data type size in byte
        :param cmp_list: List of compression methods to analyze
        :param eb_list: List of error bounds for analysis
        :param br_list: List of bit rate for analysis
        """
        self.data_folder = data_folder
        self.data_dimensions = data_dims
        self.data_size = float(data_dims[0]) * float(data_dims[1]) * float(data_dims[2])
        self.data_type = data_type
        self.data_type_size = data_type_size
        self.cmp_list = cmp_list or ['FZGPU', 'cuSZ', 'cuSZp', 'cuzfp', 'cuSZx', 'cuSZi']
        self.eb_list = eb_list or ['1e-2', '5e-3', '1e-3', '5e-4', '1e-4', '5e-5', '1e-5']
        self.br_list = br_list or ['0.5', '1', '2', '4', '6', '8', '12', '16']

        self.datafiles = os.listdir(self.data_folder)
        
        #self.datapoint_list = list(set([file.split(".")[0] for file in self.datafiles]))
        self.datapoint_list = list(set([file.split("=")[0] for file in self.datafiles]))
        self.datapoint_list = list(set([x[:len(x)-3] for x in self.datapoint_list]))
        self.datapoint_list.append('_overall')
        self.output_folder = output_folder
        self.machine = machine
        self.dataset = dataset
        # Mapping methods for analysis types
        self.analyze_functions = {
            'FZGPU': self.analyze_FZGPU,
            'cuSZ' : self.analyze_cuSZ,
            'cuSZx': self.analyze_cuSZx,
            'cuSZp': self.analyze_cuSZp,
            'cuSZi': self.analyze_cuSZi,
            'cuzfp': self.analyze_cuzfp,
            'cuZFP': self.analyze_cuzfp,
            # Add other mappings as necessary
        }
        # self.metrics = ['CR', 'BR', 'PSNR', 'NRMSE', 'cmp_cTP', 'cmp_xTP', 'nsys_cmp_cTP', 'nsys_cmp_xTP', 
        #                 'nvcomp_CR', 'nvcomp_cTP', 'nvcomp_xTP', 'bitcomp_CR', 'bitcomp_cTP', 'bitcomp_xTP',]
        self.metrics = ['CR', 'BR', 'PSNR', 'NRMSE', 'cmp_cTP', 'cmp_xTP', 'nsys_cmp_cTP', 'nsys_cmp_xTP']

        self.df = {}
        self.df_overall = {}
    
    def launch_analysis(self,):
        for cmp in self.cmp_list:
            self.analyze_functions[cmp]()
    
    def save_to_csv(self,):
        for cmp in self.cmp_list:
            self.df[cmp].to_csv(os.path.join(outputfolder, f"{cmp}_{self.dataset}_{self.machine}.csv"),  sep=',', index=True)
            self.df_overall[cmp].to_csv(os.path.join(outputfolder, f"{cmp}_{self.dataset}_{self.machine}_overall.csv"),  sep=',', index=True)
            print(cmp)
            #print(self.df_overall[cmp])
            
    def extract_overall(self, df):
        overall_df = df.xs('_overall', level='Data_Point').copy()
        overall_df.index = overall_df.index.map(float)
        overall_df.sort_index(inplace=True)
        overall_df.index = overall_df.index.map('{:0.1e}'.format)
        return overall_df
        
    def compute_throughput(self, data_size, elapsed_time):
        return (float(data_size) * self.data_type_size) /  ((2 ** 30) * elapsed_time * 1e-9)
    
    def compute_overall(self, df, eb):
        # CR
        non_overall_values = df.loc[eb, 'CR'].drop((eb, '_overall'), errors='ignore')
        harmonic_mean = hmean(non_overall_values.dropna().values) if len(non_overall_values.dropna().values) != 0 else None
        df.loc[(eb, '_overall'), 'CR'] = harmonic_mean
        
        
        # non_overall_values = df.loc[eb, 'nvcomp_CR'].drop((eb, '_overall'), errors='ignore')
        # harmonic_mean = hmean(non_overall_values.dropna().values) if len(non_overall_values.dropna().values) != 0 else None
        # df.loc[(eb, '_overall'), 'nvcomp_CR'] = harmonic_mean
        
        # non_overall_values = df.loc[eb, 'bitcomp_CR'].drop((eb, '_overall'), errors='ignore')
        # harmonic_mean = hmean(non_overall_values.dropna().values) if len(non_overall_values.dropna().values) != 0 else None
        # df.loc[(eb, '_overall'), 'bitcomp_CR'] = harmonic_mean
        
        # BR
        non_overall_values = df.loc[eb, 'BR'].drop((eb, '_overall'), errors='ignore')
        arithmetic_mean = mean(non_overall_values.dropna().values) if len(non_overall_values.dropna().values) != 0 else None
        df.loc[(eb, '_overall'), 'BR'] = arithmetic_mean
        
        # PSNR
        non_overall_values = df.loc[eb, 'NRMSE'].drop((eb, '_overall'), errors='ignore')
        NRMSE_list = non_overall_values.dropna().values
        PSNR_avg = 0
        if len(NRMSE_list) != 0:
            for NRMSE in NRMSE_list:
                PSNR_avg += NRMSE ** 2
            PSNR_avg /= len(NRMSE_list)
        
            PSNR_avg = -20.0 * log10(sqrt(PSNR_avg))
            df.loc[(eb, '_overall'), 'PSNR'] = PSNR_avg
        
        # TP
        TP_list = [TP for TP in self.metrics if 'TP' in TP]
        for TP in TP_list:
            non_overall_values = df.loc[eb, TP].drop((eb, '_overall'), errors='ignore')
            arithmetic_mean = mean(non_overall_values.dropna().values) if len(non_overall_values.dropna().values) != 0 else None
            df.loc[(eb, '_overall'), TP] = arithmetic_mean
        
        
    
    def analyze_compareData(self, lines, df, df_loc):
        """
        Get PSNR and NRMSE from compareData output
        
        :param lines: compareData output
        :param df: dataframe to save the PSNR and NRMSE
        :param df_loc: location in dataframe to save the PSNR and NRMSE
        """
        for line in lines:
            line_split = line.split()
            if 'PSNR' in line:
                # 0    1    2         3   4    5
                # PSNR = 47.661551, NRMSE = 0.0041392573250867668866
                df.loc[df_loc, 'PSNR'] = float(line_split[2][:-1])
                df.loc[df_loc, 'NRMSE'] = float(line_split[5])
                break
        return
    
    def analyze_nvcomp(self, lines, df, df_loc, data_size):
        """
        Get CR, cTP, xTP from nsys nvcomp output
        
        :param lines: compareData output
        :param df: dataframe to save the CR, cTP, xTP
        :param df_loc: location to save in dataframe
        :param data_size: data_size of intput compressed file
        """
        nsys_line_number = []
        for line_number, line in enumerate(lines):
            line_split = line.split()
            if 'compressed ratio' in line:
                # comp_size: 3015288, compressed ratio: 1.4839
                df.loc[df_loc, 'nvcomp_CR'] = float(line_split[-1])
            if (("CUDA Kernel Statistics" in line) or ("cuda_gpu_kern_sum" in line) or ("gpukernsum" in line)):
                nsys_line_number.append(line_number)
            if ((("CUDA Memory Operation Statistics (by time):" in line) or ("cuda_gpu_mem_time_sum" in line) or ("gpumemtimesum" in line)) or "Operating System Runtime API Statistics:" in line):
                nsys_line_number.append(line_number)
        if len(nsys_line_number) >= 2:
            self.analyze_nsys(lines[nsys_line_number[0]:nsys_line_number[1]], df, df_loc, 
                            'nvcomp_cTP', data_size, ["bitcomp::batch_encoder_kernel"])
            self.analyze_nsys(lines[nsys_line_number[0]:nsys_line_number[1]], df, df_loc, 
                            'nvcomp_xTP', data_size, ["bitcomp::batch_decoder_kernel"])
        return
    
    def analyze_bitcomp(self, lines, df, df_loc, data_size):
        """
        Get CR, cTP, xTP from nsys bitcomp output
        
        :param lines: compareData output
        :param df: dataframe to save the CR, cTP, xTP
        :param df_loc: location to save in dataframe
        :param data_size: data_size of intput compressed file
        """
        nsys_line_number = []
        for line_number, line in enumerate(lines):
            line_split = line.split()
            if 'Compression ratio' in line:
                # Compression ratio = 1.49
                df.loc[df_loc, 'bitcomp_CR'] = float(line_split[-1])
            if (("CUDA Kernel Statistics" in line) or ("cuda_gpu_kern_sum" in line) or ("gpukernsum" in line)):
                nsys_line_number.append(line_number)
            if ((("CUDA Memory Operation Statistics (by time):" in line) or ("cuda_gpu_mem_time_sum" in line) or ("gpumemtimesum" in line)) or "Operating System Runtime API Statistics:" in line):
                nsys_line_number.append(line_number)
        if len(nsys_line_number) >= 2:
            self.analyze_nsys(lines[nsys_line_number[0]:nsys_line_number[1]], df, df_loc, 
                            'bitcomp_cTP', data_size, ["bitcomp::encoder_kernel"])
            self.analyze_nsys(lines[nsys_line_number[0]:nsys_line_number[1]], df, df_loc, 
                            'bitcomp_xTP', data_size, ["bitcomp::decoder_kernel"])
        return
    
    def analyze_nsys(self, lines, df, df_loc, metric, data_size, func_names, statics=0):
        """
        Get PSNR and NRMSE from nsys output
        
        :param lines: nsys output
        :param df: dataframe to save the throughput
        :param df_loc: location in dataframe to save the throughput
        :param metric: corresponded throughput name
        :param data_size: data size, to compute throughput
        :param func_names: compression kernel to be counted
        :param statics: 0 for average, 1 for minimum, 2 for maximum, default 0
        
        Time(%)  Total Time (ns)  Instances  Average (ns)  Minimum (ns)  Maximum (ns)  StdDev (ns)                                                  Name                                                
        -------  ---------------  ---------  ------------  ------------  ------------  -----------  ----------------------------------------------------------------------------------------------------
        60.1           30,592          1      30,592.0        30,592        30,592          0.0  void bitcomp::decoder_kernel<(bitcompAlgorithm_t)0, unsigned char, (bitcompMode_t)0, (bitcompIntFor…
        39.9           20,352          1      20,352.0        20,352        20,352          0.0  void bitcomp::encoder_kernel<(bitcompAlgorithm_t)0, unsigned char, (bitcompDataType_t)0, (bitcompMo…
        """
        
        time = 0
        time = 0
        for line in lines:
            line_split = line.split()
            if 'Time(%)' in line or len(line_split) == 0 or  '----' in line_split[0]:
                continue
            for name in func_names:
                if name in line:
                    time += float(line_split[statics + 3].replace(",",""))
        df.loc[df_loc, metric] = self.compute_throughput(data_size, time)
        return

    def analyze_FZGPU(self, ):
        # Create the DataFrame with MultiIndex
        cmp = "FZGPU"
        self.index = pd.MultiIndex.from_product(
            [self.eb_list, self.datapoint_list],
            names=['Error_Bound', 'Data_Point']
        )
        df = pd.DataFrame(index=self.index, columns=self.metrics).sort_index()
        for eb in self.eb_list:
            for data_point in self.datapoint_list[:-1]:
                file_path =  os.path.join(self.data_folder, f"{data_point}_eb={eb}_{cmp}")
                try: 
                    file = open(file_path, 'r')
                    file_content = file.read()
                    lines = file_content.splitlines()
                    compareDATA_line_number = []
                    nvcomp_line_number = []
                    bitcomp_line_number = []
                    fzgpu_line_number = []
                    nsys_line_number = []
                    for line_number, line in enumerate(lines):
                        line_split = line.split()
                        if "compression ratio" in line:
                            index = line_split.index("ratio:") + 1
                            compression_ratio_value = float(line_split[index])
                            df.loc[(eb, data_point), 'CR'] = compression_ratio_value
                            df.loc[(eb, data_point), 'BR'] = 32.0 / compression_ratio_value
                        if "compression e2e throughput" in line and "decompression e2e throughput" not in line:
                            index = line_split.index("throughput:") + 1
                            compression_throughput = float(line_split[index])
                            df.loc[(eb, data_point), 'cmp_cTP'] = compression_throughput
                        if "decompression e2e throughput" in line:
                            index = line_split.index("throughput:") + 1
                            decompression_throughput = float(line_split[index])
                            df.loc[(eb, data_point), 'cmp_xTP'] = decompression_throughput
                        if "-fzgpu-" in line:
                            fzgpu_line_number.append(line_number)
                        if (("CUDA Kernel Statistics" in line) or ("cuda_gpu_kern_sum" in line) or ("gpukernsum" in line)) and len(fzgpu_line_number) == 1:
                            nsys_line_number.append(line_number)
                        if (("CUDA Memory Operation Statistics (by time):" in line) or ("cuda_gpu_mem_time_sum" in line) or ("gpumemtimesum" in line)) and len(fzgpu_line_number) == 1:
                            nsys_line_number.append(line_number)
                        
                        if "-compareData-" in line:
                            compareDATA_line_number.append(line_number)
                        if "-nvcomp-" in line:
                            nvcomp_line_number.append(line_number)
                        if "-bitcomp-" in line:
                            bitcomp_line_number.append(line_number)
                        
                    self.analyze_nsys(lines[nsys_line_number[0]:nsys_line_number[1]], df, (eb, data_point), 
                          'nsys_cmp_cTP', self.data_size, [" compressionFusedKernel", 
                                        "void cusz::experimental::c_lorenzo_3d1l_32x8x8data_mapto32x1x8"])
                    self.analyze_nsys(lines[nsys_line_number[0]:nsys_line_number[1]], df, (eb, data_point), 
                          'nsys_cmp_xTP', self.data_size, [" decompressionFusedKernel", 
                                        "void cusz::experimental::x_lorenzo_3d1l_32x8x8data_mapto32x1x8"])
                    compressed_size = self.data_size / compression_ratio_value
                    self.analyze_compareData(lines[compareDATA_line_number[0]:compareDATA_line_number[1]], df, (eb, data_point))
                    self.analyze_nvcomp(lines[nvcomp_line_number[0]:nvcomp_line_number[1]], df, (eb, data_point), compressed_size)
                    self.analyze_bitcomp(lines[bitcomp_line_number[0]:bitcomp_line_number[1]], df, (eb, data_point), compressed_size)
                    # # assert 0
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print(file_path)
                    # assert 0
                    pass
            self.compute_overall(df, eb)
        
        self.df[cmp] = df
        self.df_overall[cmp] = self.extract_overall(df)
        
                

    def analyze_cuSZ(self,):

        cmp = "cuSZ"
        self.index = pd.MultiIndex.from_product(
            [self.eb_list, self.datapoint_list],
            names=['Error_Bound', 'Data_Point']
        )
        df = pd.DataFrame(index=self.index, columns=self.metrics).sort_index()    
        for eb in self.eb_list:
            for data_point in self.datapoint_list[:-1]:
                file_path =  os.path.join(self.data_folder, f"{data_point}_eb={eb}_{cmp}")
                try: 
                    file = open(file_path, 'r')
                    file_content = file.read()
                    lines = file_content.splitlines()
                    compareDATA_line_number = []
                    nvcomp_line_number = []
                    bitcomp_line_number = []
                    cusz_comp_line_number = []
                    cusz_decomp_line_number = []
                    nsys_comp_line_number = []
                    nsys_decomp_line_number = []
                    for line_number, line in enumerate(lines):
                        line_split = line.split()
                        # compression
                        if "(total)" in line:
                            index = line_split.index("(total)") + 2
                            compression_throughput = float(line_split[index])
                            df.loc[(eb, data_point), 'cmp_cTP'] = compression_throughput
                        if "-cusz_compress-" in line:
                            cusz_comp_line_number.append(line_number)
                        if (("CUDA Kernel Statistics" in line) or ("cuda_gpu_kern_sum" in line) or ("gpukernsum" in line)) and len(cusz_comp_line_number) == 1:
                            nsys_comp_line_number.append(line_number)
                        if (("CUDA Memory Operation Statistics (by time):" in line) or ("cuda_gpu_mem_time_sum" in line) or ("gpumemtimesum" in line)) and len(cusz_comp_line_number) == 1:
                            nsys_comp_line_number.append(line_number)
                        
                        # decompression
                    
                        if 'metrics' in line and line_split[0] == 'metrics':
                            # CR
                            index = line_split.index("metrics") + 1
                            compression_ratio_value = float(line_split[index])
                            df.loc[(eb, data_point), 'CR'] = compression_ratio_value
                            df.loc[(eb, data_point), 'BR'] = 32.0 / compression_ratio_value
                        if "(total)" in line:
                            index = line_split.index("(total)") + 2
                            decompression_throughput = float(line_split[index])
                            df.loc[(eb, data_point), 'cmp_xTP'] = decompression_throughput
                        if "-cusz_decompress-" in line:
                            cusz_decomp_line_number.append(line_number)
                        if (("CUDA Kernel Statistics" in line) or ("cuda_gpu_kern_sum" in line) or ("gpukernsum" in line)) and len(cusz_decomp_line_number) == 1:
                            nsys_decomp_line_number.append(line_number)
                        if (("CUDA Memory Operation Statistics (by time):" in line) or ("cuda_gpu_mem_time_sum" in line) or ("gpumemtimesum" in line)) and len(cusz_decomp_line_number) == 1:
                            nsys_decomp_line_number.append(line_number)
                        
                        
                        if "-compareData-" in line:
                            compareDATA_line_number.append(line_number)
                        if "-nvcomp-" in line:
                            nvcomp_line_number.append(line_number)
                        if "-bitcomp-" in line:
                            bitcomp_line_number.append(line_number)
                    self.analyze_nsys(lines[nsys_comp_line_number[0]:nsys_comp_line_number[1]], df, (eb, data_point), 
                          'nsys_cmp_cTP', self.data_size, [" psz::detail::hf_encode_phase2_deflate", "histsp_multiwarp", 
                                        "psz::detail::hf_encode_phase1_fill", "psz::rolling::c_lorenzo_3d1l", "psz::detail::hf_encode_phase4_concatenate"])
                    self.analyze_nsys(lines[nsys_decomp_line_number[0]:nsys_decomp_line_number[1]], df, (eb, data_point), 
                          'nsys_cmp_xTP', self.data_size, [" hf_decode_kernel", 
                                        "psz::cuda_hip::__kernel::x_lorenzo_3d1l<"])
                    compressed_size = self.data_size / compression_ratio_value
                    self.analyze_compareData(lines[compareDATA_line_number[0]:compareDATA_line_number[1]], df, (eb, data_point))
                    self.analyze_nvcomp(lines[nvcomp_line_number[0]:nvcomp_line_number[1]], df, (eb, data_point), compressed_size)
                    self.analyze_bitcomp(lines[bitcomp_line_number[0]:bitcomp_line_number[1]], df, (eb, data_point), compressed_size)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print(file_path)
                    # assert 0
                    pass
            self.compute_overall(df, eb)
        
        self.df[cmp] = df
        self.df_overall[cmp] = self.extract_overall(df)
        
    def analyze_cuSZi(self,):

        cmp = "cuSZi"
        self.index = pd.MultiIndex.from_product(
            [self.eb_list, self.datapoint_list],
            names=['Error_Bound', 'Data_Point']
        )
        df = pd.DataFrame(index=self.index, columns=self.metrics).sort_index()    
        for eb in self.eb_list:
            for data_point in self.datapoint_list[:-1]:
                file_path =  os.path.join(self.data_folder, f"{data_point}_eb={eb}_{cmp}")
                try: 
                    file = open(file_path, 'r')
                    file_content = file.read()
                    lines = file_content.splitlines()

                    # Check if file has both compression and decompression reports
                    has_compression = False
                    has_decompression = False
                    for line in lines:
                        if line.startswith("(c)"):
                            has_compression = True
                        elif line.startswith("(d)"):
                            has_decompression = True
                        if has_compression and has_decompression:
                            break
                    if not has_compression or not has_decompression:
                        # Skip if missing either compression or decompression report
                        continue

                    compareDATA_line_number = []
                    # nvcomp_line_number = []
                    # bitcomp_line_number = []
                    cusz_comp_line_number = []
                    cusz_decomp_line_number = []
                    nsys_comp_line_number = []
                    nsys_decomp_line_number = []
                    for line_number, line in enumerate(lines):
                        line_split = line.split()
                        # compression
                        if "(total)" in line and line_number > 0 and "difflog" in lines[line_number-1]:
                            index = line_split.index("(total)") + 2
                            compression_throughput = float(line_split[index])
                            df.loc[(eb, data_point), 'cmp_cTP'] = compression_throughput
                        if "-cusz_compress-" in line:
                            cusz_comp_line_number.append(line_number)
                        if (("CUDA Kernel Statistics" in line) or ("cuda_gpu_kern_sum" in line) or ("gpukernsum" in line)) and len(cusz_comp_line_number) == 1:
                            nsys_comp_line_number.append(line_number)
                        if (("CUDA Memory Operation Statistics (by time):" in line) or ("cuda_gpu_mem_time_sum" in line) or ("gpumemtimesum" in line)) and len(cusz_comp_line_number) == 1:
                            nsys_comp_line_number.append(line_number)
                        
                        # decompression
                    
                        if 'metrics' in line and line_split[0] == 'metrics':
                            # CR
                            index = line_split.index("metrics") + 1
                            compression_ratio_value = float(line_split[index])
                            df.loc[(eb, data_point), 'CR'] = compression_ratio_value
                            df.loc[(eb, data_point), 'BR'] = 32.0 / compression_ratio_value
                        if "(total)" in line and line_number > 0 and "predict" in lines[line_number-1]:
                            index = line_split.index("(total)") + 2
                            decompression_throughput = float(line_split[index])
                            df.loc[(eb, data_point), 'cmp_xTP'] = decompression_throughput
                        if "-cusz_decompress-" in line:
                            cusz_decomp_line_number.append(line_number)
                        if (("CUDA Kernel Statistics" in line) or ("cuda_gpu_kern_sum" in line) or ("gpukernsum" in line)) and len(cusz_decomp_line_number) == 1:
                            nsys_decomp_line_number.append(line_number)
                        if (("CUDA Memory Operation Statistics (by time):" in line) or ("cuda_gpu_mem_time_sum" in line) or ("gpumemtimesum" in line)) and len(cusz_decomp_line_number) == 1:
                            nsys_decomp_line_number.append(line_number)
                        
                        
                        if "-compareData-" in line:
                            compareDATA_line_number.append(line_number)
                        # if "-nvcomp-" in line:
                        #     nvcomp_line_number.append(line_number)
                        # if "-bitcomp-" in line:
                        #     bitcomp_line_number.append(line_number)
                    self.analyze_nsys(lines[nsys_comp_line_number[0]:nsys_comp_line_number[1]], df, (eb, data_point), 
                          'nsys_cmp_cTP', self.data_size, ["cusz::c_spline3d_infprecis_16x16x16data", "cusz::c_spline3d_profiling_data_2", "d_encode", "psz::extrema_kernel"])
                    self.analyze_nsys(lines[nsys_decomp_line_number[0]:nsys_decomp_line_number[1]], df, (eb, data_point), 
                          'nsys_cmp_xTP', self.data_size, ["d_decode", "cusz::x_spline3d_infprecis_16x16x16data", "psz::extrema_kernel", "psz::cu_hip::spvn_scatter"])
                    # compressed_size = self.data_size / compression_ratio_value
                    self.analyze_compareData(lines[compareDATA_line_number[0]:compareDATA_line_number[1]], df, (eb, data_point))
                    # self.analyze_nvcomp(lines[nvcomp_line_number[0]:nvcomp_line_number[1]], df, (eb, data_point), compressed_size)
                    # self.analyze_bitcomp(lines[bitcomp_line_number[0]:bitcomp_line_number[1]], df, (eb, data_point), compressed_size)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print(file_path)
                    # assert 0
                    pass
            self.compute_overall(df, eb)
        self.df[cmp] = df
        self.df_overall[cmp] = self.extract_overall(df)
        

    def analyze_cuSZp(self, ):
        # Create the DataFrame with MultiIndex
        cmp = "cuSZp"
        self.index = pd.MultiIndex.from_product(
            [self.eb_list, self.datapoint_list],
            names=['Error_Bound', 'Data_Point']
        )
        df = pd.DataFrame(index=self.index, columns=self.metrics).sort_index()
        for eb in self.eb_list:
            for data_point in self.datapoint_list[:-1]:
                file_path =  os.path.join(self.data_folder, f"{data_point}_eb={eb}_{cmp}")
                try: 
                    file = open(file_path, 'r')
                    file_content = file.read()
                    lines = file_content.splitlines()
                    compareDATA_line_number = []
                    nvcomp_line_number = []
                    bitcomp_line_number = []
                    cuSZp_line_number = []
                    nsys_line_number = []
                    for line_number, line in enumerate(lines):
                        line_split = line.split()
                        if "compression ratio" in line:
                            line_split = line.split()
                            index = line_split.index("ratio:") + 1
                            compression_ratio_value = float(line_split[index])
                            df.loc[(eb, data_point), 'CR'] = compression_ratio_value
                            df.loc[(eb, data_point), 'BR'] = 32.0 / compression_ratio_value
                        if "cuSZp compression   end-to-end speed" in line and "decompression e2e throughput" not in line:
                            line_split = line.split()
                            index = line_split.index("speed:") + 1
                            compression_throughput = float(line_split[index])
                            df.loc[(eb, data_point), 'cmp_cTP'] = compression_throughput
                        if "cuSZp decompression end-to-end speed" in line:
                            line_split = line.split()
                            index = line_split.index("speed:") + 1
                            decompression_throughput = float(line_split[index])
                            df.loc[(eb, data_point), 'cmp_xTP'] = decompression_throughput
                        if "-cuszp_compress-" in line:
                            cuSZp_line_number.append(line_number)
                        if (("CUDA Kernel Statistics" in line) or ("cuda_gpu_kern_sum" in line) or ("gpukernsum" in line)) and len(cuSZp_line_number) == 1:
                            nsys_line_number.append(line_number)
                        if (("CUDA Memory Operation Statistics (by time):" in line) or ("cuda_gpu_mem_time_sum" in line) or ("gpumemtimesum" in line)) and len(cuSZp_line_number) == 1:
                            nsys_line_number.append(line_number)
                        
                        if "-compareData-" in line:
                            compareDATA_line_number.append(line_number)
                        if "-nvcomp-" in line:
                            nvcomp_line_number.append(line_number)
                        if "-bitcomp-" in line:
                            bitcomp_line_number.append(line_number)
                        
                    self.analyze_nsys(lines[nsys_line_number[0]:nsys_line_number[1]], df, (eb, data_point), 
                          'nsys_cmp_cTP', self.data_size, ["SZp_compress_kernel_f32"])
                    self.analyze_nsys(lines[nsys_line_number[0]:nsys_line_number[1]], df, (eb, data_point), 
                          'nsys_cmp_xTP', self.data_size, ["SZp_decompress_kernel_f32"])
                    compressed_size = self.data_size / compression_ratio_value
                    self.analyze_compareData(lines[compareDATA_line_number[0]:compareDATA_line_number[1]], df, (eb, data_point))
                    self.analyze_nvcomp(lines[nvcomp_line_number[0]:nvcomp_line_number[1]], df, (eb, data_point), compressed_size)
                    self.analyze_bitcomp(lines[bitcomp_line_number[0]:bitcomp_line_number[1]], df, (eb, data_point), compressed_size)
                    # # assert 0
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print(file_path)
                    # assert 0
                    pass
            
            self.compute_overall(df, eb)
        
        self.df[cmp] = df
        self.df_overall[cmp] = self.extract_overall(df)

    def analyze_cuzfp(self,):

        cmp = "cuzfp"
        self.index = pd.MultiIndex.from_product(
            [self.br_list, self.datapoint_list],
            names=['Bit_Rate', 'Data_Point']
        )
        df = pd.DataFrame(index=self.index, columns=self.metrics).sort_index()    
        for br in self.br_list:
            
            for data_point in self.datapoint_list[:-1]:
                file_path =  os.path.join(self.data_folder, f"{data_point}_br={br}_{cmp}")
                try: 
                    file = open(file_path, 'r')
                    file_content = file.read()
                    lines = file_content.splitlines()
                    compareDATA_line_number = []
                    nvcomp_line_number = []
                    bitcomp_line_number = []
                    cuzfp_comp_line_number = []
                    cuzfp_decomp_line_number = []
                    nsys_comp_line_number = []
                    nsys_decomp_line_number = []
                    df.loc[(br, data_point), 'BR'] = float(br)
                    
                    for line_number, line in enumerate(lines):
                        line_split = line.split()
                        if "zfp=" in line:
                            compressed_size = float(line_split[6][4:]) / self.data_type_size
                            df.loc[(br, data_point), 'CR'] =  self.data_size / compressed_size
                        # compression
                        if "-nsys compress-" in line:
                            cuzfp_comp_line_number.append(line_number)
                        if (("CUDA Kernel Statistics" in line) or ("cuda_gpu_kern_sum" in line) or ("gpukernsum" in line)) and len(cuzfp_comp_line_number) == 1:
                            nsys_comp_line_number.append(line_number)
                        if (("CUDA Memory Operation Statistics (by time):" in line) or ("cuda_gpu_mem_time_sum" in line) or ("gpumemtimesum" in line)) and len(cuzfp_comp_line_number) == 1:
                            nsys_comp_line_number.append(line_number)
                        
                        # decompression
                        
                        if "-nsys decompress-" in line:
                            cuzfp_decomp_line_number.append(line_number)
                        if (("CUDA Kernel Statistics" in line) or ("cuda_gpu_kern_sum" in line) or ("gpukernsum" in line)) and len(cuzfp_decomp_line_number) == 1:
                            nsys_decomp_line_number.append(line_number)
                        if (("CUDA Memory Operation Statistics (by time):" in line) or ("cuda_gpu_mem_time_sum" in line) or ("gpumemtimesum" in line)) and len(cuzfp_decomp_line_number) == 1:
                            nsys_decomp_line_number.append(line_number)
                        
                        
                        if "-compareData-" in line:
                            compareDATA_line_number.append(line_number)
                        if "-nvcomp-" in line:
                            nvcomp_line_number.append(line_number)
                        if "-bitcomp-" in line:
                            bitcomp_line_number.append(line_number)
                    self.analyze_nsys(lines[nsys_comp_line_number[0]:nsys_comp_line_number[1]], df, (br, data_point), 
                          'nsys_cmp_cTP', self.data_size, ["cuZFP::cudaEncode"])
                    self.analyze_nsys(lines[nsys_decomp_line_number[0]:nsys_decomp_line_number[1]], df, (br, data_point), 
                          'nsys_cmp_xTP', self.data_size, ["cuZFP::cudaDecode3"])
                    self.analyze_compareData(lines[compareDATA_line_number[0]:compareDATA_line_number[1]], df, (br, data_point))
                    self.analyze_nvcomp(lines[nvcomp_line_number[0]:nvcomp_line_number[1]], df, (br, data_point), compressed_size)
                    self.analyze_bitcomp(lines[bitcomp_line_number[0]:bitcomp_line_number[1]], df, (br, data_point), compressed_size)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print(file_path)
                    # assert 0
                    pass
            self.compute_overall(df, br)
        
        self.df[cmp] = df
        self.df_overall[cmp] = self.extract_overall(df)
                
    def analyze_cuSZx(self,):

        cmp = "cuSZx"
        self.index = pd.MultiIndex.from_product(
            [self.eb_list, self.datapoint_list],
            names=['Error_Bound', 'Data_Point']
        )
        df = pd.DataFrame(index=self.index, columns=self.metrics).sort_index()    
        for eb in self.eb_list:
            for data_point in self.datapoint_list[:-1]:
                file_path =  os.path.join(self.data_folder, f"{data_point}_eb={eb}_{cmp}")
                try: 
                    file = open(file_path, 'r')
                    file_content = file.read()
                    lines = file_content.splitlines()
                    compareDATA_line_number = []
                    nvcomp_line_number = []
                    bitcomp_line_number = []
                    cusz_comp_line_number = []
                    cusz_decomp_line_number = []
                    nsys_comp_line_number = []
                    nsys_decomp_line_number = []
                    for line_number, line in enumerate(lines):
                        line_split = line.split()
                        # compression
                        if "-cuszx_compress-" in line:
                            cusz_comp_line_number.append(line_number)
                        if (("CUDA Kernel Statistics" in line) or ("cuda_gpu_kern_sum" in line) or ("gpukernsum" in line)) and len(cusz_comp_line_number) == 1:
                            nsys_comp_line_number.append(line_number)
                        if (("CUDA Memory Operation Statistics (by time):" in line) or ("cuda_gpu_mem_time_sum" in line) or ("gpumemtimesum" in line)) and len(cusz_comp_line_number) == 1:
                            nsys_comp_line_number.append(line_number)
                        
                        # decompression
                    
                        if 'CR = ' in line:
                            # CR
                            compression_ratio_value = float(line_split[-1])
                            df.loc[(eb, data_point), 'CR'] = compression_ratio_value
                            df.loc[(eb, data_point), 'BR'] = 32.0 / compression_ratio_value
                        
                        if "-cuszx_decompress-" in line:
                            cusz_decomp_line_number.append(line_number)
                        if (("CUDA Kernel Statistics" in line) or ("cuda_gpu_kern_sum" in line) or ("gpukernsum" in line)) and len(cusz_decomp_line_number) == 1:
                            nsys_decomp_line_number.append(line_number)
                        if ((("CUDA Memory Operation Statistics (by time):" in line) or ("cuda_gpu_mem_time_sum" in line) or ("gpumemtimesum" in line)) or "Operating System Runtime API Statistics:" in line) and len(cusz_decomp_line_number) == 1 and len(nsys_decomp_line_number) == 1:
                            nsys_decomp_line_number.append(line_number)
                        
                        
                        if "-compareData-" in line:
                            compareDATA_line_number.append(line_number)
                        if "-nvcomp-" in line:
                            nvcomp_line_number.append(line_number)
                        if "-bitcomp-" in line:
                            bitcomp_line_number.append(line_number)
                    if len(nsys_comp_line_number) >= 2:
                        self.analyze_nsys(lines[nsys_comp_line_number[0]:nsys_comp_line_number[1]], df, (eb, data_point), 
                            'nsys_cmp_cTP', self.data_size, ["szx::compress_float"])
                    if len(nsys_decomp_line_number) >= 2:
                        self.analyze_nsys(lines[nsys_decomp_line_number[0]:nsys_decomp_line_number[1]], df, (eb, data_point), 
                            'nsys_cmp_xTP', self.data_size, ["szx::decompress_float"])
                    compressed_size = self.data_size / compression_ratio_value
                    if len(compareDATA_line_number) >= 2:
                        self.analyze_compareData(lines[compareDATA_line_number[0]:compareDATA_line_number[1]], df, (eb, data_point))
                    if len(nvcomp_line_number) >= 2:
                        self.analyze_nvcomp(lines[nvcomp_line_number[0]:nvcomp_line_number[1]], df, (eb, data_point), compressed_size)
                    if len(bitcomp_line_number) >= 2:
                        self.analyze_bitcomp(lines[bitcomp_line_number[0]:bitcomp_line_number[1]], df, (eb, data_point), compressed_size)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    # print(file_path)
                    # print(nvcomp_line_number)
                    # print(len(lines))
                    # assert 0
                    pass
            self.compute_overall(df, eb)
        
        self.df[cmp] = df
        self.df_overall[cmp] = self.extract_overall(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help="(MANDATORY) input folder for logs", type=str)
    parser.add_argument('--output', '-o', help="(MANDATORY) output folder for CSV", type=str)
    parser.add_argument('--dim', '-d', help="data dimension", type=int,default=3)
    parser.add_argument('--dims', '-m', help="(MANDATORY) data dimension", type=str,nargs="+")
    parser.add_argument('--cmp', '-c', help="specify a list of compressors", type=str,nargs="*")
    parser.add_argument('--eb', '-e', help="specify a list of error bounds", type=str,nargs="*")
    parser.add_argument('--br', '-b', help="specify a list of bit rates", type=str,nargs="*")
    parser.add_argument('--type', '-t', type=str,default="f32")
    
    
    args = parser.parse_args()
    
    datafolder   = args.input
    outputfolder = args.output
    data_size    = args.dims
    cmp_list     = args.cmp
    eb_list      = args.eb
    br_list      = args.br


    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    
    dataset = os.path.basename(os.path.normpath(datafolder))
    machine = os.getenv('MACHINE_NAME')

    
    analysis = Analysis(datafolder, outputfolder, data_size, cmp_list=cmp_list, eb_list=eb_list, br_list=br_list, dataset=dataset, machine=machine)
    analysis.launch_analysis()
    # for i in range(self)
    # analysis.analyze_FZGPU()
    # analysis.analyze_cuSZ()
    # analysis.analyze_cuSZi()
    # analysis.analyze_cuSZp()
    # analysis.analyze_cuSZx()
    # analysis.analyze_cuzfp()
    analysis.save_to_csv()
    # analysis.df['cuzfp'].to_csv(os.path.join(outputfolder, 'test.csv'),  sep=',', index=True)
    # print(analysis.df['cuzfp'])
