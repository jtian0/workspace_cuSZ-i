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
        
        # self.cmp_list = cmp_list or ['cuSZp']
        # 'cuSZi_a2_Huff_1', 
        # 'cuSZi_a2_Huff_0',
        self.cmp_list = [
                    #     'FZGPU', 
                    #        'cuSZp_plain', 'cuSZp_outlier',
                    # 'cuzfp', 
                    
                    # 'cuSZi_a3_Huff_1',
                    #   'cuSZi_a6_Huff_1',
                    #   'cuSZi_a3_Huff_0',
                    #     'cuSZi_a6_Huff_0',
                    #   'cuSZ_24', 
                      'cuSZi_24',
                    #      'cuSZi_interp_16_4steps',
                    # 'cuSZi_interp_16_4steps_reorder',
                    # 'cuSZi_interp_16_4steps_reorder_att_balance_a3',
                    # 'cuSZi_interp_16_4steps_reorder_att_balance_a6',
                      ]
        
        # self.eb_list = eb_list or ['1e-2', '1e-3', '1e-4', '1e-5']
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
            # 'cuSZ' : self.analyze_cuSZ,
            'cuSZ_24' : self.analyze_cuSZ_24,
            'cuSZi_24' : self.analyze_cuSZi_24,
            'cuSZx': self.analyze_cuSZx,
            'cuSZp_plain': self.analyze_cuSZp_plain,
            'cuSZp_outlier': self.analyze_cuSZp_outlier,
            # 'cuSZi_a2_Huff_1': self.analyze_cuSZi_a2_Huff_1,
            'cuSZi_a3_Huff_1': self.analyze_cuSZi_a3_Huff_1,
            'cuSZi_a6_Huff_1': self.analyze_cuSZi_a6_Huff_1,
            # 'cuSZi_a2_Huff_0': self.analyze_cuSZi_a2_Huff_0,
            'cuSZi_a3_Huff_0': self.analyze_cuSZi_a3_Huff_0,
            'cuSZi_a6_Huff_0': self.analyze_cuSZi_a6_Huff_0,
            'cuzfp': self.analyze_cuzfp,
            'cuZFP': self.analyze_cuzfp,
             'cuSZi_interp_16_4steps': self.analyze_cuSZi_interp_16_4steps,
            'cuSZi_interp_16_4steps_reorder': self.analyze_cuSZi_interp_16_4steps_reorder,
            'cuSZi_interp_16_4steps_reorder_att_balance_a3': self.analyze_cuSZi_interp_16_4steps_reorder_att_balance_a3,
            'cuSZi_interp_16_4steps_reorder_att_balance_a6': self.analyze_cuSZi_interp_16_4steps_reorder_att_balance_a6,
            
        }
        self.metrics = ['CR', 'BR', 'PSNR', 'NRMSE', 'cmp_cTP', 'cmp_xTP', 'nsys_cmp_cTP', 'nsys_cmp_xTP', 
                        'nvcomp_CR', 'nvcomp_cTP', 'nvcomp_xTP', 'bitcomp_CR', 'bitcomp_cTP', 'bitcomp_xTP',]
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
        overall_df = df.xs('_overall', level='Data_Point')
        overall_df.index = overall_df.index.map(float)
        overall_df.sort_index(inplace=True)
        overall_df.index = overall_df.index.map('{:0.1e}'.format)
        return overall_df
        
    def compute_throughput(self, data_size, elapsed_time, if_success=1):
        return (float(data_size) * self.data_type_size) /  ((2 ** 30) * elapsed_time * 1e-9)
    
    def compute_overall(self, df, eb, if_success=1):
        # CR
        valid_idx = [subkey for subkey in df.loc[eb].index
             if subkey != '_overall' and not pd.isna(df.loc[(eb, subkey), 'nvcomp_cTP'])]

        # 再提取这些子项在 'CR' 上的值
        non_overall_values = df.loc[(eb, valid_idx), 'CR']
        # non_overall_values = df.loc[eb, 'CR'].drop((eb, '_overall'), errors='ignore')
        harmonic_mean = hmean(non_overall_values.dropna().values) if len(non_overall_values.dropna().values) != 0 else None
        df.loc[(eb, '_overall'), 'CR'] = harmonic_mean if if_success else pd.NA
        
        
        # non_overall_values = df.loc[eb, 'nvcomp_CR'].drop((eb, '_overall'), errors='ignore')
        non_overall_values = df.loc[(eb, valid_idx), 'nvcomp_CR']
        harmonic_mean = hmean(non_overall_values.dropna().values) if len(non_overall_values.dropna().values) != 0 else None
        df.loc[(eb, '_overall'), 'nvcomp_CR'] = harmonic_mean if if_success else pd.NA
        
        # non_overall_values = df.loc[eb, 'bitcomp_CR'].drop((eb, '_overall'), errors='ignore')
        non_overall_values = df.loc[(eb, valid_idx), 'bitcomp_CR']
        harmonic_mean = hmean(non_overall_values.dropna().values) if len(non_overall_values.dropna().values) != 0 else None
        df.loc[(eb, '_overall'), 'bitcomp_CR'] = harmonic_mean if if_success else pd.NA
        
        # BR
        non_overall_values = df.loc[eb, 'BR'].drop((eb, '_overall'), errors='ignore')
        non_overall_values = df.loc[(eb, valid_idx), 'BR']
        arithmetic_mean = mean(non_overall_values.dropna().values) if len(non_overall_values.dropna().values) != 0 else None
        df.loc[(eb, '_overall'), 'BR'] = arithmetic_mean if if_success else pd.NA
        
        # PSNR
        non_overall_values = df.loc[eb, 'NRMSE'].drop((eb, '_overall'), errors='ignore')
        non_overall_values = df.loc[(eb, valid_idx), 'NRMSE']
        NRMSE_list = non_overall_values.dropna().values
        PSNR_avg = 0
        if len(NRMSE_list) != 0:
            for NRMSE in NRMSE_list:
                PSNR_avg += NRMSE ** 2
            PSNR_avg /= len(NRMSE_list)
        
            PSNR_avg = -20.0 * log10(sqrt(PSNR_avg))
            df.loc[(eb, '_overall'), 'PSNR'] = PSNR_avg if if_success else pd.NA
        
        # TP
        TP_list = [TP for TP in self.metrics if 'TP' in TP]
        for TP in TP_list:
            non_overall_values = df.loc[eb, TP].drop((eb, '_overall'), errors='ignore')
            arithmetic_mean = mean(non_overall_values.dropna().values) if len(non_overall_values.dropna().values) != 0 else None
            df.loc[(eb, '_overall'), TP] = arithmetic_mean if if_success else pd.NA
            
        
        
    
    def analyze_compareData(self, lines, df, df_loc, if_success=1):
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
                df.loc[df_loc, 'PSNR'] = float(line_split[2][:-1]) if if_success else pd.NA
                df.loc[df_loc, 'NRMSE'] = float(line_split[5]) if if_success else pd.NA
                break
        return
    
    def analyze_nvcomp(self, lines, df, df_loc, data_size, if_success=1):
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
                            'nvcomp_cTP', data_size, ["bitcomp::batch_encoder_kernel"], if_success=if_success)
            self.analyze_nsys(lines[nsys_line_number[0]:nsys_line_number[1]], df, df_loc, 
                            'nvcomp_xTP', data_size, ["bitcomp::batch_decoder_kernel"], if_success=if_success)
        return
    
    def analyze_bitcomp(self, lines, df, df_loc, data_size, if_success=1):
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
                            'bitcomp_cTP', data_size, ["bitcomp::encoder_kernel"], if_success=if_success)
            self.analyze_nsys(lines[nsys_line_number[0]:nsys_line_number[1]], df, df_loc, 
                            'bitcomp_xTP', data_size, ["bitcomp::decoder_kernel"], if_success=if_success)
        return
    
    def analyze_nsys(self, lines, df, df_loc, metric, data_size, func_names, statics=0, if_success=1):
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
        df.loc[df_loc, metric] = self.compute_throughput(data_size, time) if if_success else pd.NA
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

    def analyze_cuSZ_24(self,):

        cmp = "cuSZ_24"
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
  
        
    def analyze_cuSZi_24(self,):

        cmp = "cuSZi_24"
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
                          'nsys_cmp_cTP', self.data_size, [" cusz::c_spline3d_infprecis_32x8x8data", "psz::detail::hf_encode_phase2_deflate", 
                                        "histsp_multiwarp", "psz::detail::hf_encode_phase1_fill", "psz::extrema_kernel", "psz::detail::hf_encode_phase4_concatenate",
                                        "cusz::c_spline3d_profiling_data_2"])
                    
                    self.analyze_nsys(lines[nsys_decomp_line_number[0]:nsys_decomp_line_number[1]], df, (eb, data_point), 
                          'nsys_cmp_xTP', self.data_size, [" hf_decode_kernel", 
                                        "cusz::x_spline3d_infprecis_32x8x8data", "psz::extrema_kernel"])
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

    def analyze_cuSZi_a2_Huff_1(self,):

        cmp = "cuSZi_a2_Huff_1"
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
                    # self.analyze_nsys(lines[nsys_comp_line_number[0]:nsys_comp_line_number[1]], df, (eb, data_point), 
                    #       'nsys_cmp_cTP', self.data_size, [" cusz::c_spline_infprecis_data", "psz::detail::hf_encode_phase2_deflate", 
                    #                     "histsp_multiwarp", "psz::detail::hf_encode_phase1_fill", "psz::extrema_kernel", "psz::detail::hf_encode_phase4_concatenate",
                    #                     "cusz::c_spline_profiling_data_2", "pa_spline_infprecis_data", "d_encode_rtr"])
                    # self.analyze_nsys(lines[nsys_decomp_line_number[0]:nsys_decomp_line_number[1]], df, (eb, data_point), 
                    #       'nsys_cmp_xTP', self.data_size, [" hf_decode_kernel", 
                    #                     "cusz::x_spline_infprecis_data", "psz::extrema_kernel", "d_decode_rtr"])
                    self.analyze_nsys(lines[nsys_comp_line_number[0]:nsys_comp_line_number[1]], df, (eb, data_point), 
                          'nsys_cmp_cTP', self.data_size, [" cusz::c_spline_infprecis_data", 
                                                           "psz::detail::hf_encode_phase2_deflate", 
                                                            "histsp_multiwarp",
                                                             "psz::detail::hf_encode_phase1_fill",
                                                            #    "psz::extrema_kernel",
                                                                 "psz::detail::hf_encode_phase4_concatenate",
                                                            # "cusz::c_spline_profiling_data_2",
                                                              "pa_spline_infprecis_data",
                                                                "d_encode_rtr",
                                                                "cusz::reset_errors",
                                                                "d_reset()"
                                                                ])
                    self.analyze_nsys(lines[nsys_decomp_line_number[0]:nsys_decomp_line_number[1]], df, (eb, data_point), 
                          'nsys_cmp_xTP', self.data_size, [" hf_decode_kernel", 
                                        "cusz::x_spline_infprecis_data", "psz::extrema_kernel",
                                          "d_decode_rtr",
                                          "cu_hip::spvn_scatter<float"])
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
    def analyze_cuSZi_a3_Huff_1(self,):

        cmp = "cuSZi_a3_Huff_1"
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
                          'nsys_cmp_cTP', self.data_size, [" cusz::c_spline_infprecis_data", 
                                                           "psz::detail::hf_encode_phase2_deflate", 
                                                            "histsp_multiwarp",
                                                             "psz::detail::hf_encode_phase1_fill",
                                                            #    "psz::extrema_kernel",
                                                                 "psz::detail::hf_encode_phase4_concatenate",
                                                            # "cusz::c_spline_profiling_data_2",
                                                              "pa_spline_infprecis_data",
                                                                "d_encode_rtr",
                                                                "cusz::reset_errors",
                                                                "d_reset()"
                                                                ])
                    self.analyze_nsys(lines[nsys_decomp_line_number[0]:nsys_decomp_line_number[1]], df, (eb, data_point), 
                          'nsys_cmp_xTP', self.data_size, [" hf_decode_kernel", 
                                        "cusz::x_spline_infprecis_data", "psz::extrema_kernel",
                                          "d_decode_rtr",
                                          "cu_hip::spvn_scatter<float"])
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
    
    def analyze_cuSZi_a6_Huff_1(self,):

        cmp = "cuSZi_a6_Huff_1"
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
                          'nsys_cmp_cTP', self.data_size, [" cusz::c_spline_infprecis_data", 
                                                           "psz::detail::hf_encode_phase2_deflate", 
                                                            "histsp_multiwarp",
                                                             "psz::detail::hf_encode_phase1_fill",
                                                            #    "psz::extrema_kernel",
                                                                 "psz::detail::hf_encode_phase4_concatenate",
                                                            # "cusz::c_spline_profiling_data_2",
                                                              "pa_spline_infprecis_data",
                                                                "d_encode_rtr",
                                                                "cusz::reset_errors",
                                                                "d_reset()"
                                                                ])
                    self.analyze_nsys(lines[nsys_decomp_line_number[0]:nsys_decomp_line_number[1]], df, (eb, data_point), 
                          'nsys_cmp_xTP', self.data_size, [" hf_decode_kernel", 
                                        "cusz::x_spline_infprecis_data", "psz::extrema_kernel",
                                          "d_decode_rtr",
                                          "cu_hip::spvn_scatter<float"])
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

    def analyze_cuSZi_a2_Huff_0(self,):

        cmp = "cuSZi_a2_Huff_0"
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
                          'nsys_cmp_cTP', self.data_size, [" cusz::c_spline_infprecis_data", 
                                                           "psz::detail::hf_encode_phase2_deflate", 
                                                            "histsp_multiwarp",
                                                             "psz::detail::hf_encode_phase1_fill",
                                                            #    "psz::extrema_kernel",
                                                                 "psz::detail::hf_encode_phase4_concatenate",
                                                            # "cusz::c_spline_profiling_data_2",
                                                              "pa_spline_infprecis_data",
                                                                "d_encode_rtr",
                                                                "cusz::reset_errors",
                                                                "d_reset()"
                                                                ])
                    self.analyze_nsys(lines[nsys_decomp_line_number[0]:nsys_decomp_line_number[1]], df, (eb, data_point), 
                          'nsys_cmp_xTP', self.data_size, [" hf_decode_kernel", 
                                        "cusz::x_spline_infprecis_data", "psz::extrema_kernel",
                                          "d_decode_rtr",
                                          "cu_hip::spvn_scatter<float"])
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
    def analyze_cuSZi_a3_Huff_0(self,):

        cmp = "cuSZi_a3_Huff_0"
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
                          'nsys_cmp_cTP', self.data_size, [" cusz::c_spline_infprecis_data", 
                                                           "psz::detail::hf_encode_phase2_deflate", 
                                                            "histsp_multiwarp",
                                                             "psz::detail::hf_encode_phase1_fill",
                                                            #    "psz::extrema_kernel",
                                                                 "psz::detail::hf_encode_phase4_concatenate",
                                                            # "cusz::c_spline_profiling_data_2",
                                                              "pa_spline_infprecis_data",
                                                                "d_encode_rtr",
                                                                "cusz::reset_errors",
                                                                "d_reset()"
                                                                ])
                    self.analyze_nsys(lines[nsys_decomp_line_number[0]:nsys_decomp_line_number[1]], df, (eb, data_point), 
                          'nsys_cmp_xTP', self.data_size, [" hf_decode_kernel", 
                                        "cusz::x_spline_infprecis_data", "psz::extrema_kernel",
                                          "d_decode_rtr",
                                          "cu_hip::spvn_scatter<float"])
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
    
    def analyze_cuSZi_a6_Huff_0(self,):

        cmp = "cuSZi_a6_Huff_0"
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
                          'nsys_cmp_cTP', self.data_size, [" cusz::c_spline_infprecis_data", 
                                                           "psz::detail::hf_encode_phase2_deflate", 
                                                            "histsp_multiwarp",
                                                             "psz::detail::hf_encode_phase1_fill",
                                                            #    "psz::extrema_kernel",
                                                                 "psz::detail::hf_encode_phase4_concatenate",
                                                            # "cusz::c_spline_profiling_data_2",
                                                              "pa_spline_infprecis_data",
                                                                "d_encode_rtr",
                                                                "cusz::reset_errors",
                                                                "d_reset()"
                                                                ])
                    self.analyze_nsys(lines[nsys_decomp_line_number[0]:nsys_decomp_line_number[1]], df, (eb, data_point), 
                          'nsys_cmp_xTP', self.data_size, [" hf_decode_kernel", 
                                        "cusz::x_spline_infprecis_data", "psz::extrema_kernel",
                                          "d_decode_rtr",
                                          "cu_hip::spvn_scatter<float"])
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

#    'cuSZi_interp_16_4steps',
#                     'cuSZi_interp_16_4steps_reorder',
#                     'cuSZi_interp_16_4steps_reorder_att_balance_a3',
#                     'cuSZi_interp_16_4steps_reorder_att_balance_a6',
    def analyze_cuSZi_interp_16_4steps(self,):

        cmp = "cuSZi_interp_16_4steps"
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
                          'nsys_cmp_cTP', self.data_size, [" cusz::c_spline_infprecis_data", 
                                                           "psz::detail::hf_encode_phase2_deflate", 
                                                            "histsp_multiwarp",
                                                             "psz::detail::hf_encode_phase1_fill",
                                                            #    "psz::extrema_kernel",
                                                                 "psz::detail::hf_encode_phase4_concatenate",
                                                            "cusz::c_spline_profiling_data_2",
                                                              "pa_spline_infprecis_data",
                                                                "d_encode_rtr",
                                                                "cusz::reset_errors",
                                                                "d_reset()"
                                                                ])
                    self.analyze_nsys(lines[nsys_decomp_line_number[0]:nsys_decomp_line_number[1]], df, (eb, data_point), 
                          'nsys_cmp_xTP', self.data_size, [" hf_decode_kernel", 
                                        "cusz::x_spline_infprecis_data", "psz::extrema_kernel",
                                          "d_decode_rtr",
                                          "cu_hip::spvn_scatter<float"])
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

#    'cuSZi_interp_16_4steps',
#                     'cuSZi_interp_16_4steps_reorder',
#                     'cuSZi_interp_16_4steps_reorder_att_balance_a3',
#                     'cuSZi_interp_16_4steps_reorder_att_balance_a6',
    def analyze_cuSZi_interp_16_4steps_reorder(self,):

        cmp = "cuSZi_interp_16_4steps_reorder"
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
                          'nsys_cmp_cTP', self.data_size, [" cusz::c_spline_infprecis_data", 
                                                           "psz::detail::hf_encode_phase2_deflate", 
                                                            "histsp_multiwarp",
                                                             "psz::detail::hf_encode_phase1_fill",
                                                            #    "psz::extrema_kernel",
                                                                 "psz::detail::hf_encode_phase4_concatenate",
                                                            "cusz::c_spline_profiling_data_2",
                                                              "pa_spline_infprecis_data",
                                                                "d_encode_rtr",
                                                                "cusz::reset_errors",
                                                                "d_reset()"
                                                                ])
                    self.analyze_nsys(lines[nsys_decomp_line_number[0]:nsys_decomp_line_number[1]], df, (eb, data_point), 
                          'nsys_cmp_xTP', self.data_size, [" hf_decode_kernel", 
                                        "cusz::x_spline_infprecis_data", "psz::extrema_kernel",
                                          "d_decode_rtr",
                                          "cu_hip::spvn_scatter<float"])
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
#    'cuSZi_interp_16_4steps',
#                     'cuSZi_interp_16_4steps_reorder',
#                     'cuSZi_interp_16_4steps_reorder_att_balance_a3',
#                     'cuSZi_interp_16_4steps_reorder_att_balance_a6',
    def analyze_cuSZi_interp_16_4steps_reorder_att_balance_a3(self,):

        cmp = "cuSZi_interp_16_4steps_reorder_att_balance_a3"
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
                          'nsys_cmp_cTP', self.data_size, [" cusz::c_spline_infprecis_data", 
                                                           "psz::detail::hf_encode_phase2_deflate", 
                                                            "histsp_multiwarp",
                                                             "psz::detail::hf_encode_phase1_fill",
                                                            #    "psz::extrema_kernel",
                                                                 "psz::detail::hf_encode_phase4_concatenate",
                                                            "cusz::c_spline_profiling_data_2",
                                                              "pa_spline_infprecis_data",
                                                                "d_encode_rtr",
                                                                "cusz::reset_errors",
                                                                "d_reset()"
                                                                ])
                    self.analyze_nsys(lines[nsys_decomp_line_number[0]:nsys_decomp_line_number[1]], df, (eb, data_point), 
                          'nsys_cmp_xTP', self.data_size, [" hf_decode_kernel", 
                                        "cusz::x_spline_infprecis_data", "psz::extrema_kernel",
                                          "d_decode_rtr",
                                          "cu_hip::spvn_scatter<float"])
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
#    'cuSZi_interp_16_4steps',
#                     'cuSZi_interp_16_4steps_reorder',
#                     'cuSZi_interp_16_4steps_reorder_att_balance_a3',
#                     'cuSZi_interp_16_4steps_reorder_att_balance_a6',
    def analyze_cuSZi_interp_16_4steps_reorder_att_balance_a6(self,):

        cmp = "cuSZi_interp_16_4steps_reorder_att_balance_a6"
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
                          'nsys_cmp_cTP', self.data_size, [" cusz::c_spline_infprecis_data", 
                                                           "psz::detail::hf_encode_phase2_deflate", 
                                                            "histsp_multiwarp",
                                                             "psz::detail::hf_encode_phase1_fill",
                                                            #    "psz::extrema_kernel",
                                                                 "psz::detail::hf_encode_phase4_concatenate",
                                                            "cusz::c_spline_profiling_data_2",
                                                              "pa_spline_infprecis_data",
                                                                "d_encode_rtr",
                                                                "cusz::reset_errors",
                                                                "d_reset()"
                                                                ])
                    self.analyze_nsys(lines[nsys_decomp_line_number[0]:nsys_decomp_line_number[1]], df, (eb, data_point), 
                          'nsys_cmp_xTP', self.data_size, [" hf_decode_kernel", 
                                        "cusz::x_spline_infprecis_data", "psz::extrema_kernel",
                                          "d_decode_rtr",
                                          "cu_hip::spvn_scatter<float"])
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
        

    def analyze_cuSZp_outlier(self, ):
        # Create the DataFrame with MultiIndex
        cmp = "cuSZp_outlier"
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
                    if_success = True
                    for line_number, line in enumerate(lines):
                        if "Fail error check!" in line:
                            if_success = False
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
                          'nsys_cmp_cTP', self.data_size, ["cuSZp_compress_kernel_plain_f32", "cuSZp_compress_kernel_outlier_f32"], if_success)
                    self.analyze_nsys(lines[nsys_line_number[0]:nsys_line_number[1]], df, (eb, data_point), 
                          'nsys_cmp_xTP', self.data_size, ["cuSZp_decompress_kernel_plain_f32", "cuSZp_decompress_kernel_outlier_f32"], if_success)
                    compressed_size = self.data_size / compression_ratio_value
                    self.analyze_compareData(lines[compareDATA_line_number[0]:compareDATA_line_number[1]], df, (eb, data_point), if_success)
                    self.analyze_nvcomp(lines[nvcomp_line_number[0]:nvcomp_line_number[1]], df, (eb, data_point), compressed_size, if_success)
                    self.analyze_bitcomp(lines[bitcomp_line_number[0]:bitcomp_line_number[1]], df, (eb, data_point), compressed_size, if_success)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print(file_path)
                    # assert 0
                    pass
            # mask = df[eb, 'nvcomp_cTP'].isna()
            # df.loc[mask] = self.df[eb, 'cuSZp_plain'].loc[mask]
            plain_df = self.df['cuSZp_plain']
            mask = df.loc[eb, 'nvcomp_cTP'].isna()
            datapoints_to_replace = mask[mask].index
            rows_to_replace = [(eb, dp) for dp in datapoints_to_replace]
            df.loc[rows_to_replace, :] = plain_df.loc[rows_to_replace, :]
            print(self.data_folder, mask)

            
            self.compute_overall(df, eb)
        
        self.df[cmp] = df
        self.df_overall[cmp] = self.extract_overall(df)
    def analyze_cuSZp_plain(self, ):
        # Create the DataFrame with MultiIndex
        cmp = "cuSZp_plain"
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
                          'nsys_cmp_cTP', self.data_size, ["cuSZp_compress_kernel_plain_f32", "cuSZp_compress_kernel_outlier_f32"])
                    self.analyze_nsys(lines[nsys_line_number[0]:nsys_line_number[1]], df, (eb, data_point), 
                          'nsys_cmp_xTP', self.data_size, ["cuSZp_decompress_kernel_plain_f32", "cuSZp_decompress_kernel_outlier_f32"])
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
