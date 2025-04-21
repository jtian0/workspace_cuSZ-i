import os
import sys

GPU_name = "A100"
dataset_dims = {
    # ablation
    # "Miranda":  (384, 384, 256),
    # "NYX":      (512, 512, 512),
    # "JHTDB":    (512, 512, 512),
    # "ARAMCO":   (235, 449, 449),
    
    # NEW tp:
    "S3D":      (500, 500, 500),
    "QMCPack":  (69, 69, 33120),
    "Hurricane":(500, 500, 100),
    "SCALE":    (1200, 1200, 98),
    "CESM":     (3600, 1800, 1),
    "Miranda":  (384, 384, 256),
    "NYX":      (512, 512, 512),
    "JHTDB":    (512, 512, 512),
    "ARAMCO":   (235, 449, 449),
    "SegSalt":  (1008, 1008, 352),
}


for dataset, dims in dataset_dims.items():
    dim_str = f"{dims[0]} {dims[1]} {dims[2]}"  # 注意顺序是 z y x
    cmd = (
        f"python script_data_analysis.py "
        f"--input data_{GPU_name}/{dataset}_log/ "
        f"--output data_{GPU_name}/{dataset}_csv/ "
        f"--dims {dim_str}"
    )
    print(cmd)
    os.system(cmd)

cmp_list_ = [
            ["cuSZ_24", "cuSZi_24", "cuSZp_outlier",
            "cuSZp_plain",
            "FZGPU",
            "cuzfp",  "cuSZi_a3_Huff_1" , "cuSZi_a6_Huff_1","cuSZi_a3_Huff_0" , "cuSZi_a6_Huff_0"],

            ## ablation
            # [ "cuSZi_24", "cuSZi_a3_Huff_1" , "cuSZi_a6_Huff_1",],
            #    ['cuSZi_interp_16_4steps',
            #         'cuSZi_interp_16_4steps_reorder',
            #         'cuSZi_interp_16_4steps_reorder_att_balance_a3',
            #         'cuSZi_interp_16_4steps_reorder_att_balance_a6',]
]

data_folder_ = [f"data_{GPU_name}"]


eb = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]

for dataset, dims in dataset_dims.items():
    output_csv = ""
    output_csv_zfp = ""
    for i in range(2):
        cmp_list = cmp_list_[i]
        data_folder = data_folder_[i]
        for i, cmp in enumerate(cmp_list):
            # sys.subprocess.run(f"mkdir -p data_{cmp}", shell=True)
            csv_path = os.path.join(f"{data_folder}", f"{dataset}_csv")
            if cmp != "cuzfp":
                # print(csv_path)
                try:
                    file_path = None
                    for f_ in os.listdir(csv_path):
                        
                        # if "overall" in f_ and cmp in f_:
                        if "overall" in f_ and f_.startswith(cmp + "_" + dataset):
                            if cmp == 'cuSZ' and f_.split('_')[0] != cmp:
                                print(f_.split('_')[-1])
                                continue
                            file_path = os.path.join(csv_path, f_)
                            with open(file_path, 'r') as f:
                                lines = f.readlines()
                                for line in lines:
                                    vals = line.split(",")
                                    if "cmp_cTP" in line:
                                        vals.insert(1, "CMP")
                                        if "cmp_cTP" not in output_csv:
                                            output_csv += ",".join(vals)
                                    else:
                                        vals.insert(1, cmp)
                                        output_csv += ",".join(vals)

                except Exception as e:
                    print(e)
                    print(f"Error: {cmp} {dataset}")
                    pass
            else:
                try:
                    file_path = None
                    for f_ in os.listdir(csv_path):
                        
                        if "overall" in f_ and cmp in f_:
                            
                            file_path = os.path.join(csv_path, f_)
                            with open(file_path, 'r') as f:
                                lines = f.readlines()
                                for line in lines:
                                    vals = line.split(",")
                                    if "cmp_cTP" in line:
                                        vals.insert(1, "CMP")
                                        if "cmp_cTP" not in output_csv_zfp:
                                            output_csv_zfp += ",".join(vals)
                                    else:
                                        vals.insert(1, cmp)
                                        output_csv_zfp += ",".join(vals)

                except Exception as e:
                    print(e)
                    print(f"Error: {cmp} {dataset}")
                    pass

    # print(output_csv)
    with open(f"{data_folder_[-1]}/data_overall/{dataset}_overall.csv", 'w') as f:
        f.write(output_csv)
    with open(f"{data_folder_[-1]}/sorted_data_overall/{dataset}_overall_zfp.csv", 'w') as f:
        f.write(output_csv_zfp)
    # assert 0
            
import pandas as pd
import itertools


cmp_list = [
            "cuSZ_24", "cuSZi_24", "cuSZi_a3_Huff_1" , "cuSZi_a6_Huff_1","cuSZi_a3_Huff_0" , "cuSZi_a6_Huff_0",
            "cuSZp_outlier",
            "cuSZp_plain",
            "FZGPU",
]
eb_list = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
data_folder = f"data_{GPU_name}"


# 从 CSV 读取数据
for dataset, dims in dataset_dims.items():
    try:
        df = pd.read_csv(os.path.join(f"{data_folder}/data_overall", f"{dataset}_overall.csv"))  # 替换为实际路径
        df["Error_Bound"] = df["Error_Bound"].astype(float)
        df["CMP"] = df["CMP"].astype(str)

        # 创建完整的 MultiIndex 网格
        full_index = pd.MultiIndex.from_tuples(itertools.product(eb_list, cmp_list), names=["Error_Bound", "CMP"])

        # 设置索引并 reindex 补齐缺失项
        df = df.set_index(["Error_Bound", "CMP"])
        df_full = df.reindex(full_index).reset_index()

        # 排序（按照 eb_list 和 cmp_list 的顺序）
        df_full["eb_order"] = df_full["Error_Bound"].apply(lambda x: eb_list.index(x))
        df_full["cmp_order"] = df_full["CMP"].apply(lambda x: cmp_list.index(x))
        df_sorted = df_full.sort_values(by=["eb_order", "cmp_order"]).drop(columns=["eb_order", "cmp_order"]).reset_index(drop=True)

        # df_sorted 即为你需要的完整结果（缺失的数据为 NaN）
        df_sorted.to_csv(os.path.join(f"{data_folder}/sorted_data_overall", f"{dataset}_overall.csv"), index=False)
    except Exception as e:
        print(e)
        pass

import os
data_folder = f"data_{GPU_name}"
for dataset, dims in dataset_dims.items():
	cmd = f"cd {data_folder}; cd sorted_data_overall; cat {dataset}_overall_zfp.csv {dataset}_overall.csv > ../data_{GPU_name}_merged/{dataset}_overall_merged.csv"
	print(cmd)
	os.system(cmd)