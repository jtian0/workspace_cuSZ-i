#!/bin/bash

# python3 script_data_collection.py --input ~/datasets/100x500x500 --output ~/datasets/100x500x500_hf_log --dims 500 500 100 --cmp cuSZi
# python3 script_data_collection.py --input ~/datasets/SDRBENCH-EXASKY-NYX-512x512x512 --output ~/datasets/SDRBENCH-EXASKY-NYX-512x512x512_hf_log --dims 512 512 512 --cmp cuSZi
# python3 script_data_collection.py --input ~/datasets/SDRBENCH-Miranda-256x384x384 --output ~/datasets/SDRBENCH-Miranda-256x384x384_hf_log --dims 384 384 256 --cmp cuSZi
# python3 script_data_collection.py --input ~/datasets/SDRBENCH-SCALE_98x1200x1200 --output ~/datasets/SDRBENCH-SCALE_98x1200x1200_hf_log --dims 1200 1200 98 --cmp cuSZi

# python3 script_data_analysis.py --input ~/datasets/100x500x500_hf_log --output ~/datasets/100x500x500_hf_csv --dims 500 500 100 --cmp cuSZi
# python3 script_data_analysis.py --input ~/datasets/SDRBENCH-EXASKY-NYX-512x512x512_hf_log --output ~/datasets/SDRBENCH-EXASKY-NYX-512x512x512_hf_csv --dims 512 512 512 --cmp cuSZi
# python3 script_data_analysis.py --input ~/datasets/SDRBENCH-Miranda-256x384x384_hf_log --output ~/datasets/SDRBENCH-Miranda-256x384x384_hf_csv --dims 384 384 256 --cmp cuSZi
# python3 script_data_analysis.py --input ~/datasets/SDRBENCH-SCALE_98x1200x1200_hf_log --output ~/datasets/SDRBENCH-SCALE_98x1200x1200_hf_csv --dims 1200 1200 98 --cmp cuSZi

# python3 script_data_collection.py --input ~/datasets/100x500x500 --output ~/datasets/100x500x500_nohf_log --dims 500 500 100 --cmp cuSZi
# python3 script_data_collection.py --input ~/datasets/SDRBENCH-EXASKY-NYX-512x512x512 --output ~/datasets/SDRBENCH-EXASKY-NYX-512x512x512_nohf_log --dims 512 512 512 --cmp cuSZi
# python3 script_data_collection.py --input ~/datasets/SDRBENCH-Miranda-256x384x384 --output ~/datasets/SDRBENCH-Miranda-256x384x384_nohf_log --dims 384 384 256 --cmp cuSZi
# python3 script_data_collection.py --input ~/datasets/SDRBENCH-SCALE_98x1200x1200 --output ~/datasets/SDRBENCH-SCALE_98x1200x1200_nohf_log --dims 1200 1200 98 --cmp cuSZi

# python3 script_data_analysis.py --input ~/datasets/100x500x500_nohf_log --output ~/datasets/100x500x500_nohf_csv --dims 500 500 100 --cmp cuSZi
# python3 script_data_analysis.py --input ~/datasets/SDRBENCH-EXASKY-NYX-512x512x512_nohf_log --output ~/datasets/SDRBENCH-EXASKY-NYX-512x512x512_nohf_csv --dims 512 512 512 --cmp cuSZi
# python3 script_data_analysis.py --input ~/datasets/SDRBENCH-Miranda-256x384x384_nohf_log --output ~/datasets/SDRBENCH-Miranda-256x384x384_nohf_csv --dims 384 384 256 --cmp cuSZi
# python3 script_data_analysis.py --input ~/datasets/SDRBENCH-SCALE_98x1200x1200_nohf_log --output ~/datasets/SDRBENCH-SCALE_98x1200x1200_nohf_csv --dims 1200 1200 98 --cmp cuSZi

python3 script_data_collection.py --input ~/datasets/100x500x500 --output ~/datasets/100x500x500 --dims 500 500 100 --cmp cuSZi
python3 script_data_collection.py --input ~/datasets/SDRBENCH-EXASKY-NYX-512x512x512 --output ~/datasets/SDRBENCH-EXASKY-NYX-512x512x512 --dims 512 512 512 --cmp cuSZi
python3 script_data_collection.py --input ~/datasets/SDRBENCH-Miranda-256x384x384 --output ~/datasets/SDRBENCH-Miranda-256x384x384 --dims 384 384 256 --cmp cuSZi
python3 script_data_collection.py --input ~/datasets/SDRBENCH-SCALE_98x1200x1200 --output ~/datasets/SDRBENCH-SCALE_98x1200x1200 --dims 1200 1200 98 --cmp cuSZi
