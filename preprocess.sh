#!/bin/sh
# !!! ----------------- Please change the `src_dir` to the location of your dataset. ----------------- !!!
# !!! ----- Please do NOT change `dst_dir` as it will be used to locate the dataset in crowd.py. ----- !!!

python preprocess.py --dataset shanghaitech_a --src_dir ./data/ShanghaiTech/part_A --dst_dir ./data/sha  --min_size 448 --max_size 4096
python preprocess.py --dataset shanghaitech_b --src_dir ./data/ShanghaiTech/part_B --dst_dir ./data/shb  --min_size 448 --max_size 4096
