#!/bin/sh

# 获取脚本所在的目录
script_dir="$(dirname "$0")"

# 切换到脚本所在目录
cd "$script_dir"

# bash ./mmsegmentation/tools/dist_train.sh ./segformer_b3_20x.py 2 --work-dir ./work_dirs/segformer_b3_20x
bash ./mmsegmentation/tools/dist_test.sh ./segformer_b3_20x.py ./work_dirs/segformer_b3_20x/iter_298859.pth 2 --out ./work_dirs/segformer_b3_20x/result.pkl