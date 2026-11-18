#!/bin/sh

# 获取脚本所在的目录
script_dir="$(dirname "$0")"

# 切换到脚本所在目录
cd "$script_dir"

bash ./mmsegmentation/tools/dist_train.sh ./segformer_b3_10x.py 8 --work-dir ./work_dirs/segformer_b3_10x
# bash ./mmsegmentation/tools/dist_test.sh ./segformer_b3_10x.py ./work_dirs/segformer_b3_10x/iter_229689.pth 4 --out ./work_dirs/segformer_b3_10x/result.pkl