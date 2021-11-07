###
 # @Author: your name
 # @Date: 2021-09-28 10:55:26
 # @LastEditTime: 2021-11-07 12:39:57
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /mdap-multi-task/src/train_first_stage.sh
###

# 文件路径
data_dir="/mnt/nfsshare/wangfei/project/server_multitask/data/origin"
data_save_dir="/mnt/nfsshare/wangfei/project/server_multitask/model/"
output_dir="/mnt/nfsshare/wangfei/project/server_multitask/data/output/"
tokenizer_dir="/mnt/nfsshare/wangfei/pre_model/chinese_wwm_ext_pytorch/"
model_name_or_path="/mnt/nfsshare/wangfei/pre_model/chinese_wwm_ext_pytorch/"

# 定位conda
source /mnt/nfsshare/miniconda3/etc/profile.d/conda.sh

# 切换环境
conda activate wf_pytorch

CUDA_VISIBLE_DEVICES=2 python3 -m train_semh \
  --data_dir ${data_dir} \
  --data_save_dir ${data_save_dir} \
  --output_dir ${output_dir} \
  --model_name_or_path ${model_name_or_path} \
  --tokenizer_dir ${tokenizer_dir} \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --weight_decay 0.0 \
  --warmup_steps 0 \
  --num_train_epochs 3 \
  --logging_dir /data/christmas.wang/project/mdap/log/ \
  --logging_first_step \
  --logging_steps 1000 \
  --save_steps 2500 \
  --seed 20 \
  --do_train \
  --do_predict False\
  --do_eval False\
  --is_debug False \
  # --eval_steps 500 \
  # --report_to "wandb" \
  # --evaluation_strategy "steps" \
  # --load_best_model_at_end True \
  # --overwrite_cache True