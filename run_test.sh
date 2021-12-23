###
 # @Author: Yiming Qin
 # @Date: 2021-12-18 19:59:02
 # @LastEditTime: 2021-12-23 11:50:25
 # @LastEditors: Please set LastEditors
 # @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 # @FilePath: /yimingqin/code/WTAL-Uncertainty-Modeling/run_eval.sh
###

output_path="./outputs/UM"
log_path="./logs/UM"
model_path="./models/UM_eval"
model_file='./models/UM/model_seed_0.pkl'
data_path="/DATA7_DB7/data/cju/20/BaSNet/dataset/THUMOS14"
test_dataset='test'

CUDA_VISIBLE_DEVICES=0 python -W ignore ./main_test.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --model_file ${model_file} --data_path ${data_path} --test_dataset ${test_dataset}
