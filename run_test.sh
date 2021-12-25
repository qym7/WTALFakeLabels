output_path='./outputs/UM'
###
 # @Author: your name
 # @Date: 2021-12-25 20:15:51
 # @LastEditTime: 2021-12-25 20:15:51
 # @LastEditors: your name
 # @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 # @FilePath: /yimingqin/code/WTAL-Uncertainty-Modeling/run_test.sh
### 
log_path='./logs/UM'
model_path='./models/UM'
model_file='./models/UM/model_seed_0.pkl'
data_path="/DATA7_DB7/data/cju/20/BaSNet/dataset/THUMOS14"
test_dataset='test'

CUDA_VISIBLE_DEVICES=0 python -W ignore ./main_test.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --model_file ${model_file} --data_path ${data_path} --test_dataset ${test_dataset}
