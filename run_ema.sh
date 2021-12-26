model_path='./models/UM-EMA'
###
 # @Author: your name
 # @Date: 2021-12-26 11:45:45
 # @LastEditTime: 2021-12-26 11:47:59
 # @LastEditors: Please set LastEditors
 # @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 # @FilePath: /yimingqin/code/WTAL-Uncertainty-Modeling/run_ema.sh
### 
output_path='./outputs/UM-EMA'
log_path='./logs/UM-EMA'
seed=0
data_path='/DATA7_DB7/data/cju/20/BaSNet/dataset/THUMOS14'
supervision='self'
supervision_path='/GPFS/data/yimingqin/code/WTAL-Uncertainty-Modeling/outputs/UM/val_pred_25.pickle'
thres='0.2'
lmbd='10'
test_dataset='val'
m='0.9'
gamma='100'

CUDA_VISIBLE_DEVICES=0 python -W ignore ./main_ema.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --seed ${seed} --data_path ${data_path} --supervision ${supervision} --supervision_path ${supervision_path} --thres ${thres} --lmbd ${lmbd} --test_dataset ${test_dataset} --m ${m} --gamma ${gamma}