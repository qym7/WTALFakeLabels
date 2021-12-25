model_path='./models/UM-EMA'
output_path='./outputs/UM-EMA'
log_path='./logs/UM-EMA'
seed=0
data_path='/DATA7_DB7/data/cju/20/BaSNet/dataset/THUMOS14'
supervision='self'
supervision_path='/GPFS/data/yimingqin/code/WTAL-Uncertainty-Modeling/outputs/UM/val_pred_25.pickle'
thres='0.2'
a1='10'
a2='10'
test_dataset='val'
m='0.9'
gamma='100'

CUDA_VISIBLE_DEVICES=0 python -W ignore ./main_ema.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --seed ${seed} --data_path ${data_path} --supervision ${supervision} --supervision_path ${supervision_path} --thres ${thres} --a1 ${a1} --a2 ${a2} --test_dataset ${test_dataset} --m ${m} --gamma ${gamma}