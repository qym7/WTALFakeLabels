model_path='./models/UM-val-1-05-EMA-5-10-BCE-099'
output_path='./outputs/UM-val-1-05-EMA-5-10-BCE-099'
log_path='./logs/UM-val-1-05-EMA-5-20-BCE-099'
model_file='./models/UM-val-1-05-EMA-5-20-BCE-099/model_seed_0.pkl'
seed=0
data_path='/DATA7_DB7/data/cju/20/BaSNet/dataset/THUMOS14'
supervision='self'
# CHANGE PATH!!!!!
supervision_path='/GPFS/data/yimingqin/code/WTAL-Uncertainty-Modeling/outputs/UM-val/val_pred_25.pickle'
# supervision_path='/GPFS/data/yimingqin/code/WTAL-Uncertainty-Modeling/dataset/thumos_annotations/val_gt_25.pickle'
thres='0.2'
thres_down='-1'
lmbd='5.'
neg_lmbd='0.'
bkg_lmbd='10.'
test_dataset='val'
test_head='sup'
# Do not change ema
ema='1'
m='0.99'
gamma_f='1'
gamma_c='0.5'

CUDA_VISIBLE_DEVICES=7 python -W ignore ./main.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --seed ${seed} --data_path ${data_path} --supervision ${supervision} --supervision_path ${supervision_path} --thres ${thres} --thres_down ${thres_down} --lmbd ${lmbd} --neg_lmbd ${neg_lmbd} --bkg_lmbd ${bkg_lmbd} --test_dataset ${test_dataset} --test_head ${test_head}  --m ${m} --gamma_f ${gamma_f} --gamma_c ${gamma_c} --ema ${ema}
CUDA_VISIBLE_DEVICES=7 python -W ignore ./main_test.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --model_file ${model_file} --data_path ${data_path} --test_dataset ${test_dataset} --save '1' --supervision ${supervision}
