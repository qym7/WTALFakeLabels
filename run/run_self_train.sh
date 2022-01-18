model_path='./models/UM-val-self-mAP-mlp-5-10-BCE'
output_path='./outputs/UM-val-self-mAP-mlp-5-10-BCE'
log_path='./logs/UM-val-self-mAP-mlp-5-10-BCE'
# model_file='./models/UM-val-self-mAP-01-BCE-98-100-decay/model_seed_0.pkl'
seed=0
data_path='/DATA7_DB7/data/cju/20/BaSNet/dataset/THUMOS14'
supervision='self'
# CHANGE PATH!!!!!
supervision_path='/GPFS/data/yimingqin/code/WTAL-Uncertainty-Modeling/outputs/UM-val/val_pred_25.pickle'
# supervision_path='/GPFS/data/yimingqin/code/WTAL-Uncertainty-Modeling/dataset/thumos_annotations/val_gt_25.pickle'
thres='0.2'
thres_down='-1'
lmbd='5'
neg_lmbd='0.'
bkg_lmbd='10.'
test_dataset='val'
test_head='sup'

CUDA_VISIBLE_DEVICES=2 python -W ignore ./main.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --seed ${seed} --data_path ${data_path} --supervision ${supervision} --supervision_path ${supervision_path} --thres ${thres} --thres_down ${thres_down} --lmbd ${lmbd} --neg_lmbd ${neg_lmbd} --bkg_lmbd ${bkg_lmbd} --test_dataset ${test_dataset} --test_head ${test_head}
# CUDA_VISIBLE_DEVICES=1 python -W ignore ./main_test.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --model_file ${model_file} --data_path ${data_path} --test_dataset ${test_dataset} --save '1' --supervision ${supervision}
