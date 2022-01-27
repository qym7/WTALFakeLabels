model_path='./models/UM-val-self-mAP-sigm-01-1-BCE'
output_path='./outputs/UM-val-self-mAP-sigm-01-1-BCE'
log_path='./logs/UM-val-self-mAP-sigm-01-1-BCE'
seed=0
data_path='/GPFS/public/AR/THUMOS14'
supervision='self'
# CHANGE PATH!!!!!
supervision_path='/GPFS/data/yimingqin/code/WTAL-Uncertainty-Modeling/outputs/UM-val/val_pred_25.pickle'
# supervision_path='/GPFS/data/yimingqin/code/WTAL-Uncertainty-Modeling/dataset/thumos_annotations/val_gt_25.pickle'
thres='0.2'
thres_down='-1'
lmbd='0.1'
neg_lmbd='0'
bkg_lmbd='1'
test_dataset='val'
test_head='sup'

CUDA_VISIBLE_DEVICES=4 python -W ignore ./main.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --seed ${seed} --data_path ${data_path} --supervision ${supervision} --supervision_path ${supervision_path} --thres ${thres} --thres_down ${thres_down} --lmbd ${lmbd} --neg_lmbd ${neg_lmbd} --bkg_lmbd ${bkg_lmbd} --test_dataset ${test_dataset} --test_head ${test_head}
