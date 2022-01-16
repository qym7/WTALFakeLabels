model_path='./models/UM-val-self-mAP-01-BCE-09-07-drop'
output_path='./outputs/UM-val-self-mAP-01-BCE-09-07-drop'
log_path='./logs/UM-val-self-mAP-01-BCE-09-07-drop'
seed=0
data_path='/DATA7_DB7/data/cju/20/BaSNet/dataset/THUMOS14'
supervision='self'
# CHANGE PATH!!!!!
supervision_path='/GPFS/data/yimingqin/code/WTAL-Uncertainty-Modeling/outputs/UM-val/val_pred_25.pickle'
# supervision_path='/GPFS/data/yimingqin/code/WTAL-Uncertainty-Modeling/dataset/thumos_annotations/val_gt_25.pickle'
thres='0.2'
thres_down='-1'
lmbd='0.1'
neg_lmbd='0.'
bkg_lmbd='0.'
test_dataset='val'
test_head='sup'

CUDA_VISIBLE_DEVICES=2 python -W ignore ./main.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --seed ${seed} --data_path ${data_path} --supervision ${supervision} --supervision_path ${supervision_path} --thres ${thres} --thres_down ${thres_down} --lmbd ${lmbd} --neg_lmbd ${neg_lmbd} --bkg_lmbd ${bkg_lmbd} --test_dataset ${test_dataset} --test_head ${test_head}
