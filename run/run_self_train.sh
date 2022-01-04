model_path='./models/UM-test'
output_path='./outputs/UM-test'
log_path='./logs/UM-test'
seed=0
data_path='/DATA7_DB7/data/cju/20/BaSNet/dataset/THUMOS14'
supervision='self'
# CHANGE PATH!!!!!
# supervision_path='/GPFS/data/yimingqin/code/WTAL-Uncertainty-Modeling/outputs/UM-val/val_pred_25.pickle'
supervision_path='/GPFS/data/yimingqin/code/WTAL-Uncertainty-Modeling/dataset/thumos_annotations/val_gt_25.pickle'
thres='0.2'
lmbd='0.1'
bkg_lmbd='5.'
test_dataset='val'

python -W ignore ./main.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --seed ${seed} --data_path ${data_path} --supervision ${supervision} --supervision_path ${supervision_path} --thres ${thres} --lmbd ${lmbd} --bkg_lmbd ${bkg_lmbd} --test_dataset ${test_dataset}
