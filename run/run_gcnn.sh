model_path='./models/test'
output_path='./outputs/test'
log_path='./glogs/test'
seed=0
data_path='/GPFS/public/AR/THUMOS14'
supervision='self'
# CHANGE PATH!!!!!
# supervision_path='/GPFS/data/yimingqin/code/WTAL-Uncertainty-Modeling/outputs/UM-val/val_pred_25.pickle'
supervision_path='/GPFS/data/yimingqin/code/WTAL-Uncertainty-Modeling/dataset/thumos_annotations/val_gt_25.pickle'
batch_size='4'
N='4'
thres='0.2'
thres_down='-1'
lmbd='1'
neg_lmbd='0'
bkg_lmbd='0'
gcnn_weight='1'
test_dataset='val'
test_head='wtal'
dynamic='0'

CUDA_VISIBLE_DEVICES=1 python -W ignore ./main.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --seed ${seed} --data_path ${data_path} --supervision ${supervision} --supervision_path ${supervision_path} --thres ${thres} --thres_down ${thres_down} --lmbd ${lmbd} --neg_lmbd ${neg_lmbd} --bkg_lmbd ${bkg_lmbd} --test_dataset ${test_dataset} --test_head ${test_head} --batch_size ${batch_size} --N ${N}  --gcnn_weight ${gcnn_weight} --dynamic ${dynamic}
