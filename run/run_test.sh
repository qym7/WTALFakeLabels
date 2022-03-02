output_path='./test-outputs/UM-filter1-09-0206'
log_path='./logs/UM-average'
model_path='./models/UM-average'
model_file='./models/UM-average/model_seed_0.pkl'
data_path='/GPFS/public/AR/THUMOS14'
supervision='self'
supervision_path='/GPFS/data/yimingqin/code/WTAL-Uncertainty-Modeling/outputs/UM-val/val_pred_25.pickle'
# supervision_path='/GPFS/data/yimingqin/code/WTAL-Uncertainty-Modeling/dataset/thumos_annotations/val_gt_25.pickle'
test_dataset='val'
test_head='wtal'
save='0'

CUDA_VISIBLE_DEVICES=0 python -W ignore ./main_test.py --supervision_path ${supervision_path} --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --model_file ${model_file} --data_path ${data_path} --test_dataset ${test_dataset} --test_head ${test_head} --save ${save} --supervision ${supervision}
