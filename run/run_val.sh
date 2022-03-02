output_path='./test_outputs/UM-filter1-08'
log_path='./logs/UM-average'
model_path='./models/UM-average'
model_file='./models/UM-average/model_seed_0.pkl'
data_path='/DATA7_DB7/data/cju/20/BaSNet/dataset/THUMOS14'
test_dataset='val'
save='0'

CUDA_VISIBLE_DEVICES=0 python -W ignore ./main_test.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --model_file ${model_file} --data_path ${data_path} --test_dataset ${test_dataset} --save ${save}
