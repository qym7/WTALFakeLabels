output_path='./outputs/UM-val'
log_path='./logs/UM-val'
model_path='./models/UM-val'
model_file='./models/UM-val/model_seed_0.pkl'
data_path="/DATA7_DB7/data/cju/20/BaSNet/dataset/THUMOS14"
test_dataset='val'
save='1'

CUDA_VISIBLE_DEVICES=0 python -W ignore ./main_test.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --model_file ${model_file} --data_path ${data_path} --test_dataset ${test_dataset} --save ${save}
