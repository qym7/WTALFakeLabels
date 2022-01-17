model_path='./models/UM-val-n'
output_path='./outputs/UM-val-n'
log_path='./logs/UM-val-n'
model_file='./models/UM-val-n/model_seed_0.pkl'
seed=0
data_path='/DATA7_DB7/data/cju/20/BaSNet/dataset/THUMOS14'
test_dataset='val'

CUDA_VISIBLE_DEVICES=2 python -W ignore ./main.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --seed ${seed} --data_path ${data_path} --test_dataset ${test_dataset}
CUDA_VISIBLE_DEVICES=2 python -W ignore ./main_test.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --model_file ${model_file} --data_path ${data_path} --test_dataset ${test_dataset} --save '1'
