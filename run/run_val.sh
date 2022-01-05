output_path='./outputs/UM'
log_path='./logs/UM'
model_path='./models/UM'
model_file='./models/UM/model_seed_0.pkl'
data_path='/DATA7_DB7/data/cju/20/BaSNet/dataset/THUMOS14'
test_dataset='val'
save='1'

python -W ignore ./main_test.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --model_file ${model_file} --data_path ${data_path} --test_dataset ${test_dataset} --save ${save}
