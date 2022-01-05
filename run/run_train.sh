model_path='./models/UM-val-n'
output_path='./outputs/UM-val-n'
log_path='./logs/UM-val-n'
seed=0
data_path='/DATA7_DB7/data/cju/20/BaSNet/dataset/THUMOS14'
test_dataset='val'

python -W ignore ./main.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --seed ${seed} --data_path ${data_path} --test_dataset ${test_dataset}
