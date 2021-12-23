model_path="./models/UM-IoU"
output_path="./outputs/UM-IoU"
log_path="./logs/UM-IoU"
seed=0
data_path="/DATA7_DB7/data/cju/20/BaSNet/dataset/THUMOS14"

CUDA_VISIBLE_DEVICES=0 python -W ignore ./main.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --seed ${seed} --data_path ${data_path}
