echo "Starting the script..."
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=GLDCformer

if [ ! -d "./logs/$model_name" ]; then
    mkdir ./logs/$model_name
fi

seq_len=96

root_path_name=./dataset/traffic
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

for pred_len in 96 192 336 720
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --cycle 168 \
      --d_model 512 \
      --d_ff 512 \
      --train_epochs 30 \
      --patience 3 \
      --dropout 0.1 \
      --batch_size 16 \
      --learning_rate 0.001 \
      --exp_name 'init' \
      --ffn_layers 1 \
      --e_layers 2 \
      --bias 1.0 \
      --itr 1 >logs/$model_name/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done