echo "Starting the script..."
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=GLDC_CycleNet

if [ ! -d "./logs/$model_name" ]; then
    mkdir ./logs/$model_name
fi

seq_len=96

root_path_name=./dataset/PEMS/
data_path_name=PEMS03.npz
model_id_name=PEMS03
data_name=PEMS

for pred_len in 12 24 48 96
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
      --enc_in 358 \
      --cycle 288 \
      --train_epochs 30 \
      --patience 4 \
      --dropout 0.1 \
      --batch_size 32 \
      --learning_rate 0.001 \
      --exp_name 'init' \
      --use_norm 0 \
      --itr 1 >logs/$model_name/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
