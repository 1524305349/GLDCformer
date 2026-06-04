echo "Starting the script..."
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=GLDCformer

if [ ! -d "./logs/$model_name" ]; then
    mkdir ./logs/$model_name
fi

seq_len=96

root_path_name=./dataset/solar/
data_path_name=solar.txt
model_id_name=Solar
data_name=Solar

pred_len=96
python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 137 \
  --cycle 144 \
  --use_prior_init 0 \
  --d_model 512 \
  --d_ff 512 \
  --train_epochs 20 \
  --patience 3 \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --exp_name 'init' \
  --ffn_layers 1 \
  --e_layers 2 \
  --bias 0.0 \
  --use_norm 0 \
  --itr 1 >logs/$model_name/$model_name'_'$model_id_name'_'$seq_len.log

pred_len=192
python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 137 \
  --cycle 144 \
  --use_prior_init 0 \
  --d_model 512 \
  --d_ff 512 \
  --train_epochs 20 \
  --patience 3 \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --exp_name 'init' \
  --ffn_layers 2 \
  --e_layers 2 \
  --bias 0.0 \
  --use_norm 0 \
  --itr 1 >logs/$model_name/$model_name'_'$model_id_name'_'$seq_len.log

pred_len=336
python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 137 \
  --cycle 144 \
  --use_prior_init 1 \
  --d_model 512 \
  --d_ff 512 \
  --train_epochs 20 \
  --patience 3 \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --exp_name 'init' \
  --ffn_layers 1 \
  --e_layers 1 \
  --bias 0.5 \
  --use_norm 0 \
  --itr 1 >logs/$model_name/$model_name'_'$model_id_name'_'$seq_len.log

pred_len=720
python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 137 \
  --cycle 144 \
  --use_prior_init 0 \
  --d_model 512 \
  --d_ff 512 \
  --train_epochs 20 \
  --patience 3 \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --exp_name 'init' \
  --ffn_layers 1 \
  --e_layers 1 \
  --bias 0.5 \
  --use_norm 0 \
  --itr 1 >logs/$model_name/$model_name'_'$model_id_name'_'$seq_len.log