import argparse
import torch
import random
import numpy as np
from experiments.exp_long_term_forecast import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_init import Exp_Long_Term_Forecast as Exp_Long_Term_Forecast_Init
from utils.tools import print_args

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CycleFormer')

    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='CycleFormer', help='model name')
    parser.add_argument('--exp_name', type=str, required=True, default='init', help='experiment name')
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')

    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task')
    parser.add_argument('--target', type=str, default='OT', help='target feature')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)


    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')

    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention')
    parser.add_argument('--use_norm', type=int, default=1, help='use RevIN')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')

    parser.add_argument('--attn_layers', type=int, default=1, help='number of attention layers')
    parser.add_argument('--ffn_layers', type=int, default=1, help='number of ffn layers')
    parser.add_argument('--cycle', type=int, default=24, help='number of cycles')
    parser.add_argument('--use_tq', type=int, default=1, help='use of temporal query')
    parser.add_argument('--n_norm', type=int, default=1, help='use of LayersNorm')
    parser.add_argument('--num_cycles', type=int, default=10, help='number of cycle')
    parser.add_argument('--use_prior_init', type=int, default=0, help='use of prior init')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--bias', type=float, default=1.0, help='bias')

    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')  # 稍微大点
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.1, help='percent')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

    args = parser.parse_args()

    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print_args(args)

    if args.exp_name == 'exp':
        Exp = Exp_Long_Term_Forecast
    elif args.exp_name == 'init':
        Exp = Exp_Long_Term_Forecast_Init
    else:
        Exp = Exp_Long_Term_Forecast_Init

    if args.is_training:
        for ii in range(args.itr):
            setting = '{}_{}_{}_{}_{}_{}_{}_{}'.format(
                args.model,
                args.model_id,
                args.d_model,
                args.d_ff,
                args.e_layers,
                args.ffn_layers,
                args.seed,
                ii
            )

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_{}_{}_{}_{}'.format(
            args.model,
            args.model_id,
            args.d_model,
            args.d_ff,
            args.e_layers,
            args.ffn_layers,
            args.seed,
            ii
        )

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
