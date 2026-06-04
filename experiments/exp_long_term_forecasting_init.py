from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


def generate_empirical_prior(train_data, cycle_len, num_cycles=10, use_norm=True):
    if not isinstance(train_data, torch.Tensor):
        train_data = torch.tensor(train_data, dtype=torch.float32)

    C = train_data.shape[1]
    req_len = cycle_len * num_cycles

    actual_len = min(req_len, (train_data.shape[0] // cycle_len) * cycle_len)
    cycles_to_use = actual_len // cycle_len

    warmup_data = train_data[:actual_len, :].clone()

    if use_norm:
        seq_mean = torch.mean(warmup_data, dim=0, keepdim=True)
        seq_var = torch.var(warmup_data, dim=0, keepdim=True) + 1e-5
        processed_data = (warmup_data - seq_mean) / torch.sqrt(seq_var)
    else:
        processed_data = warmup_data

    folded_data = processed_data.view(cycles_to_use, cycle_len, C)
    prior_matrix = torch.mean(folded_data, dim=0)

    return prior_matrix

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        if self.args.use_prior_init:
            train_data, _ = self._get_data(flag='train')
            cycle_length = self.args.cycle
            prior_matrix = generate_empirical_prior(
                train_data=train_data.data_x,
                cycle_len=cycle_length,
                num_cycles=self.args.num_cycles,
                use_norm=self.args.use_norm
            )
            self.args.prior_init = prior_matrix

        model = self.model_dict[self.args.model].Model(self.args).float()
        # # ================== 👇 新增：模型规模统计 👇 ==================
        # total_params = sum(p.numel() for p in model.parameters())
        # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # # 参数内存占用 (以 Float32 计算，每个参数 4 字节)
        # param_memory_mb = (total_params * 4) / (1024 * 1024)
        #
        # print(f"\n" + "=" * 30)
        # print(f"Model: {self.args.model}")
        # print(f"Total Parameters: {total_params:,}")
        # print(f"Trainable Parameters: {trainable_params:,}")
        # print(f"Parameter Memory (Theory): {param_memory_mb:.2f} MB")
        # print("=" * 30 + "\n")
        # # ============================================================
        # # ================== 👇 智能名称感知的 MACs 统计 👇 ==================
        # try:
        #     import copy
        #     import inspect
        #     from thop import profile
        #
        #     # 1. 备份随机状态，确保 MSE/MAE 不变
        #     cpu_rng_state = torch.get_rng_state()
        #     if torch.cuda.is_available():
        #         gpu_rng_state = torch.cuda.get_rng_state_all()
        #
        #     with torch.no_grad():
        #         model_temp = copy.deepcopy(model)
        #         model_temp.eval()
        #         device_temp = torch.device('cpu')
        #         model_temp.to(device_temp)
        #
        #         # 2. 构造各种可能的模拟输入
        #         dummy_x = torch.randn(1, self.args.seq_len, self.args.enc_in).to(device_temp)
        #         dummy_cycle = torch.zeros(1).long().to(device_temp)
        #         # 某些模型可能需要 dec_input 形状
        #         dummy_dec = torch.randn(1, self.args.label_len + self.args.pred_len, self.args.enc_in).to(device_temp)
        #
        #         # 3. 🌟 核心：根据参数名称自动填充
        #         sig = inspect.signature(model.forward)
        #         input_list = []
        #
        #         for i, (name, param) in enumerate(sig.parameters.items()):
        #             # a. 匹配主输入 (x, x_enc)
        #             if i == 0 or name in ['x', 'x_enc']:
        #                 input_list.append(dummy_x)
        #             # b. 匹配周期输入 (cycle, cycle_index)
        #             elif 'cycle' in name.lower():
        #                 input_list.append(dummy_cycle)
        #             # c. 匹配解码器输入 (x_dec)
        #             elif name == 'x_dec':
        #                 input_list.append(dummy_dec)
        #             # d. 匹配掩码或标记位 (x_mark, mask) -> 给 None
        #             elif any(k in name.lower() for k in ['mark', 'mask']):
        #                 input_list.append(None)
        #             # e. 如果是必填参数且不属于上述情况，尝试给 None
        #             elif param.default is inspect.Parameter.empty:
        #                 input_list.append(None)
        #             # f. 有默认值的可选参数可以不传
        #             else:
        #                 break  # 后续可选参数不影响 positional 调用
        #
        #         # 4. 执行分析
        #         macs, params = profile(model_temp, inputs=tuple(input_list), verbose=False)
        #
        #         print("\n" + "*" * 45)
        #         print(f"  Complexity Analysis for {self.args.model}")
        #         print(f"  - MACs (FLOPs) : {macs / 1e9:.4f} G")
        #         print(f"  - Parameters   : {params / 1e6:.4f} M")
        #         print("*" * 45 + "\n")
        #
        #         del model_temp
        #
        #     # 5. 还原随机状态
        #     torch.set_rng_state(cpu_rng_state)
        #     if torch.cuda.is_available():
        #         torch.cuda.set_rng_state_all(gpu_rng_state)
        #     if torch.cuda.is_available(): torch.cuda.empty_cache()
        #
        # except Exception as e:
        #     print(f"\n[Warning] Profiling MACs failed for {self.args.model}: {e}")
        # # =====================================================================

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = 0
        total_samples = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_cycle = batch_cycle.long().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cycle_index=batch_cycle)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cycle_index=batch_cycle)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cycle_index=batch_cycle)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cycle_index=batch_cycle)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                batch_size = pred.shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        avg_loss = total_loss / total_samples
        self.model.train()
        return avg_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = None
        if self.args.lradj == 'TST':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=model_optim,
                steps_per_epoch=len(train_loader),
                pct_start=self.args.pct_start if hasattr(self.args, 'pct_start') else 0.3,
                epochs=self.args.train_epochs,
                max_lr=self.args.learning_rate
            )

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        if self.args.lradj != 'TST':
            adjust_learning_rate(model_optim, 0, self.args, scheduler)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            epoch_total_loss = 0
            epoch_total_samples = 0

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_cycle = batch_cycle.long().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cycle_index=batch_cycle)[
                                0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cycle_index=batch_cycle)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cycle_index=batch_cycle)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cycle_index=batch_cycle)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)

                batch_size = outputs.shape[0]
                epoch_total_loss += loss.item() * batch_size
                epoch_total_samples += batch_size

                if (i + 1) % 300 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = epoch_total_loss / epoch_total_samples

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # adjust_learning_rate(model_optim, epoch + 1, self.args, scheduler=scheduler)

            if self.args.lradj == 'TST':
                adjust_learning_rate(model_optim, epoch + 1, self.args, scheduler=scheduler)
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args, scheduler=scheduler)


        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_cycle = batch_cycle.long().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cycle_index=batch_cycle)[
                                0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cycle_index=batch_cycle)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cycle_index=batch_cycle)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cycle_index=batch_cycle)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_cycle = batch_cycle.long().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cycle_index=batch_cycle)[
                                0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cycle_index=batch_cycle)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cycle_index=batch_cycle)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cycle_index=batch_cycle)

                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.concatenate(preds, axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return