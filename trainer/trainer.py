import numpy as np
import torch
from tqdm import tqdm
from typing import List
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop
import sys
import torch.nn.functional as F
class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, reparametrization_net, train_criterion, metrics, optimizer, optimizer_loss, config, data_loader,
                 valid_data_loader=None, test_data_loader=None, lr_scheduler=None, lr_scheduler_overparametrization=None, len_epoch=None, val_criterion=None):
        super().__init__(model, reparametrization_net, train_criterion, metrics, optimizer, optimizer_loss, config, val_criterion)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader

        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.do_test = self.test_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_overparametrization = lr_scheduler_overparametrization
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.train_loss_list: List[float] = []
        self.val_loss_list: List[float] = []
        self.test_loss_list: List[float] = []

        self.train_criterion = train_criterion

        self.new_best_val = False
        self.val_acc = 0
        self.test_val_acc = 0

        

    def _eval_metrics(self, output, label):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, label)
            if self.writer is not None:
                self.writer.add_scalar({'{}'.format(metric.__name__): acc_metrics[i]})
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        if self.reparametrization_net is not None:
            self.reparametrization_net.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        noise_level = 0


        with tqdm(self.data_loader) as progress:
            for batch_idx, (data, data2, label, indexs, _) in enumerate(progress):
                progress.set_description_str(f'Train epoch {epoch}')


                data, label = data.to(self.device), label.long().to(self.device)

                target = torch.zeros(len(label), self.config['num_classes']).to(self.device).scatter_(1, label.view(-1,1), 1)
    

                if self.config['train_loss']['args']['ratio_consistency'] > 0:
                    data2 = data2.to(self.device)
                    data_all = torch.cat([data, data2]).cuda()
                else:
                    data_all = data
                    
                output = self.model(data_all)
                

                loss = self.train_criterion(indexs, output, target)

                self.optimizer_loss.zero_grad()
                self.optimizer.zero_grad()
                
                
                loss.backward()

                self.optimizer_loss.step()
                self.optimizer.step()


                if self.config['train_loss']['args']['ratio_consistency'] > 0:
                    output, _ = torch.chunk(output, 2)

                if self.writer is not None:

                    self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, epoch=epoch)           
                    self.writer.add_scalar({'loss': loss.item()})

                
                self.train_loss_list.append(loss.item())
                total_loss += loss.item()
                total_metrics += self._eval_metrics(output, label)


                if batch_idx % self.log_step == 0:
                    progress.set_postfix_str(' {} Loss: {:.6f}'.format(
                        self._progress(batch_idx),
                        loss.item()))

                if batch_idx == self.len_epoch:
                    break


        log = {
            'loss': total_loss / self.len_epoch,
            'noise level': noise_level/ self.len_epoch,
            'metrics': (total_metrics / self.len_epoch).tolist(),
            'learning rate': self.lr_scheduler.get_lr()
        }


        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)
        if self.do_test:
            test_log = self._test_epoch(epoch)
            log.update(test_log)



        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        # if self.lr_scheduler_overparametrization is not None:
        #     self.lr_scheduler_overparametrization.step()

        return log


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        if self.reparametrization_net is not None:
            self.reparametrization_net.eval()

        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            with tqdm(self.valid_data_loader) as progress:
                for batch_idx, (data, label, indexs, _) in enumerate(progress):
                    progress.set_description_str(f'Valid epoch {epoch}')
                    data, label = data.to(self.device), label.to(self.device)
                    output = self.model(data)
                    if self.reparametrization_net is not None:
                        output, original_output = self.reparametrization_net(output, indexs)
                    loss = self.val_criterion(output, label)

                    if self.writer is not None:
                        self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, epoch=epoch, mode = 'valid')
                        self.writer.add_scalar({'loss': loss.item()})
                    self.val_loss_list.append(loss.item())
                    total_val_loss += loss.item()
                    total_val_metrics += self._eval_metrics(output, label)
                    # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        val_acc = (total_val_metrics / len(self.valid_data_loader)).tolist()[0]
        if val_acc > self.val_acc:
            self.val_acc = val_acc
            self.new_best_val = True
            if self.writer is not None:
                self.writer.add_scalar({'Best val acc': self.val_acc}, epoch = epoch)
        else:
            self.new_best_val = False

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }

    def _test_epoch(self, epoch):
        """
        Test after training an epoch

        :return: A log that contains information about test

        Note:
            The Test metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        if self.reparametrization_net is not None:
            self.reparametrization_net.eval()
        total_test_loss = 0
        total_test_metrics = np.zeros(len(self.metrics))
        results = np.zeros((len(self.test_data_loader.dataset), self.config['num_classes']), dtype=np.float32)
        tar_ = np.zeros((len(self.test_data_loader.dataset),), dtype=np.float32)
        with torch.no_grad():
            with tqdm(self.test_data_loader) as progress:
                for batch_idx, (data, label,indexs,_) in enumerate(progress):
                    progress.set_description_str(f'Test epoch {epoch}')
                    data, label = data.to(self.device), label.to(self.device)
                    output = self.model(data)
                    if self.reparametrization_net is not None:
                        output, original_output = self.reparametrization_net(output, indexs)
                    loss = self.val_criterion(output, label)
                    if self.writer is not None:
                        self.writer.set_step((epoch - 1) * len(self.test_data_loader) + batch_idx, epoch=epoch, mode = 'test')
                        self.writer.add_scalar({'loss': loss.item()})
                    self.test_loss_list.append(loss.item())
                    total_test_loss += loss.item()
                    total_test_metrics += self._eval_metrics(output, label)
                    # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                    results[indexs.cpu().detach().numpy().tolist()] = output.cpu().detach().numpy().tolist()
                    tar_[indexs.cpu().detach().numpy().tolist()] = label.cpu().detach().numpy().tolist()

        # add histogram of model parameters to the tensorboard
        top_1_acc = (total_test_metrics / len(self.test_data_loader)).tolist()[0]
        if self.new_best_val:
            self.test_val_acc = top_1_acc
            if self.writer is not None:
                self.writer.add_scalar({'Test acc with best val': top_1_acc}, epoch = epoch)
        if self.writer is not None:
            self.writer.add_scalar({'Top-1': top_1_acc}, epoch = epoch)
            self.writer.add_scalar({'Top-5': (total_test_metrics / len(self.test_data_loader)).tolist()[1]}, epoch = epoch)

        return {
            'test_loss': total_test_loss / len(self.test_data_loader),
            'test_metrics': (total_test_metrics / len(self.test_data_loader)).tolist()
        }



    def _warmup_epoch(self, epoch):
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        self.model.train()
        if self.reparametrization_net is not None:
            self.reparametrization_net.eval()

        data_loader = self.data_loader#self.loader.run('warmup')


        with tqdm(data_loader) as progress:
            for batch_idx, (data, _, label, indexs , _) in enumerate(progress):
                progress.set_description_str(f'Warm up epoch {epoch}')

                data, label = data.to(self.device), label.long().to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                if self.reparametrization_net is not None:
                    output, original_output = self.reparametrization_net(output, indexs)
                out_prob = torch.nn.functional.softmax(output).data.detach()

                loss = torch.nn.functional.cross_entropy(output, label)

                loss.backward() 
                self.optimizer.step()
                if self.writer is not None:
                    self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, epoch=epoch)
                    self.writer.add_scalar({'loss_record': loss.item()})
                self.train_loss_list.append(loss.item())
                total_loss += loss.item()
                total_metrics += self._eval_metrics(output, label)


                if batch_idx % self.log_step == 0:
                    progress.set_postfix_str(' {} Loss: {:.6f}'.format(
                        self._progress(batch_idx),
                        loss.item()))
                    # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                if batch_idx == self.len_epoch:
                    break
        if hasattr(self.data_loader, 'run'):
            self.data_loader.run()
        log = {
            'loss': total_loss / self.len_epoch,
            'metrics': (total_metrics / self.len_epoch).tolist(),
            'learning rate': self.lr_scheduler.get_lr()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)
        if self.do_test:
            test_log = self._test_epoch(epoch)
            log.update(test_log)

        return log


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
