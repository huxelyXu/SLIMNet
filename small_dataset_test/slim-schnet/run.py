
import time
import os
import torch
from torch.optim import Adam
from torch_geometric.data import DataLoader
import numpy as np
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import logging
import time

if  __name__ == '__main__':
    logger = logging.getLogger()
    time = time.asctime()
    logger.setLevel(logging.INFO)   # 设置打印级别
    formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    fh = logging.FileHandler(f"debug "+time+".log", encoding='utf8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logging.info('Start print log......')


class run():
    r"""
    The base script for running different 3DGN methods.
    """
    def __init__(self):
        pass
        
    def run(self, device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation, epochs=500, batch_size=32, vt_batch_size=32, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=50, weight_decay=0, 
        softnet=False, p=100, save_dir='', log_dir=''):
        r"""
        The run script for training and validation.
        
        Args:
            device (torch.device): Device for computation.
            train_dataset: Training data.
            valid_dataset: Validation data.
            test_dataset: Test data.
            model: Which 3DGN model to use. Should be one of the SchNet, DimeNetPP, and SphereNet.
            loss_func (function): The used loss funtion for training.
            evaluation (function): The evaluation function. 
            epochs (int, optinal): Number of total training epochs. (default: :obj:`500`)
            batch_size (int, optinal): Number of samples in each minibatch in training. (default: :obj:`32`)
            vt_batch_size (int, optinal): Number of samples in each minibatch in validation/testing. (default: :obj:`32`)
            lr (float, optinal): Initial learning rate. (default: :obj:`0.0005`)
            lr_decay_factor (float, optinal): Learning rate decay factor. (default: :obj:`0.5`)
            lr_decay_step_size (int, optinal): epochs at which lr_initial <- lr_initial * lr_decay_factor. (default: :obj:`50`)
            weight_decay (float, optinal): weight decay factor at the regularization term. (default: :obj:`0`)
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)    
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy during training. (default: :obj:`100`)
            save_dir (str, optinal): The path to save trained models. If set to :obj:`''`, will not save the model. (default: :obj:`''`)
            log_dir (str, optinal): The path to save log files. If set to :obj:`''`, will not save the log files. (default: :obj:`''`)
        
        """        

        model = model.to(device)
        num_params = sum(p.numel() for p in model.parameters())
        logging.info(f'#Params: {num_params}')
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, vt_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False)
        best_valid = float('inf')
        best_test = float('inf')
            
        if save_dir != '':
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        if log_dir != '':
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            writer = SummaryWriter(log_dir=log_dir)
        
        for epoch in range(1, epochs + 1):
            logging.info("\n=====Epoch {}".format(epoch))
            
            logging.info('\nTraining...')
            train_mae = self.train(model, optimizer, train_loader, softnet, p, loss_func, device)

            logging.info('\nEvaluating...')
            if softnet: 
                mono_mae, valid_mae = self.val(model, valid_loader, softnet, p, evaluation, device)
            else:
                valid_mae = self.val(model, valid_loader, softnet, p, evaluation, device)

            logging.info('\nTesting...')
            if softnet: 
                mono_mae, test_mae = self.val(model, test_loader, softnet, p, evaluation, device)
            else: 
                test_mae = self.val(model, test_loader, softnet, p, evaluation, device)

            # logging.info()
            logging.info(f'Train:{train_mae}, Validation:{valid_mae}, Test:{test_mae}')

            if log_dir != '':
                writer.add_scalar('train_mae', train_mae, epoch)
                writer.add_scalar('valid_mae', valid_mae, epoch)
                writer.add_scalar('test_mae', test_mae, epoch)
            
            if test_mae < best_valid:
                best_valid = test_mae
                best_test = valid_mae
                if save_dir != '':
                    logging.info('Saving checkpoint...')
                    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_valid_mae': best_valid, 'num_params': num_params}
                    torch.save(checkpoint, os.path.join(save_dir, 'valid_checkpoint.pt'))

            scheduler.step()

        logging.info(f'Best validation MAE so far: {best_valid}')
        logging.info(f'Test MAE when got best validation result: {best_test}')
        
        if log_dir != '':
            writer.close()

    def train(self, model, optimizer, train_loader, softnet, p, loss_func, device):
        r"""
        The script for training.
        
        Args:
            model: Which 3DGN model to use. Should be one of the SchNet, DimeNetPP, and SphereNet.
            optimizer (Optimizer): Pytorch optimizer for trainable parameters in training.
            train_loader (Dataloader): Dataloader for training.
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)    
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy during training. (default: :obj:`100`)
            loss_func (function): The used loss funtion for training. 
            device (torch.device): The device where the model is deployed.

        :rtype: Traning loss. ( :obj:`mae`)
        
        """   
        model.train()
        loss_accum = 0
        for step, batch_data in enumerate(train_loader):
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            mono_y, poly_y = model(batch_data)
            if softnet:
                # print(poly_y.shape)
                # print(poly_y.ravel().shape)
                # print(poly_y.squeeze().shape)
                # print(batch_data.y.reshape(len(batch_data.batch.unique()), -1).shape)
                # print(mono_y.shape)
                # print(batch_data.monmer_y.reshape(len(batch_data.batch.unique()), -1).shape)
                # input()
                poly_loss = loss_func(poly_y.ravel(), batch_data.y)
                mono_loss = loss_func(mono_y.ravel(), batch_data.monmer_y)
                loss = 10e6 * poly_loss + mono_loss
            else:

                loss = loss_func(mono_y.ravel(), batch_data.y)

            loss.backward()
            optimizer.step()
            loss_accum += loss.detach().cpu().item()
        return loss_accum / (step + 1)

    def val(self, model, data_loader, softnet, p, evaluation, device):
        r"""
        The script for validation/test.
        
        Args:
            model: Which 3DGN model to use. Should be one of the SchNet, DimeNetPP, and SphereNet.
            data_loader (Dataloader): Dataloader for validation or test.
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)    
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy. (default: :obj:`100`)
            evaluation (function): The used funtion for evaluation.
            device (torch.device, optional): The device where the model is deployed.

        :rtype: Evaluation result. ( :obj:`mae`)
        
        """   
        model.eval()

        preds_poly = torch.Tensor([]).to(device)
        targets_poly = torch.Tensor([]).to(device)

        if softnet:
            preds_mono = torch.Tensor([]).to(device)
            targets_mono = torch.Tensor([]).to(device)
        
        for step, batch_data in enumerate(data_loader):
            batch_data = batch_data.to(device)
            mono_y, poly_y = model(batch_data)
            if softnet:
                preds_mono = torch.cat([preds_mono, mono_y.ravel().detach()], dim=0)
                targets_mono = torch.cat([targets_mono, batch_data.monmer_y], dim=0)

                preds_poly = torch.cat([preds_poly, poly_y.ravel().detach()], dim=0)
                targets_poly = torch.cat([targets_poly, batch_data.y], dim=0)

            else:
                preds_poly = torch.cat([preds_poly, mono_y.ravel().detach()], dim=0)
                targets_poly = torch.cat([targets_poly, batch_data.y], dim=0)

        if softnet:
            input_dict = {"y_true": targets_mono, "y_pred": preds_mono}
            input_dict_force = {"y_true": targets_poly, "y_pred": preds_poly}

            energy_mae = evaluation.eval(input_dict)['mae']
            force_mae = evaluation.eval(input_dict_force)['mae']

            logging.info({'mono MAE': energy_mae, 'poly MAE': force_mae})

            return energy_mae, force_mae

        else:
            input_dict = {"y_true": targets_poly, "y_pred": preds_poly}

            return evaluation.eval(input_dict)['mae']
