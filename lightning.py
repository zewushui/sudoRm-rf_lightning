# -*- encoding: utf-8 -*-
'''
@Filename    :lightning.py
@Time        :2020/07/10 20:27:23
@Author      :Kai Li
@Version     :1.0
'''

import torch
from torch import optim
import pytorch_lightning as pl
from sudormrf import SuDORMRF
import torchaudio
from itertools import permutations


def sisnr(x, s, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

def si_snr_loss(ests, egs):
    # spks x n x S
    refs = egs
    num_spks = len(refs)

    def sisnr_loss(permute):
        # for one permute
        return sum(
            [sisnr(ests[s], refs[t])
             for s, t in enumerate(permute)]) / len(permute)
        # average the value

    # P x N
    N = refs[0].shape[0]
    sisnr_mat = torch.stack(
        [sisnr_loss(p) for p in permutations(range(num_spks))])
    max_perutt, _ = torch.max(sisnr_mat, dim=0)
    # si-snr
    return -torch.sum(max_perutt) / N

def tsdr(x, s, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    
    sdrMax = torch.tensor(30)
    tao = -sdrMax/10
    
    tao = torch.pow(10,tao)

    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + tao * l2norm(t)))

def sa_sdr_loss(ests, egs):
    # spks x n x S
    egs1 =torch.cat((egs[:,0,:],egs[:,1,:]),dim = -1)
    ests1 = torch.cat((ests[:,0,:],ests[:,1,:]),dim = -1)
    loss1 = tsdr(egs1,ests1)

    ests2 = torch.cat((ests[:,1,:],ests[:,0,:]),dim = -1)
    loss2 = tsdr(egs1,ests2)

    loss = torch.stack([loss1,loss2])

    N = ests.shape[0]
    max_loss, _ = torch.max(loss,dim = 0)
    max_loss = -torch.sum(max_loss) / N

    

    return max_loss

"""
                 out_channels=128,
                 in_channels=512,
                 num_blocks=16,
                 upsampling_depth=4,
                 enc_kernel_size=21,
                 enc_num_basis=512,
                 num_sources=2,
                 lr=1e-3,
                 scheduler_mode='min',
                 scheduler_factor=0.5,
                 patience=2,
                 batch_size=16,
                 num_workers=2,
"""

class Lightning(pl.LightningModule):
    def __init__(self,
                 encoder_size=21,
                 encoder_basis=512,
                 out_channels=128,
                 in_channels=512,
                 num_blocks=16,
                 upsampling_depth=4,
                 num_sources=2,
                 lr=1e-3,
                 scheduler_mode='min',
                 scheduler_factor=0.5,
                 patience=2,
                 batch_size=16,
                 num_workers=2,
                 ):
        super(Lightning, self).__init__()
        # ------------------Dataset&DataLoader Parameter-----------------
        self.batch_size = batch_size
        self.num_workers = num_workers
        # ----------training&validation&testing Param---------
        self.learning_rate = lr
        self.scheduler_mode = scheduler_mode
        self.scheduler_factor = scheduler_factor
        self.patience = patience
        # -----------------------model-----------------------
        self.SuDORMRF = SuDORMRF(out_channels,
                    in_channels,
                    num_blocks,
                    upsampling_depth,
                    encoder_size,
                    encoder_basis,
                    num_sources)
        self.save_hyperparameters() 

    def forward(self, x):
        
        return self.SuDORMRF(x)     #   输入[batch,samples]，输出[batch,chan,samples]

    # ---------------------
    # TRAINING STEP
    # ---------------------
    def compute_loss(self, y_hat, y):
        

        #   确保长度一致
        
        y_hat_new= [y_hat[:,0,:],y_hat[:,1,:]]
        y_new = [y[:,0,:],y[:,1,:]]
        val_loss = si_snr_loss(y_hat_new, y_new)

        #val_loss = sa_sdr_loss(y_hat, y)

        return val_loss

    def training_step(self, batch, batch_idx):
        mix,refs = batch
        #mix = batch['mix']
        #refs = batch['ref']
        ests = self.forward(mix)
        #ls_fn = Loss()
        metrics = self.compute_loss(ests, refs)
        '''
        for name, param in self.SuDORMRF.named_parameters():
            if param.grad is None:
                print(name)
        '''
        #loss = metrics["cisdr_loss"]    
        self.log('train_loss', metrics,on_epoch=True,on_step=False,prog_bar=False,logger=False)
        return metrics

    # ---------------------
    # VALIDATION SETUP
    # ---------------------

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        #mix = batch['mix']
        #refs = batch['ref']
        return_metrics = {}
        with torch.no_grad():
            mix,refs = batch
            ests = self.forward(mix)
            metrics = self.compute_loss(ests, refs)
            #metrics["val_loss"] = metrics["ci_loss"]

            #return_metrics["val_loss"] = metrics["val_loss"]
        self.log('val_loss', metrics,on_epoch=True,on_step=False,prog_bar=True,logger=True,sync_dist=True)
        return metrics
    
    # ---------------------
    # TRAINING SETUP
    # ---------------------

    def configure_optimizers(self):

        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=self.scheduler_mode, factor=self.scheduler_factor, patience=self.patience, verbose=True, min_lr=1e-17)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": f"val_loss",
        }

    def test_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        #mix = batch['mix']
        #refs = batch['ref']
        return_metrics = {}
        with torch.no_grad():
            mix,refs = batch
            ests = self.forward(mix)
            metrics = self.compute_loss(ests, refs)

            '''
            filepath = 'file/ref' + str(batch_idx) + '.wav'
            tmpMix = refs.squeeze(0)
            tmpMix = tmpMix.cpu().detach()
            tmpMix = tmpMix/torch.max(torch.max(tmpMix))
            

            ref1path = 'file/ests' + str(batch_idx) + '.wav'
            tmpRef = torch.stack(ests)
            tmpRef = tmpRef.cpu().detach()
            tmpRef = tmpRef/torch.max(torch.max(tmpRef))

            

            if batch_idx < 10:
                torchaudio.save(filepath, tmpMix, 8000, channels_first = True)
                torchaudio.save(ref1path, tmpRef, 8000, channels_first = True)
            #metrics["val_loss"] = metrics["ci_loss"]
            '''
            #return_metrics["val_loss"] = metrics["val_loss"]
        self.log('test_loss', metrics,on_epoch=True,on_step=False,prog_bar=True,logger=True)
        return metrics
    
    