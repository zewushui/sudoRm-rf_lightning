

from option import parse
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning import Lightning
import torch
import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from DataLoaders import WSJ1DataModule
from pathlib import Path

torch.set_float32_matmul_precision('medium')
#torch.backends.cudnn.benchmark = True
import torch.profiler

#dataset_dir = "/media/zewushui/diskD/00_study/00_nn/00_dataset/dataset/tiss/output/wsj1_2345_db/wsj1_2_mix_m2"
dataset_dir = "/media/zewushui/diskD/00_study/00_nn/00_dataset/lmdb/wsj1"

def Train(opt,datasetPath):

    #   model
    light = Lightning(**opt['light_conf'])      #   *把接受的参数合并成一个元组；** 接受的多个参数合并成一个字典
    #light = Lightning.load_from_checkpoint('model/checkpoint/epoch=34-step=81865.ckpt')

    #light = torch.compile(light, mode = 'reduce-overhead')

    light_conf = opt["light_conf"]
    #dm = WSJ0DataModule(json_dir = datasetPath,batch_size=light_conf["batch_size"], shuffle=True, num_workers=8,sample_rate=8000, segment=4.0, cv_maxlen=8.0)

    dm = WSJ1DataModule(dataset_dir, batch_size=light_conf["batch_size"], shuffle=True, num_workers=8, noiseless=True, ref_is_reverb=True)

    # mkdir the file of Experiment path
    os.makedirs(os.path.join(opt['resume']['path'],
                             opt['resume']['checkpoint']), exist_ok=True)
    checkpoint_path = os.path.join(
        opt['resume']['path'], opt['resume']['checkpoint'])
    
    # checkpoint    监视某个指标，每次指标达到最好的时候，它就缓存当前模型
    """ModelCheckpoint当作最后一个CallBack,即总是在最后执行。若在训练过程中访问best_model_score,对应的是上一次模型缓存的结果"""
    checkpoint = ModelCheckpoint(
        checkpoint_path, monitor='val_loss', mode='min', save_top_k=1, verbose=1, save_last=True)
    
    # Don't ask GPU if they are not available.
    if torch.cuda.is_available():
        gpus = len(opt['gpu_ids'])
    else:
        gpus = None
   
    # Early Stopping
    early_stopping = False
    if opt['train']['early_stop']:
        early_stopping = EarlyStopping(monitor='val_loss', patience=opt['train']['patience'],
                                       mode='min', verbose=1)
    # default logger used by trainer
    logger = TensorBoardLogger(save_dir='./logger',version=1,name='lightning_logs')

    # Trainer
    trainer = pl.Trainer(deterministic=False,
                         max_epochs=opt['train']['epochs'],
                         callbacks=[checkpoint,early_stopping],
                         default_root_dir=checkpoint_path,
                         devices=gpus,
                         #check_val_every_n_epoch=0,
                         accumulate_grad_batches = 1,
                         num_sanity_val_steps=2,
                         gradient_clip_val=5.,
                         #logger=logger,
                         profiler="simple",
                         #precision = 'bf16-mixed'
                         )

    trainer.fit(light,dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Trains a model for AuxIVA")
    parser.add_argument("dataset", type=Path, help="Location of the dataset metadata file")   
    parser.add_argument('--opt', type=str, help='Path to option YAML file.')
    args = parser.parse_args()

    opt = parse(args.opt, is_train=True)
    Train(opt,args.dataset)
