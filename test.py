

from option import parse
from lightning import Lightning
import torch
import argparse
import os
import pytorch_lightning as pl
from DataLoaders import WSJ0DataModule
from pathlib import Path


def Train(opt,datasetPath):

    pl.seed_everything(42, workers=True)

    #   model   需要在init里定义好save_highparameters()函数，否则加载时需要用参数传入
    #light = Lightning.load_from_checkpoint('model/model_wsj1/epoch=66-step=1253436.ckpt', **opt['light_conf'])

    light = Lightning.load_from_checkpoint('model/model_wsj1_big_model/epoch=66-step=1253436.ckpt')
    light.eval()
    #   dataset
    light_conf = opt["light_conf"]
    dm = WSJ0DataModule(json_dir = datasetPath,batch_size=1, shuffle=False, num_workers=8,sample_rate=8000, segment=4.0, cv_maxlen=8.0)

   

    
    # Don't ask GPU if they are not available.
    if torch.cuda.is_available():
        gpus = len(opt['gpu_ids'])
    else:
        gpus = None
   
    # Trainer
    trainer = pl.Trainer(deterministic=False,
                         max_epochs=opt['train']['epochs'],
                         num_sanity_val_steps=2,
                         gradient_clip_val=5.)

    trainer.test(light,dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Trains a model for AuxIVA")
    parser.add_argument("dataset", type=Path, help="Location of the dataset metadata file")   
    parser.add_argument('--opt', type=str, help='Path to option YAML file.')
    args = parser.parse_args()

    opt = parse(args.opt, is_train=True)
    Train(opt,args.dataset)
