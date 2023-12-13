

from option import parse
from lightning import Lightning
import torch
import argparse
import os
import pytorch_lightning as pl
from DataLoaders import WSJ0DataModule
from pathlib import Path
from wsj1_dataloader import WSJ1SpatialDataset
from torch.utils.data import DataLoader
import torchaudio
import numpy as np
import librosa
import torchaudio.transforms as T


def data_to(data, device):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    return data.to(device)


def collator(batch_list):
    """
    Collate a bunch of multichannel signals based
    on the size of the shortest sample. The samples are cut at the center
    """
    max_len = max([s[0].shape[-1] for s in batch_list])
    mix_n_channels = batch_list[0][0].shape[0]
    ref_n_channels = batch_list[0][1].shape[0]

    mix_batch_size = (len(batch_list), mix_n_channels, max_len)
    ref_batch_size = (len(batch_list), ref_n_channels, max_len)

    data = batch_list[0][0].new_zeros(mix_batch_size)
    target = batch_list[0][1].new_zeros(ref_batch_size)

    offsets = [(max_len - s[0].shape[-1]) // 2 for s in batch_list]

    for b, ((d, t), o) in enumerate(zip(batch_list, offsets)):
        data[b, :, o : o + d.shape[-1]] = d
        target[b, :, o : o + t.shape[-1]] = t

    singleData = data[:,0,:]
    
    resampler = T.Resample(16000, 8000, dtype=singleData.dtype)
    singleData = resampler(singleData)
    target = resampler(target)
    
    return singleData, target



def Train(opt,datasetPath):

    pl.seed_everything(42, workers=True)

    #   model   需要在init里定义好save_highparameters()函数，否则加载时需要用参数传入
    light = Lightning.load_from_checkpoint('model/checkpoint/last.ckpt')
    light.eval()
    #   dataset
    light_conf = opt["light_conf"]
    #dm = WSJ0DataModule(json_dir = datasetPath,batch_size=1, shuffle=False, num_workers=8,sample_rate=8000, segment=4.0, cv_maxlen=8.0)

    tag_dataset = "eval92"
    dataset_dri = "/media/zewushui/diskD/00_study/00_nn/00_dataset/dataset/tiss/output/wsj1_2345_db/wsj1_2_mix_m2/eval92"
    dataset = WSJ1SpatialDataset(dataset_dri, shuffle_channels=False, noiseless=True, ref_is_reverb=True)
    torchaudio.set_audio_backend("soundfile")
    device = torch.device("cpu")
    for idx, (mix, ref) in enumerate(dataset):
        mix = data_to(mix, device)
        ref = data_to(ref, device)
        mixinfo = dataset.get_mixinfo(idx)
        data_id = mixinfo["data_id"]
        mix_id = f"{tag_dataset}_{data_id}"
        fs = mixinfo["wav_frame_rate_mixed"]



    dm = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=4,collate_fn=collator,pin_memory=True)
    
    # Don't ask GPU if they are not available.
    if torch.cuda.is_available():
        gpus = len(opt['gpu_ids'])
    else:
        gpus = None
   
    # Trainer
    trainer = pl.Trainer(deterministic=False,
                         max_epochs=opt['train']['epochs'],
                         num_sanity_val_steps=2,
                         gradient_clip_val=5.,
                         devices=[0])

    trainer.test(light,dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Trains a model for AuxIVA")
    parser.add_argument("dataset", type=Path, help="Location of the dataset metadata file")   
    parser.add_argument('--opt', type=str, help='Path to option YAML file.')
    args = parser.parse_args()

    opt = parse(args.opt, is_train=True)
    Train(opt,args.dataset)
