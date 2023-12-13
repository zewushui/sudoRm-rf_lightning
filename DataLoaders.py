# Copyright (c) 2022 Robin Scheibler, Kohei Saijo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchaudio
from torch.utils.data import DataLoader

import librosa
import numpy as np

from Datasets import WSJ0Dataset

from wsj1_dataloader import WSJ1SpatialDataset,Lmdb_WSJ1_Dataset
import torchaudio.transforms as T

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def wsj1_single_collator(batch_list):
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


def load_mixtures_and_sources(batch):
    """
    Each info include wav path and wav duration.
    Returns:
        mixtures: a list containing B items, each item is T np.ndarray
        sources: a list containing B items, each item is T x C np.ndarray
        T varies from item to item.
    """
    mixtures, sources = [], []
    mix_infos, s1_infos, s2_infos, sample_rate, segment_len = batch
    # 逐条读取
    for mix_info, s1_info, s2_info in zip(mix_infos, s1_infos, s2_infos):
        mix_path = mix_info[0]      #   地址
        s1_path = s1_info[0]
        s2_path = s2_info[0]
        assert mix_info[1] == s1_info[1] and s1_info[1] == s2_info[1]   #长度一致
        # read wav file
        mix, _ = librosa.load(mix_path, sr=sample_rate)     #   array
        s1, _ = librosa.load(s1_path, sr=sample_rate)
        s2, _ = librosa.load(s2_path, sr=sample_rate)
        # merge s1 and s2
        s = np.dstack((s1, s2))[0]  # T x C, C = 2
        utt_len = mix.shape[-1]
        if segment_len >= 0:
            # segment
            for i in range(0, utt_len - segment_len + 1, segment_len):
                mixtures.append(mix[i:i+segment_len])   #数据切分，存入列表中，形成minibatch
                sources.append(s[i:i+segment_len])
            if utt_len % segment_len != 0:
                mixtures.append(mix[-segment_len:])
                sources.append(s[-segment_len:])
        else:  # full utterance
            mixtures.append(mix)
            sources.append(s)
    return mixtures, sources

def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)    #   获取最大长度，在结尾补零
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad

def _collate_fn(batch):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x T, torch.Tensor
        ilens : B, torch.Tentor
        sources_pad: B x C x T, torch.Tensor
    """
    # batch should be located in list

    assert len(batch) == 1
    #  返回list,内容为按4秒长度切割好的数据, 数据类型是 np.ndarray
    mixtures, sources = load_mixtures_and_sources(batch[0])

    # get batch of lengths of input sequences
    ilens = np.array([mix.shape[0] for mix in mixtures])

    # perform padding and convert to tensor     转换成tensor
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in mixtures], pad_value)
    ilens = torch.from_numpy(ilens)
    sources_pad = pad_list([torch.from_numpy(s).float()
                            for s in sources], pad_value)
    # N x T x C -> N x C x T
    sources_pad = sources_pad.permute((0, 2, 1)).contiguous()
    return mixtures_pad, sources_pad


class AudioDataLoader(torch.utils.data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
class WSJ0DataModule(pl.LightningDataModule):
    def __init__(self, json_dir = '', batch_size=4, shuffle=True,num_workers=0, sample_rate=8000, segment=4.0, cv_maxlen=8.0):
        super().__init__()
        # set regular parameters
        self.train_json_path = os.path.join(json_dir,'tr')
        self.val_json_path = os.path.join(json_dir,'cv')
        self.test_json_path = os.path.join(json_dir,'tt')
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.segment = segment
        self.cv_maxlen = cv_maxlen


    def setup(self, stage):
        torchaudio.set_audio_backend("soundfile")
        self.wsj_train = WSJ0Dataset(self.train_json_path, batch_size=self.batch_size,sample_rate=self.sample_rate, segment=self.segment)

        self.wsj_val = WSJ0Dataset(self.val_json_path, 1,sample_rate=self.sample_rate, segment=-1, cv_maxlen=self.cv_maxlen)
        self.wsj_test = WSJ0Dataset(self.test_json_path, 1,sample_rate=self.sample_rate, segment=-1, cv_maxlen=self.cv_maxlen)


    def train_dataloader(self):

        #wsj_train = [DataLoader(self.wsj_train,batch_size=self.batch_size,shuffle=self.shuffle,num_workers=self.num_workers,pin_memory=True)]
        wsj_train = AudioDataLoader(self.wsj_train, batch_size=1,shuffle=True,num_workers=self.num_workers)
        return wsj_train

    def val_dataloader(self):
        #wsj_val = [DataLoader(self.wsj_val, batch_size=1,shuffle = False, num_workers=0)]
        wsj_val = AudioDataLoader(self.wsj_val, batch_size=1,shuffle = True,num_workers=0)
        return wsj_val

    def test_dataloader(self):
        #wsj_test = [DataLoader(self.wsj_test, batch_size=1,shuffle = False, num_workers=0)]
        wsj_test = AudioDataLoader(self.wsj_test, batch_size=1,shuffle = False, num_workers=0)
        return wsj_test

class PreDictDataModule(pl.LightningDataModule):
    def __init__(self,data_dir = '', batch_size=1, shuffle=True,num_workers=0, sample_rate=8000) -> None:
        super().__init__()
        self.data_dir = data_dir    #生成一个文件，仿照wsj0的格式
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.sample_rate = sample_rate

    def setup(self) -> None:
        torchaudio.set_audio_backend("soundfile")
        self.predict_dataset = WSJ0Dataset(self.data_dir, batch_size=self.batch_size,sample_rate=self.sample_rate, segment=self.segment)


class WSJ1DataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir = '', batch_size=4, shuffle=True,num_workers=0, noiseless=True, ref_is_reverb=True):
        super().__init__()
        # set regular parameters
        self.train_path = os.path.join(dataset_dir,'si284/wsj1_si284.lmdb')
        self.val_path = os.path.join(dataset_dir,'dev93/wsj1_dev93.lmdb')
        self.test_path = os.path.join(dataset_dir,'eval92/wsj1_eval92.lmdb')

        self.train_text_path = os.path.join(dataset_dir,'si284/si284_textlist.txt')
        self.val_text_path = os.path.join(dataset_dir,'dev93/dev93_textlist.txt')
        self.test_text_path = os.path.join(dataset_dir,'eval92/eval92_textlist.txt')


        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.noiseless = noiseless
        self.ref_is_reverb = ref_is_reverb


    def setup(self, stage):
        torchaudio.set_audio_backend("soundfile")

        """
        self.wsj_train = WSJ1SpatialDataset(self.train_path, shuffle_channels=False , noiseless=self.noiseless, ref_is_reverb=self.ref_is_reverb )
        self.wsj_val = WSJ1SpatialDataset(self.val_path, shuffle_channels=False , noiseless=self.noiseless, ref_is_reverb=self.ref_is_reverb )
        self.wsj_test = WSJ1SpatialDataset(self.test_path, shuffle_channels=False , noiseless=self.noiseless, ref_is_reverb=self.ref_is_reverb)
        """
        self.wsj_train = Lmdb_WSJ1_Dataset(self.train_path, self.train_text_path, shuffle_channels=False , noiseless=self.noiseless, ref_is_reverb=self.ref_is_reverb )
        self.wsj_val = Lmdb_WSJ1_Dataset(self.val_path, self.val_text_path, shuffle_channels=False , noiseless=self.noiseless, ref_is_reverb=self.ref_is_reverb )
        self.wsj_test = Lmdb_WSJ1_Dataset(self.test_path, self.test_text_path, shuffle_channels=False , noiseless=self.noiseless, ref_is_reverb=self.ref_is_reverb)


    def train_dataloader(self):

        #wsj_train = [DataLoader(self.wsj_train,batch_size=self.batch_size,shuffle=self.shuffle,num_workers=self.num_workers,pin_memory=True)]
        wsj_train = DataLoader(self.wsj_train, batch_size=self.batch_size,
                               shuffle=self.shuffle,num_workers=self.num_workers,
                               collate_fn= wsj1_single_collator,pin_memory=True)
        return wsj_train

    def val_dataloader(self):
        #wsj_val = [DataLoader(self.wsj_val, batch_size=1,shuffle = False, num_workers=0)]
        wsj_val = DataLoader(self.wsj_val, batch_size=1,shuffle = True,num_workers=4,collate_fn=wsj1_single_collator,pin_memory=True)
        return wsj_val

    def test_dataloader(self):
        #wsj_test = [DataLoader(self.wsj_test, batch_size=1,shuffle = False, num_workers=0)]
        wsj_test = DataLoader(self.wsj_test, batch_size=1,shuffle = False, num_workers=4,collate_fn=wsj1_single_collator,pin_memory=True)
        return wsj_test
