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

import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torchaudio
import random
import lmdb
import os


def compute_interval(n_target, n_mix, n_originals, n_offsets):
    """
    Compute an interval containing all the sources

    Parameters
    ----------
    n_target:
        target number samples
    n_mix:
        number of samples in mixture
    n_originals
        number of samples of the sources
    n_offsets
        offset of the sources in the mixtures
    """

    if n_target >= n_mix:
        return 0, n_mix

    n_mix = np.array(n_mix)
    n_originals = np.array(n_originals)
    n_offsets = np.array(n_offsets)

    # left is the sample at which starts the source that starts last
    # right is thee sample at which terminates the first source to terminate
    left, right = np.max(n_offsets), np.min(n_offsets + n_originals)

    # the midpoint between left and right should be the center of the
    # target interval
    midpoint = 0.5 * (left + right)

    # the start and end of interval
    start = midpoint - n_target // 2
    end = start + n_target

    # handle border effects
    if start < 0:
        return 0, n_target
    elif end >= n_mix:
        return n_mix - n_target, n_mix
    else:
        return int(start), int(end)


class WSJ1SpatialDataset(torch.utils.data.Dataset):
    """
    Dataloader for the WSJ1-spatialized datasets for multichannel

    Parameters
    ----------
    metafilename: pathlib.Path or str
        the path to the mixinfo.json file containing the dataset metadata
    max_len_s: float, optional
        the length in seconds of the samples
    ref_mic: int optional
        the microphone to use for the scaling reference, if not provided
        it is chosen at random
    shuffle_channels: bool, optional
        if set to True (default), the channels of the microphones will be
        shuffled at random
    ref_is_reverb: bool, optional
        if set to True (default), the reverberant clean signal is used
        as a reference, if False, the anechoic clean signal is used
    noiseless: bool, optional
        if set to False (default), use the noisy mixture, if true, use the
        noiseless mixture
    max_n_samples: int, optional
    """

    def __init__(
        self,
        dataset_location: Union[Path, str],
        max_len_s: Optional[float] = None,
        ref_mic: Optional[int] = 0,
        shuffle_channels: Optional[bool] = True,
        shuffle_ref: Optional[bool] = False,
        ref_is_reverb: Optional[bool] = True,
        noiseless: Optional[bool] = False,
        max_n_samples: Optional[int] = None,
    ):
        super().__init__()

        self.dataset_location = Path(dataset_location)
        self.metafilename = self.dataset_location / "mixinfo_noise.json"
        self.max_len_s = max_len_s
        self.ref_mic = ref_mic
        self.shuffle_channels = shuffle_channels
        self.shuffle_ref = shuffle_ref
        self.ref_is_reverb = ref_is_reverb
        self.noiseless = noiseless

        # open the metadata and find the dataset path
        with open(self.metafilename, "r") as f:
            # the metadata is stored as a dict, but a list is preferable
            self.metadata = list(json.load(f).values())

        # we truncate the dataset if required
        if max_n_samples is not None:
            self.metadata = self.metadata[:max_n_samples]

    def __len__(self):
        return len(self.metadata)

    def get_mixinfo(self, idx):
        return self.metadata[idx]

    #   输入一个索引值，返回一个样本；dataloader的shuffer，本质上是按照一个在[0,len-1]的随机permutation进行加载
    #   将dataloader变成迭代器，batch每次取出后，迭代器在perm中执行next   

    def __getitem__(self, idx):    
        #   加载第dx个配置项，
        room = self.metadata[idx]
        """
        if room["wav_n_samples_mixed"] < self.max_len_s:
            return self.__getitem__(idx=random.randint(0, self.__len__() - 1))
        """
        #   获取路径
        if self.noiseless:       #   选择路径：噪声/无噪声    
            mix_fn = Path(room["wav_dpath_mixed_reverberant"])
        else:
            mix_fn = Path(room["wav_mixed_noise_reverb"])
        mix_fn = Path("").joinpath(*mix_fn.parts[-3:])  # mix的相对路径

        #  参考的路径，两路镜像？
        if self.ref_is_reverb:
            ref_fns_list = room["wav_dpath_image_reverberant"]
        else:
            ref_fns_list = room["wav_dpath_image_anechoic"]
        ref_fns = [Path(p) for p in ref_fns_list]
        ref_fns = [Path("").joinpath(*fn.parts[-3:]) for fn in ref_fns]


        # 加载数据，load the mixture audio
        # torchaudio loads the data, converts to float, and normalize to [-1, 1] range
        audio_mix, fs_1 = torchaudio.load(self.dataset_location / mix_fn)

        #   简单的处理：打乱通道、参考
        # now we know the number of channels
        n_channels = audio_mix.shape[0]
        # randomly shuffle the order of the channels in the mixture if required
        if self.shuffle_channels:
            p = torch.randperm(n_channels)
        else:
            p = torch.arange(n_channels)

        if not self.shuffle_ref:
            # the reference mic needs to be picked according the shuffled order
            ref_mic = p[self.ref_mic]
        else:
            # pick any of the channels as reference
            ref_mic = torch.randint(n_channels, size=(1,))[0]


        #   读取参考信号    now load the references
        audio_ref_list = []
        for fn in ref_fns:
            audio, fs_2 = torchaudio.load(self.dataset_location / fn)
            assert fs_1 == fs_2
            audio_ref_list.append(audio[ref_mic, None, :])
        audio_ref = torch.cat(audio_ref_list, dim=0)

        audio_mix = audio_mix[p]

        if self.max_len_s is None:
            #return audio_mix, audio_ref,self.metadata[idx]['data_id']
            return audio_mix, audio_ref
        else:
            # the length of the different signals
            n_target = int(fs_1 * self.max_len_s)
            n_originals = room["wav_n_samples_original"]
            n_offsets = room["wav_offset"]
            n_mix = audio_mix.shape[-1]

            # compute an interval that has all sources in it
            s, e = compute_interval(n_target, n_mix, n_originals, n_offsets)
            mean_powers = torch.mean(audio_ref[..., s:e], dim=-1)
            for mean_power in mean_powers:
                if mean_power == 0:
                    # audio_mix, audio_ref = self.__getitem__(idx=random.randint(0,self.__len__()-1))
                    # return audio_mix, audio_ref
                    return self.__getitem__(idx=random.randint(0, self.__len__() - 1))

            #return audio_mix[..., s:e], audio_ref[..., s:e],self.metadata[idx]['data_id']
            return audio_mix[..., s:e], audio_ref[..., s:e]


class Lmdb_WSJ1_Dataset(torch.utils.data.Dataset):
    """
    Dataloader for the WSJ1-spatialized datasets for multichannel

    Parameters
    ----------
    metafilename: pathlib.Path or str
        the path to the mixinfo.json file containing the dataset metadata
    max_len_s: float, optional
        the length in seconds of the samples
    ref_mic: int optional
        the microphone to use for the scaling reference, if not provided
        it is chosen at random
    shuffle_channels: bool, optional
        if set to True (default), the channels of the microphones will be
        shuffled at random
    ref_is_reverb: bool, optional
        if set to True (default), the reverberant clean signal is used
        as a reference, if False, the anechoic clean signal is used
    noiseless: bool, optional
        if set to False (default), use the noisy mixture, if true, use the
        noiseless mixture
    max_n_samples: int, optional
    """

    def __init__(
        self,
        dataset_location: Union[Path, str],
        text_location:Union[Path, str],
        max_len_s: Optional[float] = None,
        ref_mic: Optional[int] = 0,
        shuffle_channels: Optional[bool] = True,
        shuffle_ref: Optional[bool] = False,
        ref_is_reverb: Optional[bool] = True,
        noiseless: Optional[bool] = False,
        max_n_samples: Optional[int] = None,
    ):
        super().__init__()

        self.dataset_location = dataset_location
        self.metafilename = text_location
        self.max_len_s = max_len_s
        self.ref_mic = ref_mic
        self.shuffle_channels = shuffle_channels
        self.shuffle_ref = shuffle_ref
        self.ref_is_reverb = ref_is_reverb
        self.noiseless = noiseless

        # open the metadata and find the dataset path
        with open(self.metafilename, "r") as f:
            # the metadata is stored as a dict, but a list is preferable
            self.metadata = []
            for line in f:
                self.metadata.append(line.strip())
        # we truncate the dataset if required
        if max_n_samples is not None:
            self.metadata = self.metadata[:max_n_samples]

    def __len__(self):
        return len(self.metadata)

    def get_mixinfo(self, idx):
        return self.metadata[idx]

    #   输入一个索引值，返回一个样本；dataloader的shuffer，本质上是按照一个在[0,len-1]的随机permutation进行加载
    #   将dataloader变成迭代器，batch每次取出后，迭代器在perm中执行next   

    def __getitem__(self, idx):    
        #   加载第dx个配置项，
        env = lmdb.open(self.dataset_location, readonly=True, lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:

            key = str(self.metadata[idx])
            buf = txn.get(key.encode('ascii'))
            img_flat = np.frombuffer(buf, dtype=np.int16)

            img_flat = img_flat.astype(np.float32) / 32768
            data = torch.from_numpy(img_flat)
        len = data.shape[-1] // 4
        chan = 2

        audio_mix = torch.zeros(chan,len)
        audio_ref = torch.zeros(chan,len)

        audio_mix[0,0:len] = data[0:len]
        audio_ref[0,0:len] = data[len: 2 * len]

        audio_mix[1,0:len] = data[2 * len : 3 * len]
        audio_ref[1,0:len] = data[3 * len : 4 * len]

        return audio_mix, audio_ref



def collator(batch_list_full):
    """
    Collate a bunch of multichannel signals based
    on the size of the shortest sample. The samples are cut at the center
    """
    batch_list, data_id = batch_list_full
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

    return data, target


class InterleavedLoaders:
    """
    This is a wrapper to sample alternatively from several dataloaders

    It samples until all of them are exhausted

    Parameters
    ----------
    dataloaders: list of torch.utils.data.DataLoader
        A list that contains all the dataloaders we want to sample from
    """

    def __init__(self, dataloaders: List[torch.utils.data.DataLoader]):
        self.dataloaders = dataloaders
        self._reset_dataloader_iter()

    def _reset_dataloader_iter(self):
        self.dataloaders_iter = [iter(d) for d in self.dataloaders]

    def __len__(self):
        return sum([len(dl) for dl in self.dataloaders])

    def __iter__(self):
        return self

    def __next__(self):
        for i, dl in enumerate(self.dataloaders_iter):
            try:
                # we loop until we find a non-empty loader
                batch = next(dl)
                # then we put dataloader at the end of the list
                self.dataloaders_iter = (
                    self.dataloaders_iter[i + 1 :] + self.dataloaders_iter[: i + 1]
                )
                # return the batch
                return batch
            except StopIteration:
                continue

        self._reset_dataloader_iter()

        raise StopIteration
