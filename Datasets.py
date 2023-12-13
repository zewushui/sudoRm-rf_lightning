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
import librosa
import math
from typing import List
import numpy as np
import torch
import os


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
        mix, _ = librosa.load(mix_path, sr=sample_rate)
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
    max_len = max(x.size(0) for x in xs)    #   获取最大长度，在开头补零
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad

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

class WSJ0Dataset(torch.utils.data.Dataset):

    def __init__(self, json_dir, batch_size, sample_rate=8000, segment=4.0, cv_maxlen=8.0):
        """
        Args:
            json_dir: directory including mix.json, s1.json and s2.json
            segment: duration of audio segment, when set to -1, use full audio

        xxx_infos is a list and each item is a tuple (wav_file, #samples)
        """
        super(WSJ0Dataset, self).__init__()

         #   从json文件读取数据路径，mix  s1   s2   分开存储
        mix_json = os.path.join(json_dir, 'mix.json')      
        s1_json = os.path.join(json_dir, 's1.json')
        s2_json = os.path.join(json_dir, 's2.json')
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        with open(s1_json, 'r') as f:
            s1_infos = json.load(f)
        with open(s2_json, 'r') as f:
            s2_infos = json.load(f)

        # 根据sample长度进行排序，降序排列
        def sort(infos): return sorted(
            infos, key=lambda info: int(info[1]), reverse=True)
        sorted_mix_infos = sort(mix_infos)
        sorted_s1_infos = sort(s1_infos)
        sorted_s2_infos = sort(s2_infos)
        if segment >= 0.0:
            # segment length and count dropped utts
            segment_len = int(segment * sample_rate)  # 4s * 8000/s = 32000 samples
            drop_utt, drop_len = 0, 0
            for _, sample in sorted_mix_infos:
                if sample < segment_len:
                    drop_utt += 1
                    drop_len += sample
            print("Drop {} utts({:.2f} h) which is short than {} samples".format(
                drop_utt, drop_len/sample_rate/36000, segment_len))
            # generate minibach infomations
            minibatch = []
            start = 0
            while True:
                num_segments = 0
                end = start
                part_mix, part_s1, part_s2 = [], [], []
                while num_segments < batch_size and end < len(sorted_mix_infos):
                    utt_len = int(sorted_mix_infos[end][1])
                    if utt_len >= segment_len:  # skip too short utt
                        num_segments += math.ceil(utt_len / segment_len)
                        # Ensure num_segments is less than batch_size
                        if num_segments > batch_size: 
                            if start == end: # if num_segments of 1st audio > batch_size, skip it；不然会永远卡在第一个
                                end += 1
                            break
                        part_mix.append(sorted_mix_infos[end])
                        part_s1.append(sorted_s1_infos[end])
                        part_s2.append(sorted_s2_infos[end])
                    end += 1
                if len(part_mix) > 0:
                    minibatch.append([part_mix, part_s1, part_s2,
                                      sample_rate, segment_len])
                if end == len(sorted_mix_infos):
                    break
                start = end
            self.minibatch = minibatch    #   最终的数据集，是一个列表，每个元素是一个miniBatch列表，包含mix s1 s2 fs len；在dataloader中加载具体的数据
        else:  # Load full utterance but not segment
            # generate minibach infomations
                        # generate minibach infomations
            minibatch = []
            start = 0
            while True:
                end = min(len(sorted_mix_infos), start + batch_size)
                # Skip long audio to avoid out-of-memory issue
                if int(sorted_mix_infos[start][1]) > cv_maxlen * sample_rate:
                    start = end
                    continue
                minibatch.append([sorted_mix_infos[start:end],
                                  sorted_s1_infos[start:end],
                                  sorted_s2_infos[start:end],
                                  sample_rate, segment])
                if end == len(sorted_mix_infos):
                    break
                start = end
            self.minibatch = minibatch

    def get_mixinfo(self, idx):
        return self.minibatch[idx]
    
    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


class MyPredictDataset(torch.utils.data.Dataset):

    def __init__(self, json_dir, batch_size, sample_rate=8000, segment=4.0, cv_maxlen=8.0):
        """
        Args:
            json_dir: directory including mix.json, s1.json and s2.json
            segment: duration of audio segment, when set to -1, use full audio

        xxx_infos is a list and each item is a tuple (wav_file, #samples)
        """
        super(WSJ0Dataset, self).__init__()

         #   从json文件读取数据路径，mix  s1   s2   分开存储
        mix_json = os.path.join(json_dir, 'mix.json')      
        s1_json = os.path.join(json_dir, 's1.json')
        s2_json = os.path.join(json_dir, 's2.json')
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        with open(s1_json, 'r') as f:
            s1_infos = json.load(f)
        with open(s2_json, 'r') as f:
            s2_infos = json.load(f)

        # 根据sample长度进行排序，降序排列
        def sort(infos): return sorted(
            infos, key=lambda info: int(info[1]), reverse=True)
        sorted_mix_infos = sort(mix_infos)
        sorted_s1_infos = sort(s1_infos)
        sorted_s2_infos = sort(s2_infos)
        if segment >= 0.0:
            # segment length and count dropped utts
            segment_len = int(segment * sample_rate)  # 4s * 8000/s = 32000 samples
            drop_utt, drop_len = 0, 0
            for _, sample in sorted_mix_infos:
                if sample < segment_len:
                    drop_utt += 1
                    drop_len += sample
            print("Drop {} utts({:.2f} h) which is short than {} samples".format(
                drop_utt, drop_len/sample_rate/36000, segment_len))
            # generate minibach infomations
            minibatch = []
            start = 0
            while True:
                num_segments = 0
                end = start
                part_mix, part_s1, part_s2 = [], [], []
                while num_segments < batch_size and end < len(sorted_mix_infos):
                    utt_len = int(sorted_mix_infos[end][1])
                    if utt_len >= segment_len:  # skip too short utt
                        num_segments += math.ceil(utt_len / segment_len)
                        # Ensure num_segments is less than batch_size
                        if num_segments > batch_size: 
                            if start == end: # if num_segments of 1st audio > batch_size, skip it；不然会永远卡在第一个
                                end += 1
                            break
                        part_mix.append(sorted_mix_infos[end])
                        part_s1.append(sorted_s1_infos[end])
                        part_s2.append(sorted_s2_infos[end])
                    end += 1
                if len(part_mix) > 0:
                    minibatch.append([part_mix, part_s1, part_s2,
                                      sample_rate, segment_len])
                if end == len(sorted_mix_infos):
                    break
                start = end
            self.minibatch = minibatch    #   最终的数据集，是一个列表，每个元素是一个miniBatch列表，包含mix s1 s2 fs len；在dataloader中加载具体的数据
        else:  # Load full utterance but not segment
            # generate minibach infomations
                        # generate minibach infomations
            minibatch = []
            start = 0
            while True:
                end = min(len(sorted_mix_infos), start + batch_size)
                # Skip long audio to avoid out-of-memory issue
                if int(sorted_mix_infos[start][1]) > cv_maxlen * sample_rate:
                    start = end
                    continue
                minibatch.append([sorted_mix_infos[start:end],
                                  sorted_s1_infos[start:end],
                                  sorted_s2_infos[start:end],
                                  sample_rate, segment])
                if end == len(sorted_mix_infos):
                    break
                start = end
            self.minibatch = minibatch

    def get_mixinfo(self, idx):
        return self.minibatch[idx]
    
    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)
