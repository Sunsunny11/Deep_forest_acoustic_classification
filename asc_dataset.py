from soundata.datasets import tau2019uas, urbansound8k
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import soundata
import os
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
import librosa
from tqdm import tqdm

tags_to_ids = {
    'airport': 0,
    'bus': 1,
    'shopping_mall': 2,
    'street_pedestrian': 3,
    'street_traffic': 4,
    'metro_station': 5,
    'metro': 6,
    'public_square': 7,
    'tram': 8,
    'park': 9
}


def load_audio(path, sr):
    y, _ = librosa.load(path, sr=sr)
    return y


class ASC_Dataset(Dataset):
    def __init__(self, split='train'):
        self.tags_to_ids = tags_to_ids
        self.dataset = tau2019uas.Dataset(
            data_home='/vol/vssp/datasets/audio/dcase2019/task1/dataset_root')

        clip_ids = self.dataset.clip_ids
        dev_clip_ids = [id for id in clip_ids if 'development' in id]
        all_clips = self.dataset.load_clips()

        self.train_clip_ids = [id for id in dev_clip_ids if all_clips[id].split == 'development.train']
        self.test_clip_ids = [id for id in dev_clip_ids if all_clips[id].split == 'development.test']

        if split == 'train':
            self.audio_ids = self.train_clip_ids
        elif split == 'test':
            self.audio_ids = self.test_clip_ids

    def __len__(self):
        return len(self.audio_ids)

    def __getitem__(self, index):
        audio_id = self.audio_ids[index]
        clip = self.dataset.clip(audio_id)

        waveform, sr = clip.audio
        waveform = np.array((waveform[0] + waveform[1]) / 2)
        max_length = sr * 10

        if len(waveform) > max_length:
            waveform = waveform[0:max_length]
        else:
            waveform = np.pad(waveform, (0, max_length - len(waveform)), 'constant')

        tag = clip.tags.labels[0]
        target = self.tags_to_ids[tag]
        target = np.eye(10)[target]

        data_dict = {
            'audio_name': audio_id, 'waveform': waveform, 'target': target, 'tag': tag}

        return data_dict


def asc_collate_fn(batch):
    audio_name = [data['audio_name'] for data in batch]
    waveform = [data['waveform'] for data in batch]
    target = [data['target'] for data in batch]

    waveform = torch.FloatTensor(waveform)
    target = torch.FloatTensor(target)

    return {'audio_name': audio_name, 'waveform': waveform, 'target': target}


def get_asc_dataloader(split,
                       batch_size,
                       shuffle=False,
                       drop_last=False,
                       num_workers=8):
    dataset = ASC_Dataset(split=split)

    return DataLoader(dataset=dataset, batch_size=batch_size,
                      shuffle=shuffle, drop_last=drop_last,
                      num_workers=num_workers, collate_fn=asc_collate_fn)


class ESC50_Dataset(Dataset):
    def __init__(self, sr=41000, data_type='train', test_fold_num=4, in_col='filename', out_col='category'):
        dataset_path = '/vol/research/NOBACKUP/CVSSP/scratch_4weeks/xl01061/ESC-50-master'
        meta = pd.read_csv(os.path.join(dataset_path, 'meta/esc50.csv'))

        self.data_type = data_type
        if data_type == 'train':
            self.data_list = meta[meta['fold'] != test_fold_num]
        elif data_type == 'test':
            self.data_list = meta[meta['fold'] == test_fold_num]

        print(f'ESC-50 {self.data_type} set using fold {test_fold_num} is creating, using sample rate {sr} Hz ...')

        self.data = []
        self.labels = []
        self.audio_name = []
        self.c2i = {}
        self.i2c = {}
        self.categories = sorted(self.data_list[out_col].unique())

        for i, category in enumerate(self.categories):
            self.c2i[category] = i
            self.i2c[i] = category
        # for ind in tqdm(range(len(self.data_list))):
        # for ind in tqdm(range(5)):
        for ind in tqdm(range(len(self.data_list))):
            row = self.data_list.iloc[ind]
            file_path = os.path.join(dataset_path, 'audio', row[in_col])
            self.audio_name.append(row[in_col])
            self.data.append(load_audio(file_path, sr=sr))
            self.labels.append(self.c2i[row['category']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_name = self.audio_name[idx]
        waveform = self.data[idx]
        target = np.eye(50)[self.labels[idx]]
        data_dict = {'audio_name': audio_name, 'waveform': waveform, 'target': target}

        return data_dict


def esc_collate_fn(batch):
    audio_name = [data['audio_name'] for data in batch]
    waveform = [data['waveform'] for data in batch]
    target = [data['target'] for data in batch]

    waveform = torch.FloatTensor(waveform)
    target = torch.FloatTensor(target)

    return {'audio_name': audio_name, 'waveform': waveform, 'target': target}


def get_esc_dataloader(data_type,
                       test_fold_num,
                       batch_size,
                       shuffle=False,
                       drop_last=False,
                       num_workers=8):
    dataset = ESC50_Dataset(data_type=data_type, test_fold_num=test_fold_num, )

    return DataLoader(dataset=dataset, batch_size=batch_size,
                      shuffle=shuffle, drop_last=drop_last,
                      num_workers=num_workers, collate_fn=esc_collate_fn)


if __name__ == '__main__':
    train_loader = get_asc_dataloader(split='train', batch_size=130, shuffle=True)
    for item in train_loader:
        print(item)
        pass

