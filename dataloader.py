import torch.utils.data
import torch
import numpy as np
import pandas as pd
import scipy.io

from typing import Union

import os

from tqdm import tqdm


class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, split: Union['train', 'test', 'val'], data_path='./data/', subset: Union['all', 'FX', 'CP'] = 'all', shm_dir='/dev/shm/sschabes/'):
        super().__init__()
        if not split in ['train', 'test', 'val']:
            raise ValueError('Split must be one of [train,test,val]')
        if not subset in ['all', 'FX', 'CP']:
            raise ValueError('Subset must be one of [all,FX,CP]')

        cached_data = shm_dir + 'ECG_data_' + split + subset + '.pt'
        cached_target = shm_dir + 'ECG_target_' + split + subset + '.pt'
        os.makedirs(shm_dir, exist_ok=True)

        if not os.path.exists(cached_data) or not os.path.exists(cached_target):
            # define split csv file
            split_file = data_path + os.sep + split + '.csv'
            # dataformat Birthdate;Sex;Filename;Classes;Interpretation
            df = pd.read_csv(split_file, sep=';')
            data = []
            target = df['Classes'].to_numpy()
            print(f'Loading {split}  with Interpretation {subset}')
            for i in tqdm(range(len(df))):
                line = df.iloc[i]
                if subset != 'all' and line['Interpretation'] not in subset:
                    continue
                filename = line['Filename']
                dat = scipy.io.loadmat(
                    './data/MAT/' + filename + '.mat')['ecg']
                data.append(dat)
            
            print('final shape:', np.stack(data, axis=0).shape)

            self.data = torch.tensor(np.stack(data, axis=0))
            self.data = self.data.float()
            if split == 'train':
                mean = self.data.mean(dim=[0,2],keepdim=True)
                std = self.data.std(dim=[0,2],unbiased=False,keepdim=True)
                torch.save(mean, shm_dir + 'ECG_mean_' + split + subset + '.pt')
                torch.save(std, shm_dir + 'ECG_std_' + split + subset + '.pt')
            else:
                if os.path.exists(shm_dir + 'ECG_mean_train' + subset + '.pt'):
                    mean = torch.load(shm_dir + 'ECG_mean_train' + subset + '.pt')
                    std = torch.load(shm_dir + 'ECG_std_train' + subset + '.pt')
                else:
                    raise ValueError('Mean and std not found run train split first')
                
            print(f'Split {split} {subset} mean: {mean.flatten()} std: {std.flatten()}')
            self.data = self.data-mean
            self.data = self.data/std

            self.target = torch.tensor(np.array(target)).long()

            torch.save(self.data, cached_data)
            torch.save(self.target, cached_target)
        else:
            self.data = torch.load(cached_data)
            self.target = torch.load(cached_target)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, ...], self.target[index, ...]
