import torch.utils.data
import torch
import numpy as np
import pandas as pd
import scipy.io

import os

from tqdm import tqdm

class ECGDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        if not os.path.exists('/dev/shm/BSC_data.pt'):
            df = pd.read_csv('./data/filtered_output_csv.csv',sep=';')
            df['Text'] = df['Text'].astype(str)
            data = []
            target = []

            for i in tqdm(range(len(df))):
                line = df.iloc[i]
                tar = 0 if len(line['Text'].strip())==0 else 1
                target.append(tar)
                filename = line['Filename']
                dat = scipy.io.loadmat('./data/output/' + filename + '_output.mat')['ecg']
                if dat.shape[1]==5000: 
                    data.append(dat)

            self.data = torch.tensor(np.stack(data,axis=0))
            self.data = self.data.float()
            self.data = self.data-self.data.mean(dim=0)
            self.data = self.data/self.data.var(dim=0)
            
            self.target = torch.tensor(np.array(target)).long()
        else:
            self.data = torch.load('/dev/shm/BSC_data.pt')    
            self.target = torch.load('/dev/shm/BSC_target.pt')    


        torch.save(self.data,'/dev/shm/BSC_data.pt')
        torch.save(self.target,'/dev/shm/BSC_target.pt')


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index,...], self.target[index,...]
    