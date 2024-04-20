import torch.utils.data
import torch
import numpy as np
import pandas as pd
import scipy.io


import os

from tqdm import tqdm

shm_dir = '/dev/shm/sschabes/'

class ECGDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        if not os.path.exists(shm_dir+ 'BSC_data.pt'):
            df = pd.read_csv('./data/output.csv',sep=';')
            df['Text'] = df['Text'].astype(str)
            data = []
            target = []

            for i in tqdm(range(len(df))):
                line = df.iloc[i]
                text = line['Text'].strip().lower()
                tar = 0 if "normal" in text  else 1
                target.append(tar)
                filename = line['Filename']
                dat = scipy.io.loadmat('./data/FX_raw/' + filename + '.mat')['x']
                if dat.shape[1]==5000: 
                    data.append(dat)

            self.data = torch.tensor(np.stack(data,axis=0))
            self.data = self.data.float()
            self.data = self.data-self.data.mean(dim=0)
            self.data = self.data/self.data.var(dim=0)
            
            self.target = torch.tensor(np.array(target)).long()
        else:
            self.data = torch.load(shm_dir+ 'BSC_data.pt')    
            self.target = torch.load(shm_dir+ 'BSC_target.pt')    

        if not os.path.exists(shm_dir):
            # Create the shm_dir
            os.makedirs(shm_dir)
        torch.save(self.data,shm_dir+ 'BSC_data.pt')
        torch.save(self.target,shm_dir+ 'BSC_target.pt')
        
        self.data = self.data.float()
        self.target = self.target

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index,...], self.target[index,...]
    