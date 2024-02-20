import torch.utils.data
import numpy as np

class ECGDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        self.data = np.float32(np.random.randn(100,12,5000))
        self.target = np.int64(np.floor(np.random.rand(100)*10))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index,...], self.target[index,...]
    