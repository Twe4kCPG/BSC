# %%
import time
import models.fft_model
import models.inception
import torch
import torch.nn as nn
import torch.utils.data
import sys
import os

import torchsummary

from tqdm import tqdm
import pandas as pd
import copy

import plotly.express as ep

from dataloader import ECGDataset

# %%
BATCH_SIZE = 128
# BATCH_SIZE = 512
# LR = 0.3e-3
LR = 0.3e-3
EPOCHS = 500
# EPOCHS = 2
subset = 'all'
# subset = 'FX'
# subset = 'CP'

# pat_prefix = 'garbage_tests_'
pat_prefix = 'modified_'
GPU_id = 0

model = "FFT"
# model = "Inception"
# model = "CNN"


if len(sys.argv) == 1:
    model = model
    LR = LR
    EPOCHS = EPOCHS
    BATCH_SIZE = BATCH_SIZE
    suppress_output = False
elif len(sys.argv) < 5:
    raise ValueError('Not enough arguments')
else:
    model = sys.argv[1]
    LR = float(sys.argv[2])
    EPOCHS = int(sys.argv[3])
    BATCH_SIZE = int(sys.argv[4])
    suppress_output = True


print(f'Model: {model}, Subset: {subset}, LR: {LR}, EPOCHS: {EPOCHS}, BATCH_SIZE: {BATCH_SIZE}')

base_path = pat_prefix + 'runs' + os.sep+model + '_' + \
    subset + os.sep+f'{LR}_{EPOCHS}_{BATCH_SIZE}'
os.makedirs(base_path)

device = f'cuda:{GPU_id}' if torch.cuda.is_available() else 'cpu'

# %%
trainset = ECGDataset('train',subset=subset,shm_dir='/dev/shm/sschabes2/')
testset = ECGDataset('test',subset=subset,shm_dir='/dev/shm/sschabes2/')
valset = ECGDataset('val',subset=subset,shm_dir='/dev/shm/sschabes2/')

trainloader = torch.utils.data.DataLoader(
    trainset, BATCH_SIZE, True, pin_memory=True)
testloader = torch.utils.data.DataLoader(
    testset, BATCH_SIZE, True, pin_memory=True)
valloader = torch.utils.data.DataLoader(
    valset, BATCH_SIZE, True, pin_memory=True)

# %%


class Network(nn.Module):
    def __init__(self, number_of_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=128,
                      kernel_size=80, stride=64),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256,
                      kernel_size=11, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=128,
                      kernel_size=7, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=number_of_classes),
        )

    def forward(self, x):
        return self.layers(x)


# %%
# model = Network(2)
if model == "Inception":
    model = models.inception.InceptionTime(2)
elif model == "FFT":
    model = models.fft_model.FFTModel(2)
elif model == "CNN":
    model = Network(2)
else:
    raise ValueError('Model not found')


model = model.to(device)


criterum = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), LR, weight_decay=1e-5)
# optimizer = torch.optim.SGD(model.parameters(),LR*10,momentum=0.9,weight_decay=1e-5)

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer=optimizer, max_lr=LR, total_steps=EPOCHS)

# # %%
# torchsummary.summary(model, (12, 5000), device=device)

# %%


def train(model, dataloader):
    running_acc = 0
    running_loss = 0
    model.train()
    with tqdm(enumerate(dataloader), total=len(dataloader), disable=suppress_output) as t:
        t.set_description_str('Train')
        for i, (data, labels) in t:
            # print('zero')
            # reset
            optimizer.zero_grad(True)

            # Data transfere
            # print('pre transfer')
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # print('post transfer')

            # print('model')
            # network inference
            target = model(data)

            # print('loss')
            # loss
            loss = criterum(target, labels)
            running_loss += loss.item()

            # print('backprob')
            # backprobagation
            loss.backward()
            optimizer.step()

            # accuracy calculation
            running_acc += (torch.argmax(target, -1) == labels).sum().item()

            t.set_postfix({'loss': running_loss/(BATCH_SIZE*(i+1)),
                          'acc': running_acc/(BATCH_SIZE*(i+1))})
            # print('done')
    return {
        'loss': running_loss/(len(dataloader)*BATCH_SIZE),
        'acc': running_acc/(len(dataloader)*BATCH_SIZE),
    }


def test(model, dataloader):
    running_acc = 0
    running_loss = 0
    model.eval()
    with tqdm(enumerate(dataloader), total=len(dataloader), disable=suppress_output) as t:
        t.set_description_str('Test ')
        for i, (data, labels) in t:
            # Data transfere
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # network inference
            target = model(data)

            # loss
            loss = criterum(target, labels)
            running_loss += loss.item()

            # accuracy calculation
            running_acc += (torch.argmax(target, -1) == labels).sum().item()

            t.set_postfix({'loss': running_loss/(BATCH_SIZE*(i+1)),
                          'acc': running_acc/(BATCH_SIZE*(i+1))})
    return {
        'loss': running_loss/(len(dataloader)*BATCH_SIZE),
        'acc': running_acc/(len(dataloader)*BATCH_SIZE),
    }


# %%

stats = []

best_test = 0
best_model = None
start = time.time()
for i in range(EPOCHS):
    train_dict = train(model, trainloader)
    test_dict = test(model, testloader)
    val_dict = test(model, valloader)

    train_dict = dict([('Train_'+key, value)
                      for key, value in train_dict.items()])
    test_dict = dict([('Test_'+key, value)
                     for key, value in test_dict.items()])
    val_dict = dict([('Val_'+key, value)
                        for key, value in val_dict.items()])

    stats.append(train_dict | test_dict | val_dict | {
                 'Epoch': i, 'LR': lr_scheduler.get_lr()[0], 'Time': time.time()-start})
    if test_dict['Test_acc'] > best_test:
        best_test = test_dict['Test_acc']
        best_model = copy.deepcopy(model)

    lr_scheduler.step()
    df = pd.DataFrame(stats)
    df.to_csv(base_path + os.sep + 'train_progress.csv')

torch.save(best_model.state_dict(), base_path +
           os.sep + 'best_model_state_dict.pt')
torch.save(best_model, base_path + os.sep + 'best_model.pt')


# %%

# best_model_load = Network(2)
# best_model_load.load_state_dict(torch.load('best_model_state_dict.pt'))


# %%
# fig = ep.line(stats,x='Epoch',y=['Train_loss','Test_loss'])
# fig.show()
# fig.write_html('./tmp.html')
# fig=ep.line(stats,x='Epoch',y=['Train_acc','Test_acc'])
# fig.show()
# fig=ep.line(stats,x='Epoch',y=['LR'])
# fig.show()

# %% [markdown]
# import umap
# from sklearn.preprocessing import StandardScaler

# %% [markdown]
# scaled_ecg = StandardScaler().fit_transform(trainset.data.reshape(trainset.data.shape[0],-1))
# reducer = umap.UMAP()
# embedding = reducer.fit_transform(scaled_ecg)
# # print(emb)

# %% [markdown]
# import plotly.graph_objects as go
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=embedding[:, 0],y=embedding[:, 1],marker_color=trainset.target,mode='markers', marker=dict(opacity=0.3)))
# fig.show()
