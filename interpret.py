# %%
import pandas as pd
import plotly.express as ep

# %%
# df = pd.read_csv('./runs/Inception_all/0.0003_500_128/train_progress.csv')
# df = pd.read_csv('./runs/Inception_FX/0.0003_500_128/train_progress.csv')
# df = pd.read_csv('./runs/Inception_CP/0.0003_500_128/train_progress.csv')
# df = pd.read_csv('./runs/FFT_all/0.0003_500_128/train_progress.csv')
# df = pd.read_csv('./runs/FFT_CP/0.0003_500_128/train_progress.csv')
# df = pd.read_csv('./runs/FFT_FX/0.0003_500_128/train_progress.csv')
# df = pd.read_csv('./runs/CNN_all/0.0003_500_128/train_progress.csv')
# df = pd.read_csv('./runs/CNN_CP/0.0003_500_128/train_progress.csv')
# df = pd.read_csv('./runs/CNN_FX/0.0003_500_128/train_progress.csv')

df = pd.read_csv('./modified_runs/FFT_all/3e-05_500_128/train_progress.csv')
# %%
df.head()

# %%
ep.line(df,x='Epoch',y=['Train_loss','Test_loss','Val_loss']).show()
ep.line(df,x='Epoch',y=['Train_acc','Test_acc','Val_acc']).show()
ep.line(df,x='Epoch',y=['LR']).show()


# %%
ep.line(df,x='Time',y=['Train_loss','Test_loss']).show()
ep.line(df,x='Time',y=['Train_acc','Test_acc']).show()
ep.line(df,x='Time',y=['LR']).show()




# %%
