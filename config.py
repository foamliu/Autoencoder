import torch

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

input_nbr = 3
imsize = 320
batch_size = 16
lr = 0.001
patience = 50
start_epoch = 0
epochs = 120
print_freq = 100
save_folder = 'models'
