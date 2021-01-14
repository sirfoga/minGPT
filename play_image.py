#!/usr/bin/env python
# coding: utf-8


import numpy as np
import torchvision
import torch
import matplotlib.pyplot as plt
from pathlib import Path


import logging

logging.basicConfig(
  format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
  datefmt="%Y-%d-%d %H:%M:%S",
  level=logging.INFO,
)


from mingpt.utils import set_seed

set_seed(42)  # make deterministic


import pickle

def load_pickle(f_path):
    with open(f_path, 'rb') as fp:
        return pickle.load(fp)


from sklearn.model_selection import train_test_split


def get_train_test_split(X, y, test_size, random_state=42, verbose=False):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state  # reproducible results
    )

    if verbose:
        print('train data: X ~ {}, y ~ {}'.format(X_train.shape, y_train.shape))
        print('test data: X ~ {}, y ~ {}'.format(X_test.shape, y_test.shape))


    return X_train, X_test, y_train, y_test



dataset = load_pickle(Path('~/martin/minGPT_data.pkl').expanduser())  # list of (image, mask)
X = dataset[0]  # list of images
y = dataset[1]  # list of corresponding mask

pixel_size = X.shape[1]  # should be = X.shape[2] = 32
image_channels = X.shape[-1]  # should be = 1
flattened_image_size = pixel_size * pixel_size

# convert pixels to [0, 255] range
X = np.array(np.ceil(X * 255), dtype='float32')
y = np.array(np.ceil(y * 255), dtype='float32')

X_train, X_test, y_train, y_test = get_train_test_split(X, y, 0.3, verbose=True)


from torch.utils.data import TensorDataset, DataLoader

tensor_X_train = torch.Tensor(X_train)  # tensors
tensor_y_train = torch.Tensor(y_train)
tensor_X_test = torch.Tensor(X_test)
tensor_y_test = torch.Tensor(y_test)

t_train_dataset = TensorDataset(tensor_X_train, tensor_y_train)
t_test_dataset = TensorDataset(tensor_X_test, tensor_y_test)


# can skip k-means codebook strategy, since flattened image is 32 * 32-long sequence with pixels in [0, 255] ...
# but do it anyway to trim some data nonetheless


# run kmeans

def kmeans(x, ncluster, niter=10):
    N, D = x.size()
    c = x[torch.randperm(N)[:ncluster]]  # init clusters at random
    for i in range(niter):
        # assign all pixels to the closest codebook element
        a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1)

        # move each codebook element to be the mean of the pixels that assigned to it
        c = torch.stack([x[a==k].mean(0) for k in range(ncluster)])

        # re-assign any poorly positioned codebook elements
        nanix = torch.any(torch.isnan(c), dim=1)
        ndead = nanix.sum().item()

        print('done step %d/%d, re-initialized %d dead clusters' % (i+1, niter, ndead))
        c[nanix] = x[torch.randperm(N)[:ndead]] # re-init dead clusters

    return c


# get random 5 pixels per image and stack them all up as rgb values to get half a million random pixels
n_pixels = 5
pluck_rgb = lambda x: torch.from_numpy(np.array(x)).view(flattened_image_size, image_channels)[torch.randperm(flattened_image_size)[:n_pixels], :]
px = torch.cat([pluck_rgb(x) for x, y in t_train_dataset], dim=0).float()

ncluster = 8  # 8-color = 3-bit image
with torch.no_grad():
    C = kmeans(px, ncluster, niter=8)


# encode the training examples with our codebook to visualize how much we've lost in the discretization
# these images should look normal ideally

n_samples = 32
n_cols = 8
n_rows = n_samples // n_cols
fig, axis = plt.subplots(n_rows, n_cols, figsize=(16, 8))
for ax, i in zip(axis.ravel(), np.random.randint(0, len(t_train_dataset), size=n_samples)):
    # encode and decode random data
    x, y = t_train_dataset[i]
    xpt = torch.from_numpy(np.array(x)).float().view(flattened_image_size, image_channels)
    ix = ((xpt[:, None, :] - C[None, :, :])**2).sum(-1).argmin(1)  # cluster assignments for each pixel

    sample = C[ix].view(pixel_size, pixel_size, image_channels).numpy().astype(np.uint8)
    ax.imshow(sample[..., 0], cmap='magma')
    ax.axis('off')

plt.savefig('results/clustered.png')


from torch.utils.data import Dataset

class ImageDataset(Dataset):
    """
    wrap up the pytorch CIFAR-10 dataset into our own, which will convert images into sequences of integers
    """

    def __init__(self, pt_dataset, clusters, perm=None):
        self.pt_dataset = pt_dataset
        self.clusters = clusters
        self.perm = torch.arange(flattened_image_size) if perm is None else perm

        self.vocab_size = clusters.size(0)
        self.block_size = flattened_image_size - 1

    def __len__(self):
        return len(self.pt_dataset)

    def __getitem__(self, idx):
        x, y = self.pt_dataset[idx]
        x = torch.from_numpy(np.array(x)).view(-1, image_channels)  # flatten out all pixels
        x = x[self.perm].float()  # reshuffle pixels with any fixed permutation and -> float
        a = ((x[:, None, :] - self.clusters[None, :, :])**2).sum(-1).argmin(1)  # cluster assignments
        return a[:-1], a[1:]  # always just predict the next one in the sequence


train_dataset = ImageDataset(t_train_dataset, C)
test_dataset = ImageDataset(t_test_dataset, C)


from mingpt.model import GPT, GPTConfig, GPT1Config

mconf = GPTConfig(
    train_dataset.vocab_size,
    train_dataset.block_size,
    embd_pdrop=0.0,
    resid_pdrop=0.0,
    attn_pdrop=0.0,
    n_layer=12,
    n_head=8,
    n_embd=256
)
model = GPT(mconf)


from mingpt.trainer import Trainer, TrainerConfig

tokens_per_epoch = len(train_dataset) * train_dataset.block_size
train_epochs = 50

# initialize a trainer instance and kick off training
checkpoint_path = 'results/latest_model.pt'
tconf = TrainerConfig(
    max_epochs=train_epochs,
    batch_size=4,
    learning_rate=3e-3,
    betas = (0.9, 0.95),
    weight_decay=0,
    lr_decay=True,
    warmup_tokens=tokens_per_epoch,
    final_tokens=train_epochs*tokens_per_epoch,
    ckpt_path=checkpoint_path,
    num_workers=1
)
trainer = Trainer(model, train_dataset, test_dataset, tconf)
trainer.train()


checkpoint = torch.load(checkpoint_path)  # load the state of the best model we've seen based on early stopping
model.load_state_dict(checkpoint)


# to sample we also have to technically "train" a separate model for the first token in the sequence
# we are going to do so below simply by calculating and normalizing the histogram of the first token

counts = torch.ones(ncluster)  # start counts as 1 not zero, this is called "smoothing"
rp = torch.randperm(len(train_dataset))
nest = X_train.shape[0] // 2  # how many images to use for the estimation
for i in range(nest):
    a, _ = train_dataset[int(rp[i])]
    t = a[0].item()  # index of first token in the sequence
    counts[t] += 1

prob = counts / counts.sum()  # normalize to have sum (prob) = 1


from mingpt.utils import sample

n_samples = 40
start_pixel = np.random.choice(np.arange(C.size(0)), size=(n_samples, 1), replace=True, p=prob.numpy())
start_pixel = torch.from_numpy(start_pixel).to(trainer.device)
pixels = sample(model, start_pixel, flattened_image_size - 1, temperature=1.0, sample=True, top_k=2)


# for visualization we have to invert the permutation used to produce the pixels
iperm = torch.argsort(train_dataset.perm)

n_cols = 8
n_rows = n_samples // n_cols
fig, axis = plt.subplots(n_rows, n_cols, figsize=(16, 8))
for i, ax in enumerate(axis.ravel()):
    pxi = pixels[i][iperm]  # note: undo the encoding permutation
    pxi = C[pxi].view(pixel_size, pixel_size).numpy().astype(np.uint8)  # grayscale -> 2D

    ax.imshow(pxi, cmap='magma')
    ax.axis('off')

plt.savefig('results/samples.png')


# visualize some of the learned positional embeddings, maybe they contain structure

fig, axis = plt.subplots(32, 8, figsize=(5, 5))  # 256 dimensions in embedding
for dim, ax in enumerate(axis.ravel()):
    ci = model.pos_emb.data[0, :, dim].cpu()  # 1023 tokens = pixels
    zci = torch.cat((torch.tensor([0.0]), ci))  # pre-cat a zero => 1024 tokens
    rzci = zci[iperm]  # undo the permutation to recover the pixel space of the image
    embd = rzci.view(pixel_size, pixel_size).numpy()

    ax.imshow(embd, cmap='jet')
    ax.set_title('dim #{}'.format(dim))
    ax.axis('off')


plt.savefig('results/pos_embd.png')

