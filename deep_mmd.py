"""
The methods here are taken from Liu et al
https://github.com/fengliu90/DK-for-TST
"""
from argparse import Namespace
import argparse
import os
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch
import pickle
from deep_kernel_utils import MatConvert, Pdist2, MMDu, TST_MMD_adaptive_bandwidth, TST_MMD_u, TST_C2ST_D, TST_LCE_D

def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x

def Pdist2(x, y):
    """compute the paired distance between x and y."""
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    Pdist[Pdist<0]=0
    return Pdist

# Setup seeds
os.makedirs("images", exist_ok=True)
np.random.seed(819)
torch.manual_seed(819)
torch.cuda.manual_seed(819)
torch.backends.cudnn.deterministic = True
is_cuda = True

# parameters setting
# parser = argparse.ArgumentParser()
# parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
# parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
# parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate for C2STs")
# parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
# parser.add_argument("--channels", type=int, default=1, help="number of image channels")
# parser.add_argument("--n", type=int, default=100, help="number of samples in one set")
# opt = parser.parse_args()

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
cuda = True if torch.cuda.is_available() else False

def deep_mmd(sample_p, sample_q, sign_level, datset):
    if datset == 'mnist':
        opt = Namespace()
        opt.n_epochs = 2000
        opt.batch_size = 100
        opt.img_size = 32  # changed to from 32 to 28, since that is the size of mnist
        opt.orig_img_size = 28
        opt.channels = 1
        # the number of samples is determined outside.
    elif datset == 'cifar10':
        opt = Namespace()
        opt.n_epochs = 1000
        opt.batch_size = 100
        opt.img_size = 64
        opt.orig_img_size = 32
        opt.channels = 3
        # the number of samples is determined outside.
    else:
        raise NotImplementedError("MMD-Deep architectures are only implemented for mnist and cifar10")

    # Define the deep network for MMD-D
    class Featurizer(nn.Module):
        def __init__(self):
            super(Featurizer, self).__init__()

            def discriminator_block(in_filters, out_filters, bn=True):
                block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True),
                         nn.Dropout2d(0)]  # 0.25
                if bn:
                    block.append(nn.BatchNorm2d(out_filters, 0.8))
                return block

            self.model = nn.Sequential(
                *discriminator_block(opt.channels, 16, bn=False),
                *discriminator_block(16, 32),
                *discriminator_block(32, 64),
                *discriminator_block(64, 128),
            )

            # The height and width of downsampled image
            ds_size = opt.img_size // 2 ** 4
            self.adv_layer = nn.Sequential(
                nn.Linear(128 * ds_size ** 2, 100))

        def forward(self, img):
            out = self.model(img)
            out = out.view(out.shape[0], -1)
            feature = self.adv_layer(out)

            return feature

    # split and prepare datasets
    sample_p = torch.from_numpy(sample_p)
    sample_q = torch.from_numpy(sample_q)


    sample_p = torch.reshape(sample_p, (-1, opt.channels, opt.orig_img_size, opt.orig_img_size))
    sample_q = torch.reshape(sample_q, (-1, opt.channels, opt.orig_img_size, opt.orig_img_size))
    # resize the way Liu et al did (mnist to 32x32 and cifar10 to 64,64
    sample_p = TF.resize(sample_p, opt.img_size)
    sample_q = TF.resize(sample_q, opt.img_size)

    # split data 50/50
    x_train, x_test = sample_p[:len(sample_p)//2], sample_p[len(sample_p)//2:]
    y_train, y_test = sample_q[:len(sample_q) // 2], sample_q[len(sample_q) // 2:]

    train_data = x_train  # this is to be consistent with Liu et al
    dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=opt.batch_size,
        shuffle=True,
    )

    # Initialize network for deep MMD --- taken from Liu et al
    featurizer = Featurizer()
    # Initialize parameters
    epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), device, dtype))
    epsilonOPT.requires_grad = True
    sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * 32 * 32), device, dtype)
    sigmaOPT.requires_grad = True
    sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.005), device, dtype)
    sigma0OPT.requires_grad = True
    if cuda:
        featurizer.cuda()

    # Initialize optimizers
    if datset == 'mnist':
        optimizer_F = torch.optim.Adam(list(featurizer.parameters()) + [epsilonOPT] + [sigmaOPT] + [sigma0OPT], lr=0.001)
    else:
        optimizer_F = torch.optim.Adam(list(featurizer.parameters()) + [epsilonOPT] + [sigmaOPT] + [sigma0OPT],
                                       lr=0.0002)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------------------------------------------------------------------------------------------------
    #  Training deep networks for MMD-D (called featurizer)
    # ----------------------------------------------------------------------------------------------------
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    for epoch in range(opt.n_epochs):
        for i, (imgs) in enumerate(dataloader):
            if True:
                ind = np.random.choice(len(y_train), imgs.shape[0], replace=False)
                y_train_batch = y_train[ind]
                # Configure input
                real_imgs = Variable(imgs.type(Tensor))
                Fake_imgs = Variable(y_train_batch.type(Tensor))
                X = torch.cat([real_imgs, Fake_imgs], 0)

                # ------------------------------
                #  Train deep network for MMD-D
                # ------------------------------
                # Initialize optimizer
                optimizer_F.zero_grad()
                # Compute output of deep network
                modelu_output = featurizer(X)
                # Compute epsilon, sigma and sigma_0
                ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))
                sigma = sigmaOPT ** 2
                sigma0_u = sigma0OPT ** 2
                # Compute Compute J (STAT_u)
                TEMP = MMDu(modelu_output, imgs.shape[0], X.view(X.shape[0],-1), sigma, sigma0_u, ep)
                mmd_value_temp = -1 * (TEMP[0])
                mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
                STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
                # Compute gradient
                STAT_u.backward()
                # Update weights using gradient descent
                optimizer_F.step()

    # Test on Test sets
    S = torch.cat([x_test.cpu(), y_test.cpu()], 0)
    N1 = len(x_test)
    N_per = 500
    alpha = 0.05
    Sv = S.view(2 * N1, -1)
    # MMD-D
    dec, pvalue = TST_MMD_u(featurizer(S), N_per, N1, Sv, sigma, sigma0_u, ep, alpha, device, dtype)
    return dec, pvalue