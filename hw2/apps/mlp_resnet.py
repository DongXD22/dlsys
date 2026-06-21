import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Residual(nn.Sequential(
            nn.Linear(dim,hidden_dim),
            norm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim,dim),
            norm(dim),
        )),
        nn.ReLU()
    )
    
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Linear(dim,hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim,num_classes)
    )
    ### END YOUR SOLUTION


def epoch(dataloader:ndl.data.DataLoader, model:ndl.nn.Module, opt:ndl.optim.Optimizer|None=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    tot_loss=0.0
    num_correct=0

    if opt == None:
        model.eval()
        for batch in dataloader:
            x,y=batch
            logits:ndl.Tensor=model(x)
            loss:ndl.Tensor=nn.SoftmaxLoss()(logits,y)
            y_pred=logits.realize_cached_data().argmax(axis=1)
            y_np=y.realize_cached_data()
            num_correct+=np.ones_like(y_np)[y_np==y_pred].sum()
            tot_loss+=loss.realize_cached_data()*x.shape[0]

    else:
        model.train()
        for batch in dataloader:
            opt.reset_grad()
            x,y=batch
            logits:ndl.Tensor=model(x)
            loss:ndl.Tensor=nn.SoftmaxLoss()(logits,y)
            y_pred=logits.realize_cached_data().argmax(axis=1)
            y_np=y.realize_cached_data()
            num_correct+=np.ones_like(y_np)[y_np==y_pred].sum()
            tot_loss+=loss.realize_cached_data()*x.shape[0]

            loss.backward()
            opt.step()
    
    return 1-num_correct/len(dataloader.dataset),tot_loss/len(dataloader.dataset)

    
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    dataset_trian=ndl.data.datasets.MNISTDataset(
        rf"{data_dir}/train-images-idx3-ubyte.gz",
        rf"{data_dir}/train-labels-idx1-ubyte.gz")
    dataset_test=ndl.data.datasets.MNISTDataset(
        rf"{data_dir}/t10k-images-idx3-ubyte.gz",
        rf"{data_dir}/t10k-labels-idx1-ubyte.gz")
    dataloader_train=ndl.data.DataLoader(dataset_trian,batch_size,True)
    dataloader_test=ndl.data.DataLoader(dataset_test,batch_size,False)
    model=MLPResNet(dataset_test[0][0].shape[0],hidden_dim)
    opt=optimizer(model.parameters(),lr,weight_decay=weight_decay)
    acc_train,loss_train,acc_test,loss_test=0,0,0,0
    for _ in range(epochs):
        acc_train,loss_train=epoch(dataloader_train,model,opt)
        acc_test,loss_test=epoch(dataloader_test,model)
    return acc_train,loss_train,acc_test,loss_test

    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
