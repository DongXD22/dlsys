"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np
from tqdm import tqdm
import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """ 以MNIST格式读取图像和标签文件。有关文件格式的说明，请参阅此页面：
    http://yann.lecun.com/exdb/mnist/

    参数:
        image_filename (str): MNIST格式的gzipped图像文件名
        label_filename (str): MNIST格式的gzipped标签文件名

    返回:
        元组 (X, y):
            X (numpy.ndarray[np.float32]): 2D numpy数组，包含加载的数据。
                数据的维度应为 (num_examples x input_dim)，其中 'input_dim' 
                是数据的完整维度，
                例如，由于MNIST图像为28x28，因此为784。值应为 np.float32 类型，
                且数据应归一化，使最小值为0.0，最大值为1.0
                （即，将原始值0缩放到0.0，255缩放到1.0）。

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy数组，包含示例的标签。
                值应为 np.uint8 类型，对于MNIST，其值范围为0到9。
    """
    ### BEGIN YOUR CODE
    X,y=None,None
    with gzip.open(image_filename,'rb') as f:
        magic,num_images,rows,cols=struct.unpack('>IIII',f.read(16))
        X:np.ndarray=np.frombuffer(f.read(),dtype=np.uint8)
        X=X.reshape(num_images,rows*cols)
        X=X.astype(np.float32)/255.0

    with gzip.open(label_filename,'rb') as f:
        magic,num_lable=struct.unpack('>II',f.read(8))
        y:np.ndarray=np.frombuffer(f.read(),dtype=np.uint8)
    
    return (X,y)

    ### END YOUR CODE


def softmax_loss(Z, y_one_hot)->ndl.Tensor:
    """返回 softmax 损失。请注意，出于本次作业的目的，
    你不需要担心“很好地”缩放数值属性
    log-sum-exp 计算的一部分，但可以直接计算它。

    参数：
        Z (ndl.Tensor[np.float32]): 形状的 2D 张量
            (batch_size, num_classes)，包含 logit 预测
            每堂课。
        y (ndl.Tensor[np.int8]): 形状的 2D 张量 (batch_size, num_classes)
            每个示例的真实标签索引处包含 1，并且
            其他地方为零。

    返回：
        样本上的平均 softmax 损失。 (ndl.张量[np.float32])
    """
    ## BEGIN YOUR CODE
    B,N=Z.shape
    l=ndl.log(ndl.summation(ndl.exp(Z),axes=1))-ndl.summation(Z*y_one_hot,1)
    l=ndl.summation(l)/B
    return l
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    B,N1=X.shape
    N2,K=W2.shape

    ### BEGIN YOUR COD:E
    for i in tqdm(range(0,B,batch)):
        X_batch=ndl.Tensor(X[i:i+batch])
        y_batch=ndl.Tensor(y[i:i+batch])

        Iy=np.zeros((batch,K))
        Iy[np.arange(batch),y_batch.realize_cached_data()]=1.0
        y_one_hot=ndl.Tensor(Iy)
        loss=softmax_loss(ndl.relu(X_batch.matmul(W1)).matmul(W2),y_one_hot)
        loss.backward()

        W1-=W1.grad*lr
        W2-=W2.grad*lr
        
    return (W1,W2)
    ### END YOUR CODE


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
