import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x+y
    ### END YOUR CODE


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


def softmax_loss(Z:np.ndarray, y:np.ndarray):
    """返回 softmax 损失。注意，对于本次作业的目的，
    您无需考虑对 log-sum-exp 计算的数值特性进行“优雅”的缩放，
    可以直接进行计算。
    Args:
        Z (np.ndarray[np.float32]): 形状为 (batch_size, num_classes) 的二维 numpy 数组，
        包含每个类别的 logit 预测。
        y (np.ndarray[np.uint8]): 形状为 (batch_size,) 的一维 numpy 数组，
        包含每个样本的真实标签。
    Returns:
        样本的平均 softmax 损失。
    """
    ## BEGIN YOUR CODE
    B,N=Z.shape
    l=np.log(np.sum(np.exp(Z),axis=1))-np.array(Z[np.arange(B),np.array(y)])
    l=l.mean()
    return l
    ### END YOUR CODE


def softmax_regression_epoch(X:np.ndarray, y:np.ndarray, theta:np.ndarray, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    N,D=X.shape
    _,C=theta.shape
    for i in range(0,N,batch):
        X_batch=X[i:i+batch]
        y_batch=y[i:i+batch]
        logits=X_batch.dot(theta)
        logits_exp=np.exp(logits)
        logits_exp_sum:np.ndarray=np.sum(logits_exp,axis=1)
        y_hat=logits_exp/logits_exp_sum.reshape((batch,-1))
        y_true=np.zeros(y_hat.shape)
        y_true[np.arange(batch),y_batch]=1
        y_err=y_hat-y_true
        theta-=X_batch.T.dot(y_err)*lr/batch
    ### END YOUR CODE


def nn_epoch(X:np.ndarray, y:np.ndarray, W1:np.ndarray, W2:np.ndarray, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    N=X.shape[0]
    def relu(x):
        return np.maximum(x,0)
    def normalize(x):
        x_sum=np.sum(x,axis=1)
        return x/x_sum.reshape((x_sum.shape[0],-1))
    ### BEGIN YOUR COD:E
    for i in range(0,N,batch):
        X_batch=X[i:i+batch]
        y_batch=y[i:i+batch]
        Z1=relu(X_batch.dot(W1))
        G2=normalize(np.exp(Z1.dot(W2)))
        Iy=np.zeros_like(G2)
        Iy[np.arange(G2.shape[0]),y_batch]=1.0
        G2-=Iy
        Z1_positive=np.zeros_like(Z1)
        Z1_positive[Z1>0]=1
        G1=Z1_positive*(G2.dot(W2.T))
        W1-=X_batch.T.dot(G1)/batch*lr
        W2-=Z1.T.dot(G2)/batch*lr
    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
