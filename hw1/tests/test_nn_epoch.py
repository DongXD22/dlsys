"""单独测试 nn_epoch_ndl"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apps'))

import numpy as np
import numdifftools as nd
import needle as ndl
from simple_ml import *


def gradient_check(f, *args, tol=1e-6, backward=False, **kwargs):
    eps = 1e-4
    numerical_grads = [np.zeros(a.shape) for a in args]
    for i in range(len(args)):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            f1 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] += eps
            numerical_grads[i].flat[j] = (f1 - f2) / (2 * eps)
    if not backward:
        out = f(*args, **kwargs)
        computed_grads = [
            x.numpy()
            for x in out.op.gradient_as_tuple(ndl.Tensor(np.ones(out.shape)), out)
        ]
    else:
        out = f(*args, **kwargs).sum()
        out.backward()
        computed_grads = [a.grad.numpy() for a in args]
    error = sum(
        np.linalg.norm(computed_grads[i] - numerical_grads[i]) for i in range(len(args))
    )
    assert error < tol
    return computed_grads


def test_relu_forward():
    np.testing.assert_allclose(
        ndl.relu(
            ndl.Tensor(
                [
                    [-46.9, -48.8, -45.45, -49.0],
                    [-49.75, -48.75, -45.8, -49.25],
                    [-45.65, -45.25, -49.3, -47.65],
                ]
            )
        ).numpy(),
        np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
    )
    print("ReLU forward: PASS")


def test_relu_backward():
    gradient_check(ndl.relu, ndl.Tensor(np.random.randn(5, 4)))
    print("ReLU backward: PASS")


def test_nn_epoch_grad():
    np.random.seed(0)
    X = np.random.randn(50, 5).astype(np.float32)
    y = np.random.randint(3, size=(50,)).astype(np.uint8)
    W1 = np.random.randn(5, 10).astype(np.float32) / np.sqrt(10)
    W2 = np.random.randn(10, 3).astype(np.float32) / np.sqrt(3)
    W1_0, W2_0 = W1.copy(), W2.copy()
    W1 = ndl.Tensor(W1)
    W2 = ndl.Tensor(W2)
    X_ = ndl.Tensor(X)
    y_one_hot = np.zeros((y.shape[0], 3))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    dW1 = nd.Gradient(
        lambda W1_: softmax_loss(
            ndl.relu(X_ @ ndl.Tensor(W1_).reshape((5, 10))) @ W2, y_
        ).numpy()
    )(W1.numpy())
    dW2 = nd.Gradient(
        lambda W2_: softmax_loss(
            ndl.relu(X_ @ W1) @ ndl.Tensor(W2_).reshape((10, 3)), y_
        ).numpy()
    )(W2.numpy())
    W1, W2 = nn_epoch(X, y, W1, W2, lr=1.0, batch=50)
    np.testing.assert_allclose(
        W1_0 - W1.numpy(), dW1.reshape(5, 10), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        W2_0 - W2.numpy(), dW2.reshape(10, 3), rtol=1e-4, atol=1e-4
    )
    print("nn_epoch gradient check: PASS")


def test_nn_epoch_full():
    X, y = parse_mnist(
        "data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz"
    )
    np.random.seed(0)
    W1 = ndl.Tensor(np.random.randn(X.shape[1], 100).astype(np.float32) / np.sqrt(100))
    W2 = ndl.Tensor(np.random.randn(100, 10).astype(np.float32) / np.sqrt(10))
    W1, W2 = nn_epoch(X, y, W1, W2, lr=0.2, batch=100)
    np.testing.assert_allclose(
        np.linalg.norm(W1.numpy()), 28.437788, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.linalg.norm(W2.numpy()), 10.455095, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        loss_err(ndl.relu(ndl.Tensor(X) @ W1) @ W2, y),
        (0.19770025, 0.06006667),
        rtol=1e-4,
        atol=1e-4,
    )
    print("nn_epoch full epoch: PASS")


if __name__ == "__main__":
    test_relu_forward()
    test_relu_backward()
    test_nn_epoch_grad()
    test_nn_epoch_full()
