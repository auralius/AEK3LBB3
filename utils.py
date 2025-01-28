import numpy as np
from numba import njit, prange


@njit(parallel=True)
def compute_distances(X, Y, L=2):
    nX = X.shape[0]
    nY = Y.shape[0]
    
    dists = np.zeros((nX, nY))
    
    for i in prange(nX):
        for j in range(nY):
            if L == 1:   # L1-norm
                dists[i, j] = np.sum(np.abs(X[i] - Y[j]))
            elif L == 2: # L2-norm
                dists[i, j] = np.sqrt(np.sum((X[i] - Y[j])**2))
    
    return dists


def svm_loss(W, X, y, reg):
    '''
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    '''

    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    N = len(y)     # number of samples
    Y_hat = X @ W  # raw scores matrix

    y_hat_true = Y_hat[range(N), y][:, np.newaxis]    # scores for true labels
    margins = np.maximum(0, Y_hat - y_hat_true + 1)   # margin for each score
    loss = margins.sum() / N - 1 + reg * np.sum(W**2) # regularized loss

    # Gradeint of the loss functoin 
    dW = (margins > 0).astype(int)    # initial gradient with respect to Y_hat
    dW[range(N), y] -= dW.sum(axis=1) # update gradient to include correct labels
    dW = X.T @ dW / N + 2 * reg * W   # gradient with respect to W

    return loss, dW