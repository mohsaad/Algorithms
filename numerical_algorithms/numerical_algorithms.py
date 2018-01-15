import numpy as np

def forward(A, b):
    x = np.zeros(b.shape)
    for j in range(0, A.shape[1]):
        if(A[j][j] == 0):
            break
        x[j] = b[j]/A[j][j]
        for i in range(j+1, A.shape[1]):
            b[i] = b[i] - A[i][j]*x[j]
    return x

def backward(A, b):
    x = np.zeros(b.shape)
    for j in range(0, A.shape[1]):
        if(A[j][j] == 0):
            break
        x[j] = b[j]/A[j][j]
        for i in range(0, j - 1):
            b[i] = b[i] - A[i][j]*x[j]
    return x
