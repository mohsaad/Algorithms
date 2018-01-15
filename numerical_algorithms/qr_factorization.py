import numpy as np

def householder_qr(A):
    m, n = A.shape
    for k in range(0, n):
        a_k = A[:, k]
        alpha_k = -1*np.sign(A[k][k]) * np.linalg.norm(a_k)
        v_k = np.zeros(a_k.shape)
        v_k[k:m] = a_k[k:m]
        e_k = np.zeros(a_k.shape)
        e_k[k] = 1
        v_k = v_k - np.multiply(alpha_k, e_k) # if you want to compute Q, take all these vectors and multiply them
        beta_k = np.dot(v_k.T, v_k)
        if(beta_k == 0):
            continue
        for j in range(k, n):
            gamma_j = np.dot(v_k.T, A[:, j])
            A[:, j] = A[:, j] - (2*gamma_j / beta_k) * v_k
    return A
