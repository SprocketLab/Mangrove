import numpy as np
import time
import itertools
from typing import Any, DefaultDict, Dict, List, NamedTuple, Optional, Set, Tuple, Union

def multimap(A, V_array):
    """Compute a tensor product as a multilinear map.(pg. 2778, Section 2)

    Parameters
    ----------
    A
        A multidimensional tensor
    V_array
        Array of vectors to compute tensor against

    """
    p = len(V_array)
    for i in range(len(V_array)):
        if len(V_array[i].shape) == 1:
            V_array[i] = np.expand_dims(V_array[i], axis=1)

    n = V_array[0].shape[0]
    dims = [a.shape[1] for a in V_array]
    dim_ranges = [range(a.shape[1]) for a in V_array]
    B = np.zeros(dims)

    all_indices = list(itertools.product(*dim_ranges))  # i_1,...,i_p
    all_vectors = list(itertools.product(range(n), repeat=p))  # j_1,...,j_p

    for ind in all_indices:
        for vec in all_vectors:
            tmp = A[vec]
            for k in range(p):
                tmp *= V_array[k][vec[k], ind[k]]
            B[ind] += tmp
    return B


def two_tensor_prod(w, x, y):
    """
    A type of outer product
    """
    r = x.shape[0]
    M2 = np.zeros([r, r])

    for a in range(w.shape[0]):
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                M2[i, j] += w[a] * x[i, a] * y[j, a]

    return M2


def three_tensor_prod(w, x, y, z):
    """
    Three-way outer product
    """
    r = x.shape[0]
    M3 = np.zeros([r, r, r])

    if len(w.shape) == 0:
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                for k in range(z.shape[0]):
                    M3[i, j, k] += w * x[i] * y[j] * z[k]
    else:
        for a in range(w.shape[0]):
            for i in range(x.shape[0]):
                for j in range(y.shape[0]):
                    for k in range(z.shape[0]):
                        M3[i, j, k] += w[a] * x[i, a] * y[j, a] * z[k, a]

    return M3


def T_map(T, u):
    """Power method base transformation (pg. 2790, equation (5))

    Parameters
    ----------
    T
        A multidimensional tensor
    u
        A candidate eigenvector

    Returns
    -------
    t
        Transformed candidate

    """

    d = u.shape[0]
    t = np.zeros(d)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                t[i] += T[i, j, k] * u[j] * u[k]
    return t


def tensor_decomp(M2, M3, comps):
    """Tensor Decomposition Algorithm (pg. 2795, Algorithm 1)
    This is combined with reduction (4.3.1)

    Parameters
    ----------
    M2
        Symmetric matrix to aid the decomposition
    M3
        Symmetric tensor to be decomposed
    comps
        Number of eigencomponents to return

    Returns
    -------
    mu_rec
        Recovered eigenvectors (a matrix with #comps eigenvectors)
    lam_rec
        Recovered eigenvalues (a vector with #comps eigenvalues)

    """
    lam_rec = np.zeros(comps)
    mu_rec = np.zeros((M2.shape[0], comps))

    for b in range(comps):
        # initial eigendecomposition used in reduction (4.3.1)
        lam, v = np.linalg.eigh(M2)
        idx = lam.argsort()[::-1]
        lam = lam[idx]
        v = v[:, idx]

        # keep only the positive eigenvalues
        n_eigpos = np.sum(lam > 1e-2)
        if n_eigpos > 0:
            W = v[:, :n_eigpos] @ np.diag(1.0 / np.sqrt(np.abs(lam[:n_eigpos])))

            B = np.linalg.pinv(W.T)  # TODO look into this
            M3_tilde = multimap(M3, [W, W, W])  # reduction complete

            # decomposition setup
            # TODO try different hps if this doesn't work
            N = 10  # number of power iterations
            restarts = 1000  # number of random restarts # NOTE critical
            tau_star = 0  # best robust eigenvalue so far
            u_star = np.zeros(n_eigpos)  # best eigenvector so far

            # repeated restarts to find best eigenvector
            for j in range(restarts):
                # randomly draw from unit sphere (step 2)
                # u = np.random.randn(n_eigpos)
                u = np.random.multivariate_normal(np.zeros(n_eigpos), np.eye(n_eigpos))
                u /= np.linalg.norm(u)

                # power iteration for N iterations
                for i in range(N):
                    u = T_map(M3_tilde, u)
                    u /= np.linalg.norm(u)

                # check for best eigenvalue
                if j == 0 or (j > 0 and multimap(M3_tilde, [u, u, u]) > tau_star):
                    tau_star = multimap(M3_tilde, [u, u, u])
                    u_star = u

            # N more power iterations for best eigenvector found
            u = u_star
            for i in range(N):
                u = T_map(M3_tilde, u)
                u /= np.linalg.norm(u)

            # recovered modified (post-reduction) eigenvalue
            lamb = (T_map(M3_tilde, u) / u)[0]

            # recover original eigenvector and eigenvalue pair
            mu_rec[:, b] = lamb * B @ u
            lam_rec[b] = 1 / lamb**2

            # deflation: remove component, repeat
            M2 -= lam_rec[b] * np.outer(mu_rec[:, b], mu_rec[:, b])
            M3 -= three_tensor_prod(
                np.array(lam_rec[b]), mu_rec[:, b], mu_rec[:, b], mu_rec[:, b]
            )

    return mu_rec, lam_rec


def lowrank(x, k):
    u, s, vh = np.linalg.svd(x)
    s_abs = np.abs(s)
    inds = np.argsort(s_abs)[::-1][:k]
    rec = np.zeros_like(x)
    for i in inds:
        rec += s[i] * np.outer(u.T[i], vh[i])
    return rec

