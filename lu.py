import numpy as np
import scipy.linalg as la
np.random.seed(42)
def lu_decomposition(A):
    """L and U"""
    n = len(A)
    L = [[0] * n for _ in range(n)]
    U = [[0] * n for _ in range(n)]

    for i in range(n):
        L[i][i] = 1

        for j in range(i, n):
            U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))

        for j in range(i + 1, n):
            L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]

    return L, U

def forward_substitution(L, b):
    """Lx = b"""
    n = len(L)
    x = [0] * n
    for i in range(n):
        x[i] = (b[i] - sum(L[i][j] * x[j] for j in range(i))) / L[i][i]
    return x

def backward_substitution(U, y):
    """Ux = y"""
    n = len(U)
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x

def inverse_via_lu(A):
    n = len(A)
    L, U = lu_decomposition(A)
    A_inv = []

    for i in range(n):
        e_i = [1 if j == i else 0 for j in range(n)]
        y = forward_substitution(L, e_i)
        x = backward_substitution(U, y)
        A_inv.append(x)
    return [list(row) for row in zip(*A_inv)]


def generate_random_matrix(n):
    return np.random.uniform(1, 10, (n, n)).tolist()
matrix_sizes = [3, 5, 10, 100]

for size in matrix_sizes:
    A = generate_random_matrix(size)
    A_inv = inverse_via_lu(A)
    print(f"First 3 rows of inverse matrix {size}x{size}:")
    for row in A_inv[:3]:
        print(row)