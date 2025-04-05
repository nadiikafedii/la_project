import numpy as np
import time

def cofactor_matrix(matrix):
    n = matrix.shape[0]
    cof = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
            cof[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)
    return cof

def adjugate_matrix(matrix):
    return cofactor_matrix(matrix).T

def invert_matrix_adjugate(matrix):
    matrix = np.array(matrix, dtype=float)
    determinant = np.linalg.det(matrix)
    if determinant == 0:
        raise ValueError("Zero determinant")
    adj = adjugate_matrix(matrix)
    inv = adj / determinant
    return inv

def print_matrix(matrix, precision=4):
    for row in matrix:
        print([round(float(x), precision) for x in row])

A = [
    [4, 7],
    [2, 6]
]

inv_A = invert_matrix_adjugate(A)
print_matrix(inv_A)

def generate_random_matrix(n):
    return np.random.uniform(1, 10, (n, n))

matrix_sizes = [3, 5, 10, 100]

for size in matrix_sizes:
    A = generate_random_matrix(size)
    start_time = time.time()
    A_inv = invert_matrix_adjugate(A)
    end_time = time.time()
    print(f"\nSize {size}x{size}: Time = {end_time - start_time:.6f} seconds")
    print("First 3 rows of inverse:")
    print_matrix(A_inv[:3])
