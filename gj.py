import time
import numpy as np
np.random.seed(42)

def gauss_jordan_inverse(matrix):
    n = len(matrix)
    augmented = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(matrix)]

    for i in range(n):
        pivot = augmented[i][i]
        if pivot == 0:
            raise ValueError("Matrix is singular and cannot be inverted.")

        for j in range(2 * n):
            augmented[i][j] /= pivot

        for k in range(n):
            if k != i:
                factor = augmented[k][i]
                for j in range(2 * n):
                    augmented[k][j] -= factor * augmented[i][j]

    inverse = [row[n:] for row in augmented]
    return inverse

A = [[1, 1, 1], [0, 1, 1], [0, 0, 1]]

inv_A = gauss_jordan_inverse(A)

for row in inv_A:
    print(row)

def generate_random_matrix(n):
    return np.random.uniform(1, 10, (n, n)).tolist()
matrix_sizes = [3, 5, 10, 100, 1000]

for size in matrix_sizes:
    A = generate_random_matrix(size)
    start_time = time.time()
    A_inv = gauss_jordan_inverse(A)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nSize {size}x{size}: Execution time = {execution_time:.6f} seconds")
    print(f"First 3 rows of inverse matrix {size}x{size}:")
    for row in A_inv[:3]:
        print(row)
