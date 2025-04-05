import time
import numpy as np

def get_minor(matrix, i, j):
    minor = [row[:j] + row[j+1:] for k, row in enumerate(matrix) if k != i]
    return minor

def determinant(matrix):
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    det = 0
    for j in range(n):
        sign = (-1) ** j
        det += sign * matrix[0][j] * determinant(get_minor(matrix, 0, j))
    return det

def cofactor_matrix(matrix):
    n = len(matrix)
    cofactors = []
    for i in range(n):
        row = []
        for j in range(n):
            minor = get_minor(matrix, i, j)
            sign = (-1) ** (i + j)
            cofactor = sign * determinant(minor)
            row.append(cofactor)
        cofactors.append(row)
    return cofactors

def transpose(matrix):
    return [list(row) for row in zip(*matrix)]

def adjugate_matrix(matrix):
    return transpose(cofactor_matrix(matrix))

def invert_matrix(matrix):
    det = determinant(matrix)
    if det == 0:
        raise ValueError("Zero determinant")
    adj = adjugate_matrix(matrix)
    inverse = [[elem / det for elem in row] for row in adj]
    return inverse

A = [
    [4, 7],
    [2, 6]
]

inv = invert_matrix(A)

for row in inv:
    print(row)


def generate_random_matrix(n):
    return np.random.uniform(1, 10, (n, n)).tolist()
matrix_sizes = [3, 5, 10, 100]

for size in matrix_sizes:
    A = generate_random_matrix(size)
    start_time = time.time()
    A_inv = invert_matrix(A)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nSize {size}x{size}: Execution time = {execution_time:.6f} seconds")
    print(f"First 3 rows of inverse matrix {size}x{size}:")
    for row in A_inv[:3]:
        print(row)
