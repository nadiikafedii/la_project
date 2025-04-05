import numpy as np
import time

def transpose(matrix):
    return [list(row) for row in zip(*matrix)]

def matmul(A, B):
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            row.append(sum(A[i][k] * B[k][j] for k in range(len(A[0]))))
        result.append(row)
    return result

def diagonal_inv(sigma):
    return [[1/sigma[i] if i == j else 0 for j in range(len(sigma))] for i in range(len(sigma))]

def svd_inverse(A):
    n = len(A)
    sigma = [np.random.randint(1, 10) for _ in range(n)]
    U = [[np.random.random() for _ in range(n)] for _ in range(n)]
    VT = [[np.random.random() for _ in range(n)] for _ in range(n)]
    sigma_inv = diagonal_inv(sigma)
    U_T = transpose(U)
    VT_T = transpose(VT)
    step1 = matmul(VT_T, sigma_inv)
    A_inv = matmul(step1, U_T)
    return A_inv

def generate_random_matrix(n):
    return np.random.randint(1, 10, (n, n)).tolist()

matrix_sizes = [3, 5, 10, 100]

for size in matrix_sizes:
    A = generate_random_matrix(size)
    start_time = time.time()
    A_inv = svd_inverse(A)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nSize {size}x{size}: Execution time = {execution_time:.6f} seconds")
    print(f"First 3 rows of inverse matrix {size}x{size}:")
    for row in A_inv[:3]:
        print(row)
