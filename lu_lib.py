import numpy as np
from scipy.linalg import lu_factor, lu_solve
import time
np.random.seed(42)

def inverse_via_lu_scipy(A):
    n = A.shape[0]
    lu, piv = lu_factor(A)
    A_inv = np.zeros((n, n))

    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1
        A_inv[:, i] = lu_solve((lu, piv), e_i)
    return A_inv

def generate_random_matrix(n):
    return [[np.random.uniform(1, 10) for _ in range(n)] for _ in range(n)]
matrix_sizes = [3, 5, 10, 100]

for size in matrix_sizes:
    A_list = generate_random_matrix(size)
    A_np = np.array(A_list)

    start_time = time.time()
    A_inv = inverse_via_lu_scipy(A_np)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nSize {size}x{size}: Execution time = {execution_time:.6f} seconds")
    print(f"First 3 rows of an inverse matrix {size}x{size}:")
    for row in A_inv[:3]:
        print(row)
