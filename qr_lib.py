import numpy as np
import time
from scipy.linalg import qr, inv

def qr_inverse(A):
    Q, R = qr(A)
    R_inv = inv(R)
    A_inv = R_inv @ Q.T 
    return A_inv

def generate_random_matrix(n):
    return np.random.uniform(1, 10, (n, n)).tolist()
matrix_sizes = [3, 5, 10, 100]

for size in matrix_sizes:
    A = generate_random_matrix(size)
    start_time = time.time()
    A_inv = qr_inverse(A)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nSize {size}x{size}: Execution time = {execution_time:.6f} seconds")
    print(f"First 3 rows of inverse matrix {size}x{size}:")
    for row in A_inv[:3]:
        print(row)
