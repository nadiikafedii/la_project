import numpy as np
import time


def svd_inverse(A):
    u, sigma, vt = np.linalg.svd(A)

    sigma_inv = np.diag(1.0 / sigma)

    A_inv = vt.T @ sigma_inv @ u.T
    return A_inv

def generate_random_matrix(n):
    return np.random.randint(1, 10, (n, n))

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
