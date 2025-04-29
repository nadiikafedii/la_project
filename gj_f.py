import time
import numpy as np
from memory_profiler import memory_usage

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

def generate_random_matrix(n):
    return np.random.uniform(1, 10, (n, n)).tolist()

def compute_error(A, A_inv):
    A_np = np.array(A)
    A_inv_np = np.array(A_inv)
    identity = np.eye(len(A_np))
    approx_I = A_np @ A_inv_np
    error = np.linalg.norm(approx_I - identity)
    return error

def wrapper(func, *args):
    return func(*args)

if __name__ == '__main__':
    matrix_sizes = [3, 5, 10, 100, 1000]

    for size in matrix_sizes:
        A = generate_random_matrix(size)
        try:
            cond_number = np.linalg.cond(A)

            start_time = time.time()
            mem_usage, A_inv = memory_usage(
                (wrapper, (gauss_jordan_inverse, A)),
                retval=True, max_iterations=1
            )
            end_time = time.time()

            execution_time = end_time - start_time
            peak_memory = max(mem_usage) - min(mem_usage)
            error = compute_error(A, A_inv)

            print(f"\nSize {size}x{size}")
            print(f"Condition number: {cond_number:.2e}")
            print(f"Execution time: {execution_time:.6f} seconds")
            print(f"Peak memory usage: {peak_memory:.6f} MiB")
            print(f"Error ||AA⁻¹ - I||: {error:.2e}")
            print(f"Inverse matrix {size}x{size}:")
            for row in A_inv:
                print(row)
        except Exception as e:
            print(f"\nSize {size}x{size}: FAILED — {e}")
