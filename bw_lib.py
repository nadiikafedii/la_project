import numpy as np
import time
from memory_profiler import memory_usage

np.random.seed(42)

def generate_random_matrix(n):
    return np.random.randint(1, 10, (n, n)).astype(float)

def blockwise_inverse(A, min_block_size=2):
    n = A.shape[0]
    
    if n <= min_block_size:
        return np.linalg.inv(A)

    mid = n // 2
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]

    A11_inv = blockwise_inverse(A11, min_block_size)

    S = A22 - A21 @ A11_inv @ A12
    S_inv = blockwise_inverse(S, min_block_size)

    top_left = A11_inv + A11_inv @ A12 @ S_inv @ A21 @ A11_inv
    top_right = -A11_inv @ A12 @ S_inv
    bottom_left = -S_inv @ A21 @ A11_inv
    bottom_right = S_inv

    top = np.hstack((top_left, top_right))
    bottom = np.hstack((bottom_left, bottom_right))

    return np.vstack((top, bottom))

def compute_error(A, A_inv):
    identity = np.eye(len(A))
    return np.linalg.norm(A @ A_inv - identity)

def wrapper(func, *args):
    return func(*args)

if __name__ == '__main__':
    matrix_sizes = [3, 5, 10, 100, 1000]
    
    for size in matrix_sizes:
        print(f"\nSize {size}x{size}")
        
        A = generate_random_matrix(size)
        cond_number = np.linalg.cond(A)
        
        start_time = time.time()
        mem_usage, A_inv = memory_usage((wrapper, (blockwise_inverse, A)), retval=True, max_iterations=1)
        end_time = time.time()
        
        execution_time = end_time - start_time
        peak_memory = max(mem_usage) - min(mem_usage)
        error = compute_error(A, A_inv)
        
        print(f"Condition number: {cond_number:.2e}")
        print(f"Execution time: {execution_time:.6f} seconds")
        print(f"Peak memory usage: {peak_memory:.6f} MiB")
        print(f"Error ||AA⁻¹ - I||: {error:.2e}")
        print(f"Inverse matrix {size}x{size}:")
        with np.printoptions(precision=4, suppress=True):
            print(A_inv)
