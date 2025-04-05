import numpy as np
import time
def gauss_jordan_inverse(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]

    augmented = np.hstack((A, np.identity(n)))

    for i in range(n):
        if augmented[i, i] == 0:
            raise ValueError("Matrix is singular and cannot be inverted.")

        augmented[i] = augmented[i] / augmented[i, i]

        for j in range(n):
            if i != j:
                augmented[j] -= augmented[i] * augmented[j, i]

    inverse = augmented[:, n:]
    return inverse

A = [[1, 1, 1], [0, 1, 1], [0, 0, 1]]

inv = gauss_jordan_inverse(A)
print(inv)

def generate_random_matrix(n):
    return np.random.uniform(1, 10, (n, n)).tolist()
matrix_sizes = [3, 5, 10, 100]

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
