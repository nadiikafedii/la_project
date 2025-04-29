import numpy as np
import time
np.random.seed(42)

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

def identity_matrix(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

def diagonal_inv(sigma, tol=1e-10):
    return [[1/sigma[i] if i == j and sigma[i] > tol else 0 for j in range(len(sigma))] for i in range(len(sigma))]

def norm(v):
    return sum(x**2 for x in v) ** 0.5

def dot(u, v):
    return sum(u[i]*v[i] for i in range(len(u)))

def subtract(u, v):
    return [u[i] - v[i] for i in range(len(u))]

def scalar_multiply(s, v):
    return [s * x for x in v]

def gram_schmidt(A):
    Q = []
    for a in A:
        u = a[:]
        for q in Q:
            proj = scalar_multiply(dot(u, q), q)
            u = subtract(u, proj)
        u_norm = norm(u)
        if u_norm > 1e-10:
            Q.append(scalar_multiply(1/u_norm, u))
    return Q

def eigen_decomposition_sym(A, num_iter=100):
    n = len(A)
    V = identity_matrix(n)
    for _ in range(num_iter):
        for p in range(n):
            for q in range(p+1, n):
                if abs(A[p][q]) < 1e-10:
                    continue
                theta = 0.5 * np.arctan2(2*A[p][q], A[q][q] - A[p][p])
                cos = np.cos(theta)
                sin = np.sin(theta)

                for i in range(n):
                    Api = A[i][p]
                    Aiq = A[i][q]
                    A[i][p] = cos * Api - sin * Aiq
                    A[i][q] = sin * Api + cos * Aiq

                for j in range(n):
                    Apj = A[p][j]
                    Aqj = A[q][j]
                    A[p][j] = cos * Apj - sin * Aqj
                    A[q][j] = sin * Apj + cos * Aqj

                for i in range(n):
                    Vip = V[i][p]
                    Viq = V[i][q]
                    V[i][p] = cos * Vip - sin * Viq
                    V[i][q] = sin * Vip + cos * Viq
    eigenvalues = [A[i][i] for i in range(n)]
    return eigenvalues, V

def svd_inverse(A):
    A_T = transpose(A)
    ATA = matmul(A_T, A)
    eigvals, V = eigen_decomposition_sym([row[:] for row in ATA])
    
    sigma = [eigval**0.5 if eigval > 0 else 0 for eigval in eigvals]
    sigma_inv = diagonal_inv(sigma)

    V_T = transpose(V)

    AV = matmul(A, V)
    U = []
    for i in range(len(AV[0])):
        col = [AV[j][i] for j in range(len(AV))]
        s = sigma[i]
        if s > 1e-10:
            U.append([x / s for x in col])
        else:
            U.append([0.0] * len(AV))
    U_T = transpose(U)

    temp = matmul(V, sigma_inv)
    A_inv = matmul(temp, U_T)
    return A_inv

def generate_random_matrix(n):
    return np.random.randint(1, 10, (n, n)).tolist()


matrix_sizes = [3, 5, 10]

for size in matrix_sizes:
    A = generate_random_matrix(size)
    start_time = time.time()
    A_inv = svd_inverse(A)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nSize {size}x{size}: Execution time = {execution_time:.6f} seconds")
    print(f"First 3 rows of inverse matrix {size}x{size}:")
    for row in A_inv[:3]:
        formatted_row = [round(float(val), 4) for val in row]
        print(formatted_row)
