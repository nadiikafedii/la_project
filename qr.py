import time

def identity_matrix(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

def zeros_matrix(rows, cols):
    return [[0 for _ in range(cols)] for _ in range(rows)]

def transpose(M):
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]

def dot(u, v):
    return sum([u[i] * v[i] for i in range(len(u))])

def matmul(A, B):
    result = zeros_matrix(len(A), len(B[0]))
    for i in range(len(A)):
        for j in range(len(B[0])):
            result[i][j] = sum([A[i][k] * B[k][j] for k in range(len(B))])
    return result

def norm(v):
    return sum([v[i] * v[i] for i in range(len(v))]) ** 0.5

def sign(x):
    return -1 if x < 0 else 1

def householder_reflection(A):
    m = len(A)
    n = len(A[0])
    Q = identity_matrix(m)
    R = [row[:] for row in A]

    for i in range(n):
        x = [R[j][i] for j in range(i, m)]
        norm_x = norm(x)
        if norm_x == 0:
            continue
        s = sign(x[0])
        u1 = x[0] + s * norm_x
        v = [u1] + x[1:]
        v_norm = norm(v)
        v = [vi / v_norm for vi in v]

        H_i = identity_matrix(m)
        for r in range(i, m):
            for c in range(i, m):
                H_i[r][c] -= 2 * v[r - i] * v[c - i]

        R = matmul(H_i, R)
        Q = matmul(Q, H_i)

    return Q, R

def upper_tri_inverse(R, tol=1e-6): 
    n = len(R)
    R_inv = zeros_matrix(n, n)

    for i in range(n - 1, -1, -1):
        if abs(R[i][i]) < tol: 
            print(f"Warning: near-zero value on diagonal at index {i}. Using regularization.")
            R_inv[i][i] = 1 / (R[i][i] if abs(R[i][i]) >= tol else tol)
        else:
            R_inv[i][i] = 1 / R[i][i]
        
        for j in range(i + 1, n):
            total = 0
            for k in range(i + 1, j + 1):
                total += R[i][k] * R_inv[k][j]
            R_inv[i][j] = -total / R[i][i]

    return R_inv



def qr_inverse(A):
    Q, R = householder_reflection(A)
    R_inv = upper_tri_inverse(R)
    Q_T = transpose(Q)
    return matmul(R_inv, Q_T)

def generate_random_matrix(n):
    return [[(i+j) % 10 + 1 for j in range(n)] for i in range(n)]

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
