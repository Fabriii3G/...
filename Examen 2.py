import numpy as np
import matplotlib.pyplot as plt
from math import factorial

#Pregunta 3
#Función para crear matriz A
def matrizA(matriz):
    for i in range(45):       # i = 0,...,44
        for j in range(30):   # j = 0,...,29
            matriz[i][j] = (i+1)**2 + (j+1)**2  # sumamos 1 porque el enunciado empieza en 1
    return matriz
matrizRand = np.ones((45, 30))
A = matrizA(matrizRand)
p = 1
tol = 10**-5

#Función del metodo de W. Li y Z. Li
def pseudoInv(A, p, tol, max_iter):
    # Inicialización tipo Newton–Schulz
    normA2 = np.linalg.norm(A, 2) ** 2
    X0 = A.T / normA2

    for k in range(max_iter):
        # Calcular M = A X^(k)
        M = A @ X0 # @ Se usa para multiplicar matrices

        # Construir el sumatorio
        S = np.zeros_like(M) #Matriz de 0s de las dimensiones de M
        for q in range(1, p + 1):
            coef = ((-1) ** (q - 1)) * factorial(p) / (factorial(q) * factorial(p - q))
            S += coef * np.linalg.matrix_power(M, q - 1)

        # Actualización
        X = X0 @ S

        # Criterio de parada
        err = np.linalg.norm(A @ X @ A - A, 'fro')
        if err < tol:
            return X, k + 1

        X0 = X

    return X0, max_iter


p_values = [1,2,3,4,5,6,7,8,10]
iters_needed = []
print("PseudoInv")
for p in p_values:
    _, iters = pseudoInv(A, p, 1e-5, 1000)
    iters_needed.append(iters)
    print(f"p={p} -> {iters} iteraciones")



#Pregunta 4
#Funcion sustitucion hacia atras (matriz triangular superior)
def sust_atras(A, b):
    m,n = A.shape
    x = np.zeros_like(b)

    for i in range(m-1, -1, -1):
        suma = np.dot(A[i, i+1:], x[i+1:])
        x[i] = (b[i] - suma)/ A[i, i]
    return x

# Sustitución hacia adelante (matriz triangular inferior)
def sust_adelante(A, b):
    m, n = A.shape
    x = np.zeros_like(b)

    for i in range(m):  # de la primera fila a la última
        suma = np.dot(A[i, :i], x[:i])
        x[i] = (b[i] - suma) / A[i, i]

    return x

print("SustAtras")
# Ejemplo de uso:
U = np.array([[1, 1, -1, 3],
              [0,  -1, -1, -5],
              [0,  0, 3, 13],
              [0,  0, 0, -13]])

b = np.array([4, -7, 13, -13])

x = sust_atras(U, b)
print("Solución:", x)

print("SustAdelante")
# Ejemplo de uso:
a = np.array([[1, 0, 0, 0],
              [1,  2, 0, 0],
              [1,  -1, 5, 0],
              [1,  -1, 1, -1]])

C = np.array([1, 3, 5, 0])

x1 = sust_adelante(a, C)
print("Solución:", x1)

def triang_sup(A, b):
    # Construir la matriz aumentada
    A_ = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])
    m, n = A.shape

    # Eliminación hacia adelante (Gauss sin pivoteo)
    for k in range(m-1):           # recorrer columnas pivote
        for i in range(k+1, m):    # recorrer filas debajo del pivote
            m_ik = A_[i, k] / A_[k, k]
            for j in range(k, n+1):   # de la columna pivote hasta el final
                A_[i, j] -= m_ik * A_[k, j]

    # Extraer matriz triangular superior y vector modificado
    A_Trig = A_[:, :n]
    b_Trig = A_[:, n]

    return A_Trig, b_Trig


#Funcion eliminacion gaussiana
def elimi_gauss(A, b):
    A_, b_ = triang_sup(A, b)
    x = sust_atras(A_, b_)

    return x

def crear_matriz(n=10):
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal principal con múltiplos de 10
                A[i, j] = (i+1) * 10
            elif j < i:
                # Parte inferior izquierda: valores decrecientes
                A[i, j] = i - j
            else:
                # Parte superior derecha: valores crecientes continuos
                A[i, j] = j
    return A

def crear_vector(n=10):
    return np.arange(1, n+1)

# --- Ejemplo ---
AA = crear_matriz(10)
b = crear_vector(10)


x1 = np.linalg.solve(AA, b)
x = elimi_gauss(AA, b)
print("Resuelta con np")
print(x1)
print("Eliminacion Gaussiana")
print(x)


def Fact_LU(A):
    A = np.array(A, dtype=float)   # Convertir a float por seguridad
    m, n = A.shape
    if m != n:
        raise ValueError("La matriz A debe ser cuadrada")

    # Paso 1: U = A
    U = A.copy()

    # Paso 2: L = I
    L = np.eye(m)

    # Paso 3: Eliminación Gaussiana
    for k in range(m-1):  # k = 0, 1, ..., m-2
        for j in range(k+1, m):
            # 3.1 L[j,k] = U[j,k] / U[k,k]
            L[j, k] = U[j, k] / U[k, k]

            # 3.2 U[j, k:m] = U[j, k:m] - L[j, k] * U[k, k:m]
            U[j, k:m] = U[j, k:m] - L[j, k] * U[k, k:m]

    return L, U

def det_fact_lu(A):
    # Factorización LU (ya implementada)
    L, U = Fact_LU(A)

    # Determinante = producto de la diagonal de U
    det = np.prod(np.diag(U))
    return det

A = np.array([[2, -1, -2],
              [-4, 6, 3],
              [-4, -2, 8]], dtype=float)

L, U = Fact_LU(A)
print("Matriz A:")
print(A)
print("\nMatriz L:")
print(L)
print("\nMatriz U:")
print(U)
print("\nVerificación A ≈ L @ U:")
print(np.allclose(A, L @ U))
print(L @ U)
print("determinante a mano")
print(det_fact_lu(A))
print("determinante con np")
print(np.linalg.det(A))


import numpy as np

def thomas_general(A: np.ndarray, d: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    d = np.asarray(d, dtype=float)
    n = A.shape[0]
    if A.shape[1] != n or d.shape[0] != n:
        raise ValueError("Dimensiones incompatibles")

    # Extraer diagonales de A
    a = np.array([A[i, i-1] for i in range(1, n)])   # subdiagonal
    b = np.array([A[i, i]   for i in range(n)])     # diagonal
    c = np.array([A[i, i+1] for i in range(n-1)])   # superdiagonal

    # Vectores p y q
    p = np.zeros(n-1, dtype=float)
    q = np.zeros(n, dtype=float)

    p[0] = c[0] / b[0]
    q[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i-1]*p[i-1]
        if i < n-1:
            p[i] = c[i] / denom
        q[i] = (d[i] - a[i-1]*q[i-1]) / denom

    # Sustitución hacia atrás
    x = np.zeros(n, dtype=float)
    x[-1] = q[-1]
    for i in range(n-2, -1, -1):
        x[i] = q[i] - p[i]*x[i+1]

    return x


n = 100

# Matriz tridiagonal:
# 5 en la diagonal principal y 1 en las dos diagonales adyacentes
A = 5*np.eye(n) + np.eye(n, k=1) + np.eye(n, k=-1)

# Vector d:
# -14 en todas las posiciones, excepto la primera y la última que son -12
d = -14*np.ones(n)
d[0] = -12
d[-1] = -12


print("Thomas")
x = thomas_general(A, d)
print("Solución:", x)


def cholesky(A):
    """
    Calcula la factorización de Cholesky A = L L^T
    devolviendo la matriz L triangular inferior.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(n):
        # Elementos en la diagonal
        suma = sum(L[i, k]**2 for k in range(i))
        L[i, i] = np.sqrt(A[i, i] - suma)

        # Elementos debajo de la diagonal
        for j in range(i+1, n):
            suma = sum(L[j, k] * L[i, k] for k in range(i))
            L[j, i] = (A[j, i] - suma) / L[i, i]
    return L

def cholesky_solve(A, b):
    """
    Resuelve Ax = b usando la factorización de Cholesky,
    delegando las sustituciones a las funciones que ya tienes.
    """
    L = cholesky(A)              # Factorización: A = L L^T
    y = sust_adelante(L, b)      # Ly = b
    x = sust_atras(L.T, y)       # L^T x = y
    return x, L


# Matriz simétrica definida positiva de ejemplo
A = np.array([[4, 2, 2],
              [2, 5, 3],
              [2, 3, 6]], dtype=float)
b = np.array([12, 17, 23], dtype=float)

print("Cholesky")
x, L = cholesky_solve(A, b)
print("Matriz L:")
print(L)
print("\nSolución x:")
print(x)

# Comprobación
print("\nError ||Ax - b||:", np.linalg.norm(A @ x - b))

import numpy as np


def FactQR(A):
    A = np.array(A, dtype=float)
    m, n = A.shape  # m = número de filas, n = número de columnas
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        # Empezamos con la columna j de A
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)  # Longitud del nuevo vector ortogonal
        Q[:, j] = v / R[j, j]        # Normaliza el vector para formar la columna j de Q

    return Q, R

A = np.array([[12., -51., 4.],
              [6., 167., -68.],
              [-4., 24., -41.]])

Q, R = FactQR(A)
print("Q =\n", Q)
print("R =\n", R)
print("Reconstrucción A ≈ Q @ R:\n", Q @ R)
print("Error ||A - QR|| =", np.linalg.norm(A - Q @ R))

import numpy as np


def jacobi(A, b, x0, tol, max_iter):
    # Vector inicial
    if x0 is None:
        x = np.zeros_like(b, dtype=float)
    else:
        x = np.array(x0, dtype=float)

    # Descomposición A = D + L + U
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)

    # Matriz de iteración T y vector d
    D_inv = np.linalg.inv(D)
    T = -D_inv @ (L + U)
    d = D_inv @ b

    # Iteración
    for k in range(1, max_iter + 1):
        x_new = T @ x + d
        err = np.linalg.norm(A @ x_new - b, 2)

        if err < tol:
            return x_new, k, err
        x = x_new

    return x, max_iter, err

A = np.array([[4, -1, 0, 0],
              [-1, 4, -1, 0],
              [0, -1, 4, -1],
              [0, 0, -1, 3]], float)

b = np.array([15, 10, 10, 10], float)
x0 = np.zeros(4)

x_aprox, k, err = jacobi(A, b, x0, 1e-6, 100)
print("x ≈", x_aprox)
print("Iteraciones:", k)
print("Error final:", err)



def GaussSeidel(A, b, x0, tol, iterMax):

    x = np.array(x0, dtype=float)

    # Descomposición A = L + D + U
    D = np.diag(np.diag(A))             # diagonal
    L = np.tril(A, -1)                  # triangular inferior (sin diagonal)
    U = np.triu(A, 1)                   # triangular superior (sin diagonal)

    # M = D + L
    M = D + L

    # d = M^-1 * b
    d = np.linalg.solve(M, b)

    for k in range(iterMax):
        # y^(k) = -U x^(k)
        y = -np.dot(U, x)

        # z^(k) = M^-1 * y^(k)
        z = np.linalg.solve(M, y)

        # x^(k+1) = z + d
        x_new = z + d

        # error
        err = np.linalg.norm(np.dot(A, x_new) - b, 2)

        if err < tol:
            return x_new, k+1, err  # solucion, iteraciones, error

        x = x_new.copy()

    # si no converge dentro de iterMax
    return x, iterMax, err

A = [[4, 1, 2],
     [3, 5, 1],
     [1, 1, 3]]

b = [4, 7, 3]
x0 = [0, 0, 0]
tol = 1e-6
iterMax = 100

x, iters, err = GaussSeidel(A, b, x0, tol, iterMax)

print("Solución aproximada:", x)
print("Iteraciones:", iters)
print("Error final:", err)
