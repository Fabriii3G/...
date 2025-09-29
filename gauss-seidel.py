import numpy as np

def sust_atras(U, b):
    """Sustitución hacia atrás para sistemas triangulares superiores Ux=b."""
    m = U.shape[0]
    x = np.zeros_like(b, dtype=float)
    for i in reversed(range(m)):
        x[i] = (b[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

def sol_GaussSeidel(A, b, tol=1e-10, iterMax=1000):
    """
    Gauss-Seidel para diagonal dominante
    """
    m = len(b)
    xk = np.zeros(m, dtype=float)

    # Paso 1: separar L, D, U (sin poner negativos)
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)      # triangular inferior estricta
    U = np.triu(A, 1)       # triangular superior estricta

    # Paso 2: M = D + U (triangular superior)
    M = D + U

    # Paso 3: d = M^-1 b → M d = b
    d = sust_atras(M, b)

    for k in range(iterMax):
        # Paso 5: yk = -L * xk
        yk = -L @ xk

        # Paso 6: zk = M^-1 * yk → M zk = yk
        zk = sust_atras(M, yk)

        # Paso 7: xk1 = zk + d
        xk1 = zk + d

        # Paso 6.2: error
        erk = np.linalg.norm(A @ xk1 - b, 2)

        # Paso 6.3: criterio de parada
        if erk < tol:
            return xk1, erk, k+1

        xk = xk1

    return xk, erk, iterMax



# =============================
# Ejemplo de uso

A = np.array([[4, -1, 0, 0],
              [-1, 4, -1, 0],
              [0, -1, 4, -1],
              [0, 0, -1, 3]], dtype=float)

b = np.array([15, 10, 10, 10], dtype=float)

xk, erk, k = sol_GaussSeidel(A, b)

print("Solución aproximada:", xk)
print("Error final:", erk)
print("Iteraciones:", k)
print("Solución exacta:", np.linalg.solve(A, b))

