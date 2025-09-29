import numpy as np

def sol_Jacobi(A, b, tol=1e-10, iterMax=1000):
    """
    Método de Jacobi para resolver Ax=b

    Entradas:
      A : matriz cuadrada
      b : vector
      tol : tolerancia
      iterMax : iteraciones máximas

    Salidas:
      xk : aproximación de la solución
      erk : error al detenerse
      k : número de iteraciones realizadas
    """
    m = len(b)
    xk = np.zeros(m)  # Vector inicial (x0)

    # Paso 1: D = diag(diag(A))
    D = np.diag(np.diag(A))

    # Paso 2: D^-1
    Dinv = np.diag(1 / np.diag(A))

    # Paso 3: L+U = A - D
    LmU = A - D

    # Paso 4: d = D^-1 b
    d = Dinv @ b

    # Paso 5: T = -D^-1 (L+U)
    T = -Dinv @ LmU

    # Iterar
    for k in range(iterMax):
        # Paso 6.1: x(k+1) = T xk + d
        xk1 = T @ xk + d

        # Paso 6.2: error
        erk = np.linalg.norm(A @ xk1 - b, 2)

        # Paso 6.3: criterio de parada
        if erk < tol:
            xk = xk1
            return xk, erk, k+1

        # actualizar
        xk = xk1

    # Si no converge en iterMax
    return xk, erk, iterMax

# ======================================
# Ejemplo de uso

A = np.array([[10, -1, 2, 0],
              [-1, 11, -1, 3],
              [2, -1, 10, -1],
              [0, 3, -1, 8]], dtype=float)

b = np.array([6, 25, -11, 15], dtype=float)

xk, erk, k = sol_Jacobi(A, b)

print("Solución aproximada:", xk)
print("Error final:", erk)
print("Iteraciones:", k)

# Comparar con solución exacta
print("Solución exacta:", np.linalg.solve(A, b))
