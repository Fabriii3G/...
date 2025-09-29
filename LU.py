import numpy as np

def fact_LU(A):
    """
    Factorización LU sin pivoteo.
    Devuelve L (triangular inferior) y U (triangular superior).

    triang superior de eliminación gaussiana modificado
    """
    m = A.shape[0]
    U = A.copy().astype(float)   # empezamos con A como base para U
    L = np.eye(m)                # L empieza como identidad

    # Bucle principal
    for k in range(m-1):         # recorre columnas
        for i in range(k+1, m):  # recorre filas debajo del pivote
            # 3.1: obtener L[j,k]
            L[i, k] = U[i, k] / U[k, k] # Cada L son los factores 
            # 3.2: actualizar fila j de U
            for j in range(k, m):   # recorre columnas desde k
                U[i, j] = U[i, j] - L[i, k] * U[k, j]

    return L, U


# ===========================
# Ejemplo de uso

A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]], dtype=float)

L, U = fact_LU(A)

print("Matriz A original:\n", A)
print("L:\n", L)
print("U:\n", U)

# =========================
# Funciones auxiliares
# =========================

def sust_adelante(L, b):
    """Resuelve Ly = b por sustitución hacia adelante."""
    m = L.shape[0]
    y = np.zeros_like(b, dtype=float)
    for i in range(m):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y

def sust_atras(U, y):
    """Resuelve Ux = y por sustitución hacia atrás."""
    m = U.shape[0]
    x = np.zeros_like(y, dtype=float)
    for i in reversed(range(m)):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x


# =========================
# Función principal
# =========================
def sist_ec_LU(A, b):
    """
    Resuelve Ax = b usando factorización LU.
    """

    L, U = fact_LU(A)
    y = sust_adelante(L, b)
    x = sust_atras(U, y)
    return x


# =========================
# Ejemplo de uso
# =========================

A = np.array([[2, -1, 1],
                  [3,  3, 9],
                  [3,  3,  5]], dtype=float)
b = np.array([2, -1, 4], dtype=float)

x = sist_ec_LU(A, b)
print("Solución x:", x)

# Verificación
print("A·x =", A @ x)
print("b   =", b)