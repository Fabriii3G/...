import numpy as np

def sust_adelante(A, b):
    """
    Sustitución hacia adelante
    A: matriz triangular inferior (invertible)
    b: vector columna
    Retorna: solución x al sistema Ax = b
    """
    m = A.shape[0]
    x = np.zeros(m)

    for i in range(m):  # de 0 hasta m-1
        suma = 0 # define sumatoria
        for j in range(i):  # pasa por todos los elementos de la ultima fila
            # si es fila 4 por ejemplo, son i=4 elementos

            suma += A[i, j] * x[j] # misma formula 
        x[i] = (b[i] - suma) / A[i, i] # misma formula
    
    return x

# Ejemplo
A = np.array([[2, 0, 0],
              [3, 1, 0],
              [-1, 4, 5]], dtype=float)

b = np.array([2, 5, 10], dtype=float)

x = sust_adelante(A, b)
print("Solución x:", x)

# Verificación
print("Verificación A*x:", A @ x)
print("b:", b)
