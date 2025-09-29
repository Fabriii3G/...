import numpy as np

def sust_atras(A, b):
    """
    Sustitución hacia atrás
    A: matriz triangular superior (invertible)
    b: vector columna
    Retorna: solución x al sistema Ax = b
    """
    m = A.shape[0]
    x = np.zeros(m)

    for i in range(m - 1, -1, -1):  # desde m-1 hasta 0, -1 al final es para ir al reves
        suma = 0 # define sumatoria
        for j in range(i + 1, m): # inicia sumatoria
            suma += A[i, j] * x[j] # realiza sumatoria

        x[i] = (b[i] - suma) / A[i, i] # calcula cada elemento del vector x
    
    return x

# Ejemplo
A = np.array([[2, -1, 1],
              [0, 3, -2],
              [0, 0, 1]], dtype=float)

b = np.array([1, 4, -2], dtype=float)

x = sust_atras(A, b)
print("Solución x:", x)

# Verificación
print("Verificación A*x:", A @ x)
print("b:", b)
