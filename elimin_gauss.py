import numpy as np

def triang_sup(A, b):
    """
    Transformar (A|b) en sistema triangular superior mediante
    eliminación Gaussiana sin pivoteo.
    """
    # Valores necesarios
    m = A.shape[0]

    # 1) Crear matriz aumentada
    Ab = np.c_[A,b]

    # 2) Eliminación
    for k in range (m-1): 
        for i in range (k+1, m):
            mik = Ab[i, k]/Ab[k,k]
            for j in range (k, m+1):
                Ab[i, j] = Ab[i,j] - mik*Ab[k,j]

    # 5 y 6) Separar A y b nuevos
    A_tri = Ab[:, :m]
    b_new = Ab[:, m]

    return A_tri, b_new


def sust_atras(A, b):
    # definimos valores necesarios m, x

    m = A.shape[0]
    x = np.zeros(m)

    # bucle i, hacia atras
    for i in range (m-1,-1,-1):
        suma = 0
        # bucle j por formula
        for j in range (i+1,m):
            suma += A[i,j] * x[j]
# termina formula
        x[i] = (b[i] - suma)/A[i,i]

    return x
        

def sist_ec_Elimin_Gauss(A, b):
    """
    Resuelve Ax=b usando eliminación Gaussiana (sin pivoteo).
    """
    # Paso 0: obtener triangular superior
    A_tri, b_new = triang_sup(A, b)

    # Paso 7: sustitución hacia atrás
    x = sust_atras(A_tri, b_new)

    return x

A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]], dtype=float)

b = np.array([8, -11, -3], dtype=float)

x = sist_ec_Elimin_Gauss(A, b)

print("Solución:", x)
print("Verificación A*x:", A @ x)
print("b:", b)
