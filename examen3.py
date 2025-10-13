import numpy as np
import sympy as sp
# ----------------------------------------------------------
# Función que calcula el ángulo theta
# ----------------------------------------------------------
def angulo(A, i, j):
    """
    Calcula el ángulo theta usado en la rotación de Jacobi.
    """
    if abs(A[i, i] - A[j, j]) > 1e-16:
        # Fórmula del método de Jacobi
        theta = 0.5 * np.arctan((2 * A[i, j]) / (A[i, i] - A[j, j]))
    else:
        # Si la diferencia es muy pequeña, se toma theta = 0
        theta = 0.0
    return theta


# ----------------------------------------------------------
# Función que construye la matriz de rotación G
# ----------------------------------------------------------
def matriz_rotacion(i, j, m, theta):
    """
    Construye la matriz de rotación G (Jacobi) de tamaño m x m.
    """
    G = np.eye(m)  # Matriz identidad

    # Se modifican los elementos correspondientes a i y j
    G[i, i] = np.cos(theta)
    G[j, j] = np.cos(theta)

    if i != j:
        G[i, j] = -np.sin(theta)
        G[j, i] = np.sin(theta)

    return G


# ----------------------------------------------------------
# Función principal del método de Jacobi
# ----------------------------------------------------------
def jacobi_valores_propios(A, iterMax=100, tol=1e-8):
    """
    Implementa el método de Jacobi para aproximar los valores propios de A.
    Retorna:
      xk -> vector de valores propios aproximados
      ek -> error final (norma 2 de la diferencia entre iteraciones)
    """
    A = np.array(A, dtype=float)
    m = A.shape[0]
    Ak = A.copy()
    x0 = np.diag(Ak)

    for k in range(iterMax):
        Bk = Ak.copy()

        # Recorre todos los pares (i, j)
        for i in range(m):
            for j in range(m):
                if i != j:
                    theta = angulo(Bk, i, j)
                    G = matriz_rotacion(i, j, m, theta)
                    # Transformación de Jacobi
                    Bk = G.T @ Bk @ G

        Ak = Bk
        xk = np.diag(Ak)
        ek = np.linalg.norm(xk - x0)

        if ek < tol:
            break

        x0 = xk

    return xk, ek



A = np.array([
    [4, -2, 2],
    [-2, 2, -4],
    [2, -4, 11]
], dtype=float)

valores_propios, error = jacobi_valores_propios(A, iterMax=100, tol=1e-8)

print("Valores propios aproximados:")
print(valores_propios)
print(f"Error final: {error:.2e}")
print(np.linalg.eig(A))



# ======================================================
# 1) Recurrencia de Sturm
# ======================================================
def sturm(T):
    """
    Calcula el polinomio característico p_m(λ) de una matriz tridiagonal simétrica T
    usando la Recurrencia de Sturm.
    Retorna: (p_m(λ), λ)
    """
    T = np.array(T, dtype=float)
    m = T.shape[0]
    lam = sp.Symbol('lambda')

    # Extraer diagonales
    alpha = [T[i, i] for i in range(m)]
    beta = [T[i, i+1] for i in range(m-1)] if m > 1 else []

    # Recurrencia de Sturm
    p_prev2 = sp.Integer(1)
    p_prev1 = alpha[0] - lam
    for k in range(2, m+1):
        pk = (alpha[k-1] - lam) * p_prev1 - (beta[k-2]**2) * p_prev2
        p_prev2, p_prev1 = p_prev1, sp.expand(pk)

    return sp.expand(p_prev1), lam


# ======================================================
# 2) Intervalos de Gershgorin
# ======================================================
def gershgorin(T):
    """
    Calcula los intervalos de Gershgorin para una matriz tridiagonal simétrica T.
    Retorna: (intervalos, (a, b))
    """
    T = np.array(T, dtype=float)
    m = T.shape[0]
    alpha = [T[i, i] for i in range(m)]
    beta = [T[i, i+1] for i in range(m-1)] if m > 1 else []

    intervalos = []
    for j in range(m):
        Bj = abs(beta[j]) if j < m-1 else 0.0
        Bj_1 = abs(beta[j-1]) if j > 0 else 0.0
        Rj = Bj + Bj_1
        intervalos.append((alpha[j] - Rj, alpha[j] + Rj))

    min_all = min(a for a, _ in intervalos)
    max_all = max(b for _, b in intervalos)
    return intervalos, (min_all, max_all)


# ======================================================
# 3) Método de Falsa Posición
# ======================================================
def falsa_posicion(p, lam_symbol, a, b, tol=1e-8, iterMax=100):
    """
    Aplica el método de falsa posición para encontrar un cero de p(λ) en [a, b].
    """
    f = sp.lambdify(lam_symbol, p, 'numpy')
    fa, fb = float(f(a)), float(f(b))

    if fa == 0.0:
        return float(a)
    if fb == 0.0:
        return float(b)
    if fa * fb > 0:
        return None

    ca, cb = a, b
    fa_c, fb_c = fa, fb
    for _ in range(iterMax):
        c = (ca * fb_c - cb * fa_c) / (fb_c - fa_c)
        fc = float(f(c))
        if abs(fc) < tol:
            return float(c)
        if fa_c * fc < 0:
            cb, fb_c = c, fc
        else:
            ca, fa_c = c, fc
        if abs(cb - ca) < tol:
            return float((ca + cb) / 2)
    return float((ca + cb) / 2)


# ======================================================
# 4) FUNCIÓN PRINCIPAL — Combinación de las tres
# ======================================================
def valores_propios_tridiagonal(T, tol=1e-8, iterMax=200, N=300):
    """
    Calcula los valores propios de una matriz tridiagonal simétrica T
    combinando:
      - Recurrencia de Sturm
      - Intervalos de Gershgorin
      - Falsa Posición
    """
    # Obtener polinomio característico y variable simbólica
    p, lam = sturm(T)

    # Calcular el rango global con Gershgorin
    _, (a, b) = gershgorin(T)

    # Evaluar el polinomio en una malla del intervalo [a,b]
    xs = np.linspace(a, b, N + 1)
    fnum = sp.lambdify(lam, p, 'numpy')

    raices = []
    for i in range(N):
        x0, x1 = xs[i], xs[i + 1]
        y0, y1 = float(fnum(x0)), float(fnum(x1))
        if y0 * y1 < 0:
            r = falsa_posicion(p, lam, x0, x1, tol, iterMax)
            if r is not None:
                raices.append(r)
        elif y0 == 0:
            raices.append(float(x0))
        elif y1 == 0:
            raices.append(float(x1))

    # Eliminar duplicados cercanos
    raices = sorted(raices)
    valores_propios = []
    for r in raices:
        if not valores_propios or abs(r - valores_propios[-1]) > 1e-7:
            valores_propios.append(r)

    return valores_propios



T = np.array([
    [4, 1, 0],
    [1, 2, 2],
    [0, 2, 3]
], dtype=float)

print(np.linalg.eig(T))
vp = valores_propios_tridiagonal(T, tol=1e-10)
print("Valores propios aproximados:")
for v in vp:
    print(f"{v:.12f}")


