import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


from scipy.optimize import fminbound

# 1. defines tu función
def f(x):
    return ((np.pi / 2)**4) * np.sin(x*np.pi / 2)  # pon aquí tu función

# 2. defines el intervalo donde quieres buscar
a = -1    # inicio del intervalo
b = 2     # fin del intervalo

# 3. creas muchos puntos en ese intervalo
x_vals = np.linspace(a, b, 2000)  # mientras más puntos, más preciso

# 4. evalúas la función en esos puntos
y_vals = f(x_vals)

# 5. encuentras el índice del máximo
i_max = np.argmax(y_vals)

# 6. recuperas el x y el y del máximo
x_max = x_vals[i_max]
y_max = y_vals[i_max]

print("x donde se da el máximo:", x_max)
print("máximo f(x):", y_max)


# minimizar -f para encontrar el máximo
x_max = fminbound(lambda x: -f(x), a, b)
f_max = f(x_max)

print("x_max =", x_max)
print("f_max =", f_max)

# -------------------------------------------------------
# 1. Definimos la función f(x) = log_x(arcsin x)
# -------------------------------------------------------
def f_num(x):
    return np.log(np.arcsin(x)) / np.log(x)

# nodos
x_vals = np.arange(0.1, 0.9, 0.1)   # 0.1 ... 0.8
y_vals = f_num(x_vals)
print(x_vals)

# -------------------------------------------------------
# 2. Implementación del pseudocódigo NEWTONSIMBOLICO
# -------------------------------------------------------


def newton_simbolico(x, y):
    """
    x, y: listas/arrays de longitud n+1
    Devuelve el polinomio simbólico de Newton usando sympy.Matrix
    (traducción del pseudocódigo).
    """
    n = len(x) - 1
    X = sp.Symbol('X')

    # matriz (n+1) x (n+1) de ceros
    tabla = sp.zeros(n+1, n+1)

    # Paso 1: primera columna con los y
    for i in range(n+1):
        tabla[i, 0] = sp.nsimplify(y[i])

    # Paso 1: diferencias divididas
    for j in range(1, n+1):
        for i in range(0, n - j + 1):
            num = tabla[i+1, j-1] - tabla[i, j-1]
            den = x[i+j] - x[i]
            tabla[i, j] = sp.simplify(num / den)

    # Paso 2: construir polinomio
    polinomio = tabla[0, 0]
    termino = 1
    for k in range(1, n+1):
        termino = termino * (X - x[k-1])
        polinomio = polinomio + tabla[0, k] * termino

    return sp.expand(polinomio)


# obtener polinomio simbólico
P = newton_simbolico(x_vals, y_vals)
print("Polinomio de Newton:")
print(P)

# -------------------------------------------------------
# 3. Graficar f, el polinomio y los puntos
# -------------------------------------------------------
# convertimos P(X) a función numérica
X = sp.Symbol('X')
P_lamb = sp.lambdify(X, P, 'numpy')

x_plot = np.linspace(0.1, 0.8, 400)
y_true = f_num(x_plot)
y_interp = P_lamb(x_plot)

plt.figure(figsize=(7,5))
plt.plot(x_vals, y_vals, 'o', label='Datos')
plt.plot(x_plot, y_true, label='f(x)')
plt.plot(x_plot, y_interp, '--', label='Polinomio Newton')
plt.legend()
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interpolación de Newton')
plt.show()