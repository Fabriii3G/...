import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Definición del problema (mismo ejemplo que en tu script)
# -----------------------------------------------------------
# EDO: y' = (x + y) / x
f  = lambda x, y: (x + y) / x               # y' = f(x,y)
fx = lambda x, y: 1/x - (x + y)/x**2        # ∂f/∂x
fy = lambda x, y: 1/x                       # ∂f/∂y
a = 2                                       # Límite inferior
b = 10                                      # Límite superior
y0 = 4                                      # Condición inicial
n  = 6                                      # Número de pasos


# -----------------------------------------------------------
# Nombre: metodo_euler
# -----------------------------------------------------------
def metodo_euler(f, a, b, y0, n):
    """
    Nombre: metodo_euler
    Entrada:
      f  -> función de la EDO (y' = f(x,y))
      a  -> límite inferior del intervalo
      b  -> límite superior del intervalo
      y0 -> condición inicial
      n  -> número de pasos
    Salida:
      x  -> vector de puntos en el eje x
      y  -> vector de soluciones aproximadas
    """
    h = (b - a) / n
    x = np.arange(n + 1, dtype=float) * h + a
    y = np.zeros(n + 1, dtype=float)
    y[0] = y0
    for k in range(n):
        y[k+1] = y[k] + h * f(x[k], y[k])
    return x, y


# -----------------------------------------------------------
# Nombre: heun (Euler mejorado)
# -----------------------------------------------------------
def heun(f, a, b, y0, n):
    """
    Nombre: heun (Método de Heun o Euler mejorado)
    """
    h = (b - a) / n
    x = np.arange(n + 1, dtype=float) * h + a
    y = np.zeros(n + 1, dtype=float)
    y[0] = y0
    for k in range(n):
        K1 = f(x[k], y[k])
        K2 = f(x[k] + h, y[k] + h * K1)
        y[k+1] = y[k] + (h/2.0) * (K1 + K2)
    return x, y


# -----------------------------------------------------------
# Nombre: rk2
# -----------------------------------------------------------
def rk2(f, a, b, y0, n):
    """
    Nombre: rk2 (Método de Runge-Kutta de segundo orden)
    """
    h = (b - a) / n
    x = np.arange(n + 1, dtype=float) * h + a
    y = np.zeros(n + 1, dtype=float)
    y[0] = y0
    for k in range(n):
        K1 = f(x[k], y[k])
        K2 = f(x[k] + h/2.0, y[k] + (h/2.0)*K1)
        y[k+1] = y[k] + h * K2
    return x, y


# -----------------------------------------------------------
# Nombre: rk3
# -----------------------------------------------------------
def rk3(f, a, b, y0, n):
    """
    Nombre: rk3 (Método de Runge-Kutta de tercer orden)
    """
    h = (b - a) / n
    x = np.arange(n + 1, dtype=float) * h + a
    y = np.zeros(n + 1, dtype=float)
    y[0] = y0
    for k in range(n):
        K1 = f(x[k], y[k])
        K2 = f(x[k] + h/2.0, y[k] + (h/2.0)*K1)
        K3 = f(x[k] + h, y[k] - h*K1 + 2*h*K2)
        y[k+1] = y[k] + (h/6.0) * (K1 + 4*K2 + K3)
    return x, y


# -----------------------------------------------------------
# Nombre: rk4
# -----------------------------------------------------------
def rk4(f, a, b, y0, n):
    """
    Nombre: rk4 (Método de Runge-Kutta de cuarto orden)
    """
    h = (b - a) / n
    x = np.arange(n + 1, dtype=float) * h + a
    y = np.zeros(n + 1, dtype=float)
    y[0] = y0
    for k in range(n):
        K1 = f(x[k], y[k])
        K2 = f(x[k] + h/2.0, y[k] + (h/2.0)*K1)
        K3 = f(x[k] + h/2.0, y[k] + (h/2.0)*K2)
        K4 = f(x[k] + h, y[k] + h*K3)
        y[k+1] = y[k] + (h/6.0) * (K1 + 2*K2 + 2*K3 + K4)
    return x, y


# -----------------------------------------------------------
# Nombre: taylor2
# -----------------------------------------------------------
def taylor2(f, fx, fy, a, b, y0, n):
    """
    Nombre: taylor2 (Método de Taylor de segundo orden)
    Entrada:
      f, fx=∂f/∂x, fy=∂f/∂y
    """
    h = (b - a) / n
    x = np.arange(n + 1, dtype=float) * h + a
    y = np.zeros(n + 1, dtype=float)
    y[0] = y0
    for k in range(n):
        fxy = fx(x[k], y[k]) + fy(x[k], y[k]) * f(x[k], y[k])
        y[k+1] = y[k] + h*f(x[k], y[k]) + (h**2 / 2.0) * fxy
    return x, y


# -----------------------------------------------------------
# Nombre: ab2 (Adams-Bashforth 2 pasos)
# -----------------------------------------------------------
def ab2(f, a, b, y0, n):
    """
    Nombre: ab2 (Método de Adams-Bashforth de 2 pasos)
    """
    h = (b - a) / n
    x = np.arange(n + 1, dtype=float) * h + a
    y = np.zeros(n + 1, dtype=float)
    y[0] = y0

    # Paso inicial con RK4 (idéntico a tu script)
    K1 = f(x[0], y[0])
    K2 = f(x[0] + h/2.0, y[0] + (h/2.0)*K1)
    K3 = f(x[0] + h/2.0, y[0] + (h/2.0)*K2)
    K4 = f(x[0] + h, y[0] + h*K3)
    y[1] = y[0] + (h/6.0) * (K1 + 2*K2 + 2*K3 + K4)

    for k in range(1, n):
        f_k   = f(x[k],   y[k])
        f_k_1 = f(x[k-1], y[k-1])
        y[k+1] = y[k] + (h/2.0) * (3*f_k - f_k_1)
    return x, y


# -----------------------------------------------------------
# Nombre: ab3 (Adams-Bashforth 3 pasos)
# -----------------------------------------------------------
def ab3(f, a, b, y0, n):
    """
    Nombre: ab3 (Método de Adams-Bashforth de 3 pasos)
    """
    h = (b - a) / n
    x = np.arange(n + 1, dtype=float) * h + a
    y = np.zeros(n + 1, dtype=float)
    y[0] = y0

    # Dos pasos iniciales con RK4 (idéntico a tu script)
    for k in range(0, 2):
        K1 = f(x[k], y[k])
        K2 = f(x[k] + h/2.0, y[k] + (h/2.0)*K1)
        K3 = f(x[k] + h/2.0, y[k] + (h/2.0)*K2)
        K4 = f(x[k] + h, y[k] + h*K3)
        y[k+1] = y[k] + (h/6.0) * (K1 + 2*K2 + 2*K3 + K4)

    for k in range(2, n):
        f_k   = f(x[k],   y[k])
        f_k_1 = f(x[k-1], y[k-1])
        f_k_2 = f(x[k-2], y[k-2])
        y[k+1] = y[k] + (h/12.0) * (23*f_k - 16*f_k_1 + 5*f_k_2)
    return x, y


# -----------------------------------------------------------
# Nombre: ab4 (Adams-Bashforth 4 pasos)
# -----------------------------------------------------------
def ab4(f, a, b, y0, n):
    """
    Nombre: ab4 (Método de Adams-Bashforth de 4 pasos)
    """
    h = (b - a) / n
    x = np.arange(n + 1, dtype=float) * h + a
    y = np.zeros(n + 1, dtype=float)
    y[0] = y0

    # Tres pasos iniciales con RK4 (idéntico a tu script)
    for k in range(0, 3):
        K1 = f(x[k], y[k])
        K2 = f(x[k] + h/2.0, y[k] + (h/2.0)*K1)
        K3 = f(x[k] + h/2.0, y[k] + (h/2.0)*K2)
        K4 = f(x[k] + h, y[k] + h*K3)
        y[k+1] = y[k] + (h/6.0) * (K1 + 2*K2 + 2*K3 + K4)

    for k in range(3, n):
        f_k   = f(x[k],   y[k])
        f_k_1 = f(x[k-1], y[k-1])
        f_k_2 = f(x[k-2], y[k-2])
        f_k_3 = f(x[k-3], y[k-3])
        y[k+1] = y[k] + (h/24.0) * (55*f_k - 59*f_k_1 + 37*f_k_2 - 9*f_k_3)
    return x, y


# -----------------------------------------------------------
# Solución exacta (misma que en tu script)
# -----------------------------------------------------------
y_exact = lambda x: (np.log(x) - np.log(2) + 2) * x


if __name__ == "__main__":
    # -----------------------------------------------------------
    # Cálculo numérico con los distintos métodos
    # -----------------------------------------------------------
    x_eu,  y_eu  = metodo_euler(f, a, b, y0, n)
    x_he,  y_he  = heun(f, a, b, y0, n)
    x_r2,  y_r2  = rk2(f, a, b, y0, n)
    x_r3,  y_r3  = rk3(f, a, b, y0, n)
    x_r4,  y_r4  = rk4(f, a, b, y0, n)
    x_ty,  y_ty  = taylor2(f, fx, fy, a, b, y0, n)
    x_ab2, y_ab2 = ab2(f, a, b, y0, n)
    x_ab3, y_ab3 = ab3(f, a, b, y0, n)
    x_ab4, y_ab4 = ab4(f, a, b, y0, n)

    # -----------------------------------------------------------
    # Solución exacta evaluada en los mismos puntos
    # -----------------------------------------------------------
    y_ex = y_exact(x_eu)

    # -----------------------------------------------------------
    # Gráfica comparativa (mismos estilos)
    # -----------------------------------------------------------
    plt.figure()
    plt.plot(x_eu,  y_ex,  'k-',  linewidth=2,  label='Solución exacta')
    plt.plot(x_eu,  y_eu,  'r--', linewidth=1.2, label='Euler')
    plt.plot(x_he,  y_he,  'b--', linewidth=1.2, label='Heun')
    plt.plot(x_r2,  y_r2,  'g--', linewidth=1.2, label='RK2')
    plt.plot(x_r3,  y_r3,  'm--', linewidth=1.2, label='RK3')
    plt.plot(x_r4,  y_r4,  'c--', linewidth=1.2, label='RK4')
    plt.plot(x_ty,  y_ty,  'y--', linewidth=1.2, label='Taylor 2°')
    plt.plot(x_ab2, y_ab2, color=(0.5, 0.2, 0.8), linestyle='--', linewidth=1.2, label='Adams-Bashforth 2')
    plt.plot(x_ab3, y_ab3, color=(0.2, 0.6, 0.8), linestyle='--', linewidth=1.2, label='Adams-Bashforth 3')
    plt.plot(x_ab4, y_ab4, color=(0.8, 0.4, 0.2), linestyle='--', linewidth=1.2, label='Adams-Bashforth 4')

    plt.legend(loc='upper left')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Comparación de métodos numéricos para el problema de Cauchy')
    plt.grid(True)
    plt.show()
