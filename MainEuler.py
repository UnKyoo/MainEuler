#   Codigo que implementa el metodo de Euler
#   para resolver una ecuacion diferencial
#   
#           Autor:
#   Gilbert Alexander Mendez Cervera
#   mendezgilbert222304@outlook.com
#   Version 1.01 : 25/04/2025
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Definición de la EDO del capacitor: dV/dt = (1 / RC) * (V_fuente - V)
def f(t, V):
    R = 1000           # Resistencia en ohmios (Ω)
    C = 0.001          # Capacitancia en faradios (F)
    V_fuente = 5       # Voltaje de la fuente en voltios (V)
    return (1 / (R * C)) * (V_fuente - V)

# Condiciones iniciales y parámetros
x0 = 0      # Tiempo inicial (s)
y0 = 0      # Voltaje inicial (V)
xf = 5      # Tiempo final (s)
n = 20      # Número de pasos

h = (xf - x0) / n  # Tamaño del paso (Δt)

# Inicialización de listas
x_vals = [x0]         # Lista de tiempos
y_vals = [y0]         # Lista de voltajes aproximados
y_exact = [0]         # Lista de voltajes exactos
errores = [0]         # Lista de errores absolutos

# Valores constantes para la solución exacta
R = 1000
C = 0.001
V_fuente = 5

# Método de Euler + comparación con la solución exacta
x = x0
y = y0
for i in range(n):
    y = y + h * f(x, y)            # Euler: nuevo valor de voltaje
    x = x + h                      # Avanza en el tiempo
    y_real = V_fuente * (1 - np.exp(-x / (R * C)))  # Solución analítica
    error = abs(y_real - y)        # Error absoluto
    
    # Guardar valores
    x_vals.append(x)
    y_vals.append(y)
    y_exact.append(y_real)
    errores.append(error)

# Guardar resultados en archivo CSV
df = pd.DataFrame({
    "Tiempo (s)": x_vals,
    "V_aproximada (V)": y_vals,
    "V_exacta (V)": y_exact,
    "Error_absoluto (V)": errores
})
df.to_csv("carga_capacitor_comparacion.csv", index=False)

# Gráfica
plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals, 'o-', label='Solución aproximada (Euler)', color='blue')
plt.plot(x_vals, y_exact, '-', label='Solución exacta', color='red')
plt.title('Carga de un Capacitor - Comparación Euler vs Exacta')
plt.xlabel('Tiempo (s)')
plt.ylabel('Voltaje (V)')
plt.legend()
plt.grid(True)
plt.savefig("carga_capacitor_comparacion.png")
plt.show()


"""EJERCICIO #2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Definición de la EDO: dv/dt = g - (k/m) * v
def f(x, y):
    g = 9.81      # Aceleración de la gravedad (m/s²)
    m = 2         # Masa del objeto (kg)
    k = 0.5       # Coeficiente de fricción (kg/s)
    return g - (k / m) * y

# Condiciones iniciales
x0 = 0     # Tiempo inicial (s)
y0 = 0     # Velocidad inicial (v(0) = 0)
xf = 10    # Tiempo final (s)
n = 50     # Número de pasos

# Paso de integración
h = (xf - x0) / n  # Tamaño de paso (Δt)

# Inicialización de listas
x_vals = [x0]      # Lista de tiempos
y_vals = [y0]      # Lista de velocidades aproximadas

# Método de Euler
x = x0
y = y0
for i in range(n):
    y = y + h * f(x, y)  # Estimación de la velocidad
    x = x + h            # Avance en el tiempo
    x_vals.append(x)
    y_vals.append(y)

# Solución exacta para comparar
g = 9.81
m = 2
k = 0.5
x_exact = np.linspace(x0, xf, 200)
y_exact = (m * g / k) * (1 - np.exp(-(k / m) * x_exact))

# Guardar resultados en CSV
data = {
    "Tiempo (s)": x_vals,
    "v_aproximada (m/s)": y_vals
}
df = pd.DataFrame(data)
df.to_csv("caida_libre_euler.csv", index=False)

# Gráfica
plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals, 'o-', label='Euler (aproximado)', color='blue')
plt.plot(x_exact, y_exact, '-', label='Solución exacta', color='red')
plt.title('Caída Libre con Resistencia del Aire - Método de Euler')
plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad (m/s)')
plt.grid(True)
plt.legend()
plt.savefig("caida_libre_euler.png")
plt.show()

"""

"""EJERCICIO #3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parámetros del problema
T0 = 90           # Temperatura inicial (°C)
Tamb = 25         # Temperatura ambiente (°C)
k = 0.07          # Constante de enfriamiento
t0 = 0            # Tiempo inicial (minutos)
tf = 30           # Tiempo final (minutos)
n = 30            # Número de pasos

# Paso de tiempo
h = (tf - t0) / n

# Inicializar listas
t_vals = [t0]
T_euler = [T0]
T_exacta = [T0]  # La inicial es la misma

# Método de Euler
t = t0
T = T0
for i in range(n):
    T = T + h * (-k * (T - Tamb))
    t = t + h
    T_euler.append(T)
    t_vals.append(t)
    # Solución exacta en ese t
    T_ex = Tamb + (T0 - Tamb) * np.exp(-k * t)
    T_exacta.append(T_ex)

# Guardar en CSV
df = pd.DataFrame({
    "Tiempo (min)": t_vals,
    "T (Euler)": T_euler,
    "T (Exacta)": T_exacta
})
df.to_csv("enfriamiento_euler_vs_exacta.csv", index=False)

# Graficar resultados
plt.figure(figsize=(10, 6))
plt.plot(t_vals, T_euler, 'o-', label="Euler", color='blue')
plt.plot(t_vals, T_exacta, '--', label="Exacta", color='red')
plt.title("Enfriamiento de un cuerpo - Método de Euler vs Exacta")
plt.xlabel("Tiempo (min)")
plt.ylabel("Temperatura (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("grafica_enfriamiento.png")
plt.show()

"""