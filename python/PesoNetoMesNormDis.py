import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

nombreExcel = 'excels\Impo-Franca9007.xlsx'
nombreHoja ='Peso neto'

#Graficas de distribucion normal del peso neto
# Lee el archivo Excel
df = pd.read_excel(nombreExcel, sheet_name=nombreHoja)

# Calcula el promedio de peso neto por mes
promedio_peso_neto = df['Peso neto']

media = np.mean(promedio_peso_neto)
varianza = np.var(promedio_peso_neto)
desviacion_estandar = np.std(promedio_peso_neto)
print("Valor del Peso Neto")
print("Media:", media)
print("Varianza:", varianza)
print("Desviación Estándar:", desviacion_estandar)
print("\n")

# Transformación logarítmica
peso_neto_transformado = np.log(promedio_peso_neto)

# Ajuste de la distribución normal
mu, sigma = stats.norm.fit(peso_neto_transformado)

# Crear el histograma con los datos transformados
plt.hist(peso_neto_transformado, bins=8, density=True, alpha=0.5, color="darkturquoise")

# Crear la línea de la distribución normal
x = np.linspace(peso_neto_transformado.min(), peso_neto_transformado.max(), 100)
y = stats.norm.pdf(x, mu, sigma)
plt.plot(x, y, color="darkcyan", linewidth=2)

# Configuración del gráfico
plt.xlabel("Peso Neto (Transformado)")
plt.ylabel("Densidad")
plt.title("Transformación Logarítmica y Distribución Normal del Peso Neto")
plt.legend(["Distribución Normal", "Datos Transformados"])
plt.grid(True)
plt.tight_layout()

# Mostrar el gráfico
plt.show()

"""
#Graficas de peso neto promedio por mes

# Lee el archivo Excel
df = pd.read_excel(nombreExcel, sheet_name=nombreHoja)

# Calcula el promedio de peso neto por mes
promedio_peso_neto = df.groupby('Mes')['Peso neto'].mean()

# Cálculo de la media y la desviación estándar
media = np.mean(promedio_peso_neto)
desviacion_estandar = np.std(promedio_peso_neto)

# Cálculo de los valores de densidad de probabilidad
rango_peso = np.linspace(promedio_peso_neto.min(), promedio_peso_neto.max(), 100)
densidad_probabilidad = (1 / (desviacion_estandar * np.sqrt(2 * np.pi))) * np.exp(-((rango_peso - media) ** 2) / (2 * desviacion_estandar ** 2))

# Creación del gráfico
plt.plot(rango_peso, densidad_probabilidad, label='Distribución normal')
plt.hist(promedio_peso_neto, density=True, bins=8, alpha=0.5, label='Datos observados')

# Etiquetas de los ejes y título del gráfico
plt.xlabel('Promedio Peso Neto')
plt.ylabel('Densidad de probabilidad')
plt.title('Distribución normal del Peso Neto por Mes')

# Mostrar leyenda
plt.legend()

# Mostrar gráfico
plt.show()


# Lee el archivo Excel
df = pd.read_excel(nombreExcel, sheet_name=nombreHoja)

# Calcula el promedio de peso neto por mes
promedio_peso_neto = df.groupby('Mes')['Peso neto'].mean()

media = np.mean(promedio_peso_neto)
varianza = np.var(promedio_peso_neto)
desviacion_estandar = np.std(promedio_peso_neto)
print("Valor promedio por mes del Peso Neto")
print("Media:", media)
print("Varianza:", varianza)
print("Desviación Estándar:", desviacion_estandar)


# Transformación logarítmica
peso_neto_transformado = np.log(promedio_peso_neto)

# Ajuste de la distribución normal
mu, sigma = stats.norm.fit(peso_neto_transformado)

# Crear el histograma con los datos transformados
plt.hist(peso_neto_transformado, bins=8, density=True, alpha=0.5, color="steelblue")

# Crear la línea de la distribución normal
x = np.linspace(peso_neto_transformado.min(), peso_neto_transformado.max(), 100)
y = stats.norm.pdf(x, mu, sigma)
plt.plot(x, y, color="orange", linewidth=2)

# Configuración del gráfico
plt.xlabel("Peso Neto (Transformado)")
plt.ylabel("Densidad")
plt.title("Transformación Logarítmica y Distribución Normal del Valor promedio Peso Neto por Mes")
plt.legend(["Distribución Normal", "Datos Transformados"])
plt.grid(True)
plt.tight_layout()

# Mostrar el gráfico
plt.show()
"""