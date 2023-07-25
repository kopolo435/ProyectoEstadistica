import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

nombreExcel = 'excels\Impo-Franca9007.xlsx'
nombreHoja = 'Peso bruto'

# Graficas de distribucion normal del peso bruto
# Lee el archivo Excel
df = pd.read_excel(nombreExcel, sheet_name=nombreHoja)

# Calcula el promedio de peso bruto por mes
promedio_peso_bruto = df['Peso bruto']

media = np.mean(promedio_peso_bruto)
varianza = np.var(promedio_peso_bruto)
desviacion_estandar = np.std(promedio_peso_bruto)
print("Valor del Peso bruto")
print("Media:", media)
print("Varianza:", varianza)
print("Desviación Estándar:", desviacion_estandar)
print("\n")

# Transformación logarítmica
peso_bruto_transformado = np.log(promedio_peso_bruto)

# Ajuste de la distribución normal
mu, sigma = stats.norm.fit(peso_bruto_transformado)

# Crear el histograma con los datos transformados
hist_values, bin_edges, _ = plt.hist(peso_bruto_transformado, bins=8, density=True, alpha=0.5, color="darkturquoise")

# Ajustar la amplitud de la curva normal para que coincida con el histograma
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
curve_values = stats.norm.pdf(bin_centers, mu, sigma)
area_total_histograma = np.sum(hist_values * np.diff(bin_edges))
area_total_curva = np.sum(curve_values * (bin_edges[1:] - bin_edges[:-1]))
ajuste_amplitud = area_total_histograma / area_total_curva
curve_values *= ajuste_amplitud

# Crear la línea de la distribución normal
x = np.linspace(peso_bruto_transformado.min(), peso_bruto_transformado.max(), 100)
y = stats.norm.pdf(x, mu, sigma)

# Configuración del gráfico
plt.plot(x, y, color="darkcyan", linewidth=2)
plt.xlabel("Peso bruto (Transformado)")
plt.ylabel("Densidad")
plt.title("Transformación Logarítmica y Distribución Normal del Peso bruto")
plt.legend(["Distribución Normal", "Datos Transformados"])
plt.grid(True)
plt.tight_layout()

# Mostrar el gráfico
plt.show()


# Graficas de peso bruto promedio por mes

# Lee el archivo Excel
df = pd.read_excel(nombreExcel, sheet_name=nombreHoja)

# Calcula el promedio de peso bruto por mes
promedio_peso_bruto = df.groupby('Mes')['Peso bruto'].mean()

# Cálculo de la media y la desviación estándar
media = np.mean(promedio_peso_bruto)
desviacion_estandar = np.std(promedio_peso_bruto)

# Cálculo de los valores de densidad de probabilidad
rango_peso = np.linspace(promedio_peso_bruto.min(), promedio_peso_bruto.max(), 100)
densidad_probabilidad = (1 / (desviacion_estandar * np.sqrt(2 * np.pi))) * np.exp(-((rango_peso - media) ** 2) / (2 * desviacion_estandar ** 2))

# Creación del gráfico
plt.plot(rango_peso, densidad_probabilidad, label='Distribución normal')
plt.hist(promedio_peso_bruto, density=True, bins=8, alpha=0.5, label='Datos observados')

# Etiquetas de los ejes y título del gráfico
plt.xlabel('Promedio Peso bruto')
plt.ylabel('Densidad de probabilidad')
plt.title('Distribución normal del Peso bruto por Mes')

# Mostrar leyenda
plt.legend()

# Mostrar gráfico
plt.show()
